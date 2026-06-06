"""
forecaster.py
Multiple forecasting models with scenario analysis and KPI calculations.
Models: Linear Trend, Exponential Smoothing, ARIMA, Random Forest
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


# ------------------------------------------------------------------ #
#  Data classes                                                        #
# ------------------------------------------------------------------ #
@dataclass
class ForecastResult:
    model_name:    str
    forecast:      pd.Series           # future predictions
    confidence_lo: pd.Series           # lower 80% CI
    confidence_hi: pd.Series           # upper 80% CI
    fitted:        pd.Series           # in-sample fitted values
    mape:          float               # mean absolute % error
    rmse:          float
    params:        dict = field(default_factory=dict)


@dataclass
class ScenarioBundle:
    base:  ForecastResult
    bull:  ForecastResult
    bear:  ForecastResult
    col_name: str
    historical: pd.Series


# ------------------------------------------------------------------ #
#  Individual models                                                   #
# ------------------------------------------------------------------ #
class LinearTrendModel:
    name = "Linear Trend"

    def fit_predict(self, series: pd.Series, periods: int) -> ForecastResult:
        y    = series.values.astype(float)
        X    = np.arange(len(y)).reshape(-1, 1)
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)

        model = LinearRegression()
        model.fit(X_sc, y)

        fitted     = pd.Series(model.predict(X_sc), index=series.index)
        future_X   = np.arange(len(y), len(y) + periods).reshape(-1, 1)
        future_X_sc = scaler.transform(future_X)
        forecast   = model.predict(future_X_sc)

        residuals  = y - fitted.values
        std_resid  = np.std(residuals)
        z          = 1.28  # 80% CI

        future_idx = self._future_index(series, periods)
        mape, rmse = self._errors(y, fitted.values)

        return ForecastResult(
            model_name    = self.name,
            forecast      = pd.Series(forecast, index=future_idx),
            confidence_lo = pd.Series(forecast - z * std_resid, index=future_idx),
            confidence_hi = pd.Series(forecast + z * std_resid, index=future_idx),
            fitted        = fitted,
            mape          = mape,
            rmse          = rmse,
            params        = {"slope": float(model.coef_[0]),
                             "intercept": float(model.intercept_)},
        )

    @staticmethod
    def _future_index(series, periods):
        if isinstance(series.index, pd.DatetimeIndex):
            freq = pd.infer_freq(series.index) or "QS"
            return pd.date_range(series.index[-1], periods=periods + 1, freq=freq)[1:]
        return range(series.index[-1] + 1, series.index[-1] + 1 + periods)

    @staticmethod
    def _errors(actual, predicted):
        mask = actual != 0
        mape = float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask]))) * 100
        rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
        return round(mape, 2), round(rmse, 2)


class ExponentialSmoothingModel:
    name = "Exponential Smoothing (Holt-Winters)"

    def fit_predict(self, series: pd.Series, periods: int) -> ForecastResult:
        y = series.values.astype(float)

        try:
            trend = "add" if len(y) >= 4 else None
            seasonal = "add" if len(y) >= 8 else None
            seasonal_periods = 4 if seasonal and len(y) >= 8 else None

            model  = ExponentialSmoothing(
                y, trend=trend, seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                initialization_method="estimated"
            )
            result = model.fit(optimized=True, remove_bias=True)
        except Exception:
            model  = ExponentialSmoothing(y, trend="add", initialization_method="estimated")
            result = model.fit(optimized=True)

        fitted    = pd.Series(result.fittedvalues, index=series.index)
        forecast  = result.forecast(periods)
        sim       = result.simulate(periods, repetitions=200, random_errors="bootstrap")
        lo = np.percentile(sim, 10, axis=1)
        hi = np.percentile(sim, 90, axis=1)

        future_idx = LinearTrendModel._future_index(series, periods)
        actual     = y
        mape, rmse = LinearTrendModel._errors(actual, fitted.values)

        return ForecastResult(
            model_name    = self.name,
            forecast      = pd.Series(forecast, index=future_idx),
            confidence_lo = pd.Series(lo, index=future_idx),
            confidence_hi = pd.Series(hi, index=future_idx),
            fitted        = fitted,
            mape          = mape,
            rmse          = rmse,
        )


class ARIMAModel:
    name = "ARIMA"

    def fit_predict(self, series: pd.Series, periods: int) -> ForecastResult:
        y = series.values.astype(float)
        d = self._differencing_order(y)
        order = self._auto_order(y, d)

        try:
            model  = ARIMA(y, order=order)
            result = model.fit()
        except Exception:
            model  = ARIMA(y, order=(1, 1, 0))
            result = model.fit()

        fitted   = pd.Series(np.array(result.fittedvalues), index=series.index)
        forecast_res = result.get_forecast(steps=periods)
        fc_mean  = forecast_res.predicted_mean
        fc_ci    = forecast_res.conf_int(alpha=0.2)

        future_idx = LinearTrendModel._future_index(series, periods)
        mape, rmse = LinearTrendModel._errors(y, fitted.values)

        return ForecastResult(
            model_name    = self.name,
            forecast      = pd.Series(fc_mean, index=future_idx),
            confidence_lo = pd.Series(np.array(fc_ci)[:, 0], index=future_idx),
            confidence_hi = pd.Series(np.array(fc_ci)[:, 1], index=future_idx),
            fitted        = fitted,
            mape          = mape,
            rmse          = rmse,
            params        = {"order": order},
        )

    @staticmethod
    def _differencing_order(y):
        try:
            p = adfuller(y)[1]
            return 0 if p < 0.05 else 1
        except Exception:
            return 1

    @staticmethod
    def _auto_order(y, d):
        best, best_aic = (1, d, 1), np.inf
        for p in range(3):
            for q in range(3):
                try:
                    aic = ARIMA(y, order=(p, d, q)).fit().aic
                    if aic < best_aic:
                        best_aic = aic
                        best = (p, d, q)
                except Exception:
                    pass
        return best


class RandomForestModel:
    name = "Random Forest"

    def fit_predict(self, series: pd.Series, periods: int,
                    lags: int = 4) -> ForecastResult:
        y   = series.values.astype(float)
        lags = min(lags, len(y) // 2)

        X, Y = [], []
        for i in range(lags, len(y)):
            X.append(y[i - lags: i])
            Y.append(y[i])
        X, Y = np.array(X), np.array(Y)

        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(X, Y)

        fitted_vals = model.predict(X)
        fitted = pd.Series(
            np.concatenate([np.full(lags, np.nan), fitted_vals]),
            index=series.index
        )

        # Recursive multi-step forecast
        history   = list(y)
        fc_vals   = []
        fc_lo     = []
        fc_hi     = []
        for _ in range(periods):
            feat       = np.array(history[-lags:]).reshape(1, -1)
            trees_pred = np.array([tree.predict(feat)[0] for tree in model.estimators_])
            pred       = trees_pred.mean()
            fc_vals.append(pred)
            fc_lo.append(np.percentile(trees_pred, 10))
            fc_hi.append(np.percentile(trees_pred, 90))
            history.append(pred)

        future_idx = LinearTrendModel._future_index(series, periods)
        actual_cmp = y[lags:]
        mape, rmse = LinearTrendModel._errors(actual_cmp, fitted_vals)

        return ForecastResult(
            model_name    = self.name,
            forecast      = pd.Series(fc_vals, index=future_idx),
            confidence_lo = pd.Series(fc_lo, index=future_idx),
            confidence_hi = pd.Series(fc_hi, index=future_idx),
            fitted        = fitted,
            mape          = mape,
            rmse          = rmse,
            params        = {"lags": lags, "n_estimators": 200},
        )


class GradientBoostingModel:
    name = "Gradient Boosting"

    def fit_predict(self, series: pd.Series, periods: int,
                    lags: int = 4) -> ForecastResult:
        y    = series.values.astype(float)
        lags = min(lags, len(y) // 2)

        X, Y = [], []
        for i in range(lags, len(y)):
            X.append(y[i - lags: i])
            Y.append(y[i])
        X, Y = np.array(X), np.array(Y)

        model = GradientBoostingRegressor(n_estimators=200, random_state=42,
                                          learning_rate=0.05, max_depth=3)
        model.fit(X, Y)

        fitted_vals = model.predict(X)
        fitted = pd.Series(
            np.concatenate([np.full(lags, np.nan), fitted_vals]),
            index=series.index
        )

        history = list(y)
        fc_vals = []
        std_resid = np.std(y[lags:] - fitted_vals) * 1.28

        for _ in range(periods):
            feat = np.array(history[-lags:]).reshape(1, -1)
            pred = float(model.predict(feat)[0])
            fc_vals.append(pred)
            history.append(pred)

        future_idx = LinearTrendModel._future_index(series, periods)
        mape, rmse = LinearTrendModel._errors(y[lags:], fitted_vals)
        fc_arr     = np.array(fc_vals)

        return ForecastResult(
            model_name    = self.name,
            forecast      = pd.Series(fc_arr, index=future_idx),
            confidence_lo = pd.Series(fc_arr - std_resid, index=future_idx),
            confidence_hi = pd.Series(fc_arr + std_resid, index=future_idx),
            fitted        = fitted,
            mape          = mape,
            rmse          = rmse,
        )


# ------------------------------------------------------------------ #
#  Ensemble                                                            #
# ------------------------------------------------------------------ #
class EnsembleForecaster:
    """Runs all models and builds scenario bundles."""

    MODELS = [
        LinearTrendModel,
        ExponentialSmoothingModel,
        ARIMAModel,
        RandomForestModel,
        GradientBoostingModel,
    ]

    def __init__(self):
        self.results: dict[str, ForecastResult] = {}

    def run_all(self, series: pd.Series, periods: int = 8) -> dict[str, ForecastResult]:
        for ModelClass in self.MODELS:
            m = ModelClass()
            try:
                self.results[m.name] = m.fit_predict(series, periods)
            except Exception as e:
                print(f"  ⚠  {m.name} failed: {e}")
        return self.results

    def best_model(self) -> Optional[ForecastResult]:
        if not self.results:
            return None
        return min(self.results.values(), key=lambda r: r.mape)

    def ensemble_forecast(self) -> Optional[pd.Series]:
        if not self.results:
            return None
        forecasts = pd.DataFrame({r.model_name: r.forecast for r in self.results.values()})
        weights   = np.array([1 / (r.mape + 1e-6) for r in self.results.values()])
        weights  /= weights.sum()
        return (forecasts * weights).sum(axis=1)

    def scenarios(self, series: pd.Series, periods: int = 8,
                  bull_factor: float = 1.15,
                  bear_factor: float = 0.85) -> ScenarioBundle:
        base_result = self.best_model() or list(self.results.values())[0]
        base_fc     = base_result.forecast

        bull_fc = base_fc * bull_factor
        bear_fc = base_fc * bear_factor

        def _wrap(fc, name):
            spread = fc * 0.05
            return ForecastResult(
                model_name    = name,
                forecast      = fc,
                confidence_lo = fc - spread,
                confidence_hi = fc + spread,
                fitted        = base_result.fitted,
                mape          = base_result.mape,
                rmse          = base_result.rmse,
            )

        return ScenarioBundle(
            base        = base_result,
            bull        = _wrap(bull_fc, "Bull Case"),
            bear        = _wrap(bear_fc, "Bear Case"),
            col_name    = series.name or "Value",
            historical  = series,
        )


# ------------------------------------------------------------------ #
#  KPI calculations                                                    #
# ------------------------------------------------------------------ #
class KPICalculator:
    @staticmethod
    def cagr(series: pd.Series) -> float:
        s, e = float(series.iloc[0]), float(series.iloc[-1])
        if s <= 0:
            return 0.0
        n = len(series) - 1
        return round((e / s) ** (1 / n) - 1, 4) if n > 0 else 0.0

    @staticmethod
    def yoy_growth(series: pd.Series, freq: int = 4) -> pd.Series:
        return series.pct_change(freq).dropna().round(4)

    @staticmethod
    def rolling_avg(series: pd.Series, window: int = 4) -> pd.Series:
        return series.rolling(window).mean()

    @staticmethod
    def seasonality_index(series: pd.Series, freq: int = 4) -> Optional[pd.Series]:
        if len(series) < freq * 2:
            return None
        ratios = []
        for i, val in enumerate(series):
            period_mean = series[i % freq::freq].mean()
            if period_mean != 0:
                ratios.append(val / period_mean)
            else:
                ratios.append(np.nan)
        return pd.Series(ratios, index=series.index).round(4)

    @staticmethod
    def compound_forecast_growth(result: ForecastResult) -> float:
        fc = result.forecast
        if fc.iloc[0] <= 0:
            return 0.0
        n = len(fc) - 1
        return round((fc.iloc[-1] / fc.iloc[0]) ** (1 / n) - 1, 4) if n > 0 else 0.0

    @staticmethod
    def full_report(series: pd.Series) -> dict:
        return {
            "periods":           len(series),
            "start_value":       round(float(series.iloc[0]), 2),
            "end_value":         round(float(series.iloc[-1]), 2),
            "total_growth_pct":  round((series.iloc[-1] / series.iloc[0] - 1) * 100, 2),
            "cagr_pct":          round(KPICalculator.cagr(series) * 100, 2),
            "mean":              round(float(series.mean()), 2),
            "std_dev":           round(float(series.std()), 2),
            "cv_pct":            round(float(series.std() / series.mean()) * 100, 2),
            "min":               round(float(series.min()), 2),
            "max":               round(float(series.max()), 2),
        }
