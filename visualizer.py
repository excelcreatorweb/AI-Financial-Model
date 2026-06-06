"""
visualizer.py
All charts and visualizations for the AI Financial Model.
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from forecaster import ForecastResult, ScenarioBundle, KPICalculator


# ------------------------------------------------------------------ #
#  Style                                                               #
# ------------------------------------------------------------------ #
COLORS = {
    "primary":   "#1F4E79",
    "accent":    "#2E75B6",
    "bull":      "#2ECC71",
    "bear":      "#E74C3C",
    "neutral":   "#95A5A6",
    "ci":        "#BDE3F7",
    "grid":      "#EAECEE",
    "text":      "#2C3E50",
    "bg":        "#FAFAFA",
    "models":    ["#1F4E79", "#E74C3C", "#2ECC71", "#F39C12", "#9B59B6"],
}

def _base_style():
    plt.rcParams.update({
        "figure.facecolor":  COLORS["bg"],
        "axes.facecolor":    "white",
        "axes.edgecolor":    "#CCCCCC",
        "axes.labelcolor":   COLORS["text"],
        "axes.titlecolor":   COLORS["text"],
        "axes.grid":         True,
        "grid.color":        COLORS["grid"],
        "grid.linestyle":    "--",
        "grid.linewidth":    0.6,
        "xtick.color":       COLORS["text"],
        "ytick.color":       COLORS["text"],
        "font.family":       "DejaVu Sans",
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


# ------------------------------------------------------------------ #
#  Individual charts                                                   #
# ------------------------------------------------------------------ #
def plot_forecast(result: ForecastResult, historical: pd.Series,
                  title: Optional[str] = None, ax: Optional[plt.Axes] = None,
                  color: str = COLORS["primary"]) -> plt.Figure:
    _base_style()
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.figure

    ax.plot(historical.index, historical.values, color=color,
            linewidth=2, label="Historical", zorder=3)
    ax.plot(result.fitted.dropna().index, result.fitted.dropna().values,
            color=color, linewidth=1, linestyle="--", alpha=0.5, label="Fitted")

    last_hist = pd.Series([historical.iloc[-1]], index=[historical.index[-1]])
    fc_plot = pd.concat([last_hist, result.forecast])
    ci_lo   = pd.concat([last_hist, result.confidence_lo])
    ci_hi   = pd.concat([last_hist, result.confidence_hi])

    ax.plot(fc_plot.index, fc_plot.values, color=COLORS["accent"],
            linewidth=2.5, linestyle="-", label=f"Forecast ({result.model_name})", zorder=3)
    ax.fill_between(ci_lo.index, ci_lo.values, ci_hi.values,
                    color=COLORS["ci"], alpha=0.5, label="80% CI")

    ax.axvline(historical.index[-1], color="#AAAAAA", linestyle=":", linewidth=1)
    ax.set_title(title or f"{result.model_name} Forecast  |  MAPE {result.mape:.1f}%",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel("Value", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.tick_params(axis="x", rotation=30)

    if standalone:
        plt.tight_layout()
    return fig


def plot_scenarios(bundle: ScenarioBundle, ax: Optional[plt.Axes] = None,
                   title: Optional[str] = None) -> plt.Figure:
    _base_style()
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.figure

    hist = bundle.historical
    ax.plot(hist.index, hist.values, color=COLORS["primary"],
            linewidth=2.5, label="Historical", zorder=4)
    ax.axvline(hist.index[-1], color="#AAAAAA", linestyle=":", linewidth=1)

    for scenario, color, label in [
        (bundle.bull, COLORS["bull"], "Bull Case"),
        (bundle.base, COLORS["accent"], "Base Case"),
        (bundle.bear, COLORS["bear"], "Bear Case"),
    ]:
        last = pd.Series([hist.iloc[-1]], index=[hist.index[-1]])
        fc   = pd.concat([last, scenario.forecast])
        ax.plot(fc.index, fc.values, color=color, linewidth=2,
                label=f"{label}", zorder=3)
        lo = pd.concat([last, scenario.confidence_lo])
        hi = pd.concat([last, scenario.confidence_hi])
        ax.fill_between(lo.index, lo.values, hi.values,
                        color=color, alpha=0.08)

    ax.set_title(title or f"Scenario Analysis — {bundle.col_name}",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel("Value", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.tick_params(axis="x", rotation=30)

    if standalone:
        plt.tight_layout()
    return fig


def plot_model_comparison(results: dict[str, ForecastResult],
                           historical: pd.Series) -> plt.Figure:
    _base_style()
    fig, ax = plt.subplots(figsize=(13, 6))

    ax.plot(historical.index, historical.values, color=COLORS["primary"],
            linewidth=3, label="Historical", zorder=5)
    ax.axvline(historical.index[-1], color="#AAAAAA", linestyle=":", linewidth=1)

    for i, (name, result) in enumerate(results.items()):
        color = COLORS["models"][i % len(COLORS["models"])]
        last  = pd.Series([historical.iloc[-1]], index=[historical.index[-1]])
        fc    = pd.concat([last, result.forecast])
        ax.plot(fc.index, fc.values, color=color, linewidth=1.8,
                linestyle="--", label=f"{name}  (MAPE {result.mape:.1f}%)", zorder=3)

    ax.set_title("Model Comparison", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Value", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.8, loc="upper left")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    return fig


def plot_kpi_dashboard(series: pd.Series, results: dict[str, ForecastResult]) -> plt.Figure:
    _base_style()
    kpi = KPICalculator.full_report(series)
    best = min(results.values(), key=lambda r: r.mape) if results else None

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(COLORS["bg"])
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Historical + best forecast
    ax1 = fig.add_subplot(gs[0, :2])
    if best:
        plot_forecast(best, series, title="Best Model Forecast", ax=ax1)

    # 2. Growth rates
    ax2 = fig.add_subplot(gs[0, 2])
    yoy = KPICalculator.yoy_growth(series, freq=min(4, len(series) // 3))
    if not yoy.empty:
        colors = [COLORS["bull"] if v >= 0 else COLORS["bear"] for v in yoy.values]
        ax2.bar(range(len(yoy)), yoy.values * 100, color=colors, edgecolor="white", width=0.7)
        ax2.set_title("Period-over-Period Growth (%)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Growth %", fontsize=9)
        ax2.tick_params(labelbottom=False)
        ax2.axhline(0, color="#AAAAAA", linewidth=0.8)

    # 3. Rolling average
    ax3 = fig.add_subplot(gs[1, 0])
    roll = KPICalculator.rolling_avg(series, window=min(4, len(series) // 2))
    ax3.plot(series.index, series.values, color=COLORS["neutral"],
             linewidth=1, alpha=0.6, label="Actual")
    ax3.plot(roll.index, roll.values, color=COLORS["primary"],
             linewidth=2, label="Rolling Avg")
    ax3.set_title("Rolling Average", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.tick_params(axis="x", rotation=30)

    # 4. Model MAPE bar chart
    ax4 = fig.add_subplot(gs[1, 1])
    if results:
        names  = list(results.keys())
        mapes  = [r.mape for r in results.values()]
        colors = COLORS["models"][:len(names)]
        bars   = ax4.barh(names, mapes, color=colors, edgecolor="white", height=0.5)
        ax4.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=8)
        ax4.set_title("Model MAPE (lower = better)", fontsize=11, fontweight="bold")
        ax4.set_xlabel("MAPE %", fontsize=9)
        ax4.invert_yaxis()

    # 5. KPI summary card
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    kpi_lines = [
        ("Periods",         str(kpi["periods"])),
        ("Start Value",     f"{kpi['start_value']:,.0f}"),
        ("End Value",       f"{kpi['end_value']:,.0f}"),
        ("Total Growth",    f"{kpi['total_growth_pct']:+.1f}%"),
        ("CAGR",            f"{kpi['cagr_pct']:+.1f}%"),
        ("Mean",            f"{kpi['mean']:,.0f}"),
        ("Std Dev",         f"{kpi['std_dev']:,.0f}"),
        ("Coeff of Var",    f"{kpi['cv_pct']:.1f}%"),
    ]
    if best:
        fc_cagr = KPICalculator.compound_forecast_growth(best) * 100
        kpi_lines.append(("Forecast CAGR",  f"{fc_cagr:+.1f}%"))

    y_pos = 0.95
    ax5.text(0.1, y_pos, "📊  Key Metrics", fontsize=11, fontweight="bold",
             color=COLORS["primary"], transform=ax5.transAxes, va="top")
    ax5.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.85, transform=ax5.transAxes,
                                 facecolor="white", edgecolor=COLORS["accent"],
                                 linewidth=1, clip_on=False))
    for i, (label, val) in enumerate(kpi_lines):
        y = 0.85 - i * 0.085
        ax5.text(0.12, y, label, fontsize=9, color=COLORS["text"],
                 transform=ax5.transAxes, va="center")
        ax5.text(0.88, y, val, fontsize=9, fontweight="bold",
                 color=COLORS["primary"], transform=ax5.transAxes,
                 va="center", ha="right")

    fig.suptitle(f"Financial Forecast Dashboard — {series.name or 'Revenue'}",
                 fontsize=15, fontweight="bold", color=COLORS["text"], y=1.01)
    return fig


# ------------------------------------------------------------------ #
#  Save helpers                                                        #
# ------------------------------------------------------------------ #
def save_all_charts(series: pd.Series, results: dict[str, ForecastResult],
                    bundle: ScenarioBundle, output_dir: str = ".") -> list[str]:
    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    paths = []

    for name, result in results.items():
        safe = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        path = str(out / f"forecast_{safe}.png")
        fig  = plot_forecast(result, series)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    fig = plot_scenarios(bundle)
    fig.savefig(str(out / "scenarios.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(out / "scenarios.png"))

    fig = plot_model_comparison(results, series)
    fig.savefig(str(out / "model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(out / "model_comparison.png"))

    fig = plot_kpi_dashboard(series, results)
    fig.savefig(str(out / "kpi_dashboard.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(out / "kpi_dashboard.png"))

    return paths
