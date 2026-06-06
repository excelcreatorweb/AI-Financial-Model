import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import tempfile, os, sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import FinancialDataLoader
from forecaster import EnsembleForecaster, KPICalculator
from excel_reporter import export_report

# ------------------------------------------------------------------ #
#  Page config                                                         #
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="AI Financial Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        border-left: 4px solid #2E75B6;
    }
    .stMetric { background: white; border-radius: 10px; padding: 12px; }
    h1 { color: #1F4E79; }
    h2 { color: #2E75B6; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ #
#  Sidebar                                                             #
# ------------------------------------------------------------------ #
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/financial-growth.png", width=60)
    st.title("AI Financial Model")
    st.caption("Multi-model forecasting · Scenario analysis")
    st.divider()

    uploaded = st.file_uploader("📂 Upload Excel File", type=["xlsx", "xls"],
                                 help="Upload any Excel file with financial data")

    st.divider()
    periods = st.slider("📅 Forecast Periods", min_value=2, max_value=20, value=8,
                        help="Number of future periods to forecast")
    bull_factor = st.slider("🐂 Bull Case multiplier", 1.05, 1.50, 1.15, 0.05)
    bear_factor = st.slider("🐻 Bear Case multiplier", 0.50, 0.95, 0.85, 0.05)

    st.divider()
    st.caption("Built with Python · scikit-learn · statsmodels")


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #
COLORS = {
    "historical": "#1F4E79",
    "forecast":   "#2E75B6",
    "bull":       "#27AE60",
    "bear":       "#E74C3C",
    "base":       "#2E75B6",
    "ci":         "rgba(46,117,182,0.15)",
    "models":     ["#1F4E79","#E74C3C","#27AE60","#F39C12","#9B59B6"],
}

def fmt(v):
    if abs(v) >= 1_000_000: return f"{v/1_000_000:.1f}M"
    if abs(v) >= 1_000:     return f"{v/1_000:.1f}K"
    return f"{v:.1f}"


def plot_forecast(result, series, title=""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(series.index.astype(str)), y=list(series.values),
        name="Historical", line=dict(color=COLORS["historical"], width=3),
        mode="lines+markers", marker=dict(size=5)
    ))
    fig.add_trace(go.Scatter(
        x=list(result.fitted.dropna().index.astype(str)),
        y=list(result.fitted.dropna().values),
        name="Fitted", line=dict(color=COLORS["forecast"], width=1.5, dash="dot"),
        opacity=0.6
    ))
    # Bridge historical to forecast
    bridge_x = [str(series.index[-1])] + list(result.forecast.index.astype(str))
    bridge_y = [series.iloc[-1]]       + list(result.forecast.values)
    ci_lo_y  = [series.iloc[-1]]       + list(result.confidence_lo.values)
    ci_hi_y  = [series.iloc[-1]]       + list(result.confidence_hi.values)

    fig.add_trace(go.Scatter(
        x=bridge_x + bridge_x[::-1],
        y=ci_hi_y + ci_lo_y[::-1],
        fill="toself", fillcolor=COLORS["ci"],
        line=dict(color="rgba(0,0,0,0)"),
        name="80% CI", showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=bridge_x, y=bridge_y,
        name=f"Forecast", line=dict(color=COLORS["forecast"], width=3),
        mode="lines+markers", marker=dict(size=6, symbol="diamond")
    ))
    fig.add_vline(x=str(series.index[-1]), line_dash="dot", line_color="gray", opacity=0.5)
    fig.update_layout(
        title=title or f"{result.model_name}  |  MAPE {result.mape:.1f}%",
        template="plotly_white", height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def plot_scenarios(bundle):
    fig = go.Figure()
    hist = bundle.historical

    fig.add_trace(go.Scatter(
        x=list(hist.index.astype(str)), y=list(hist.values),
        name="Historical", line=dict(color=COLORS["historical"], width=3),
        mode="lines+markers", marker=dict(size=5)
    ))
    fig.add_vline(x=str(hist.index[-1]), line_dash="dot", line_color="gray", opacity=0.5)

    for scenario, color, label in [
        (bundle.bull, COLORS["bull"], "🐂 Bull"),
        (bundle.base, COLORS["base"], "📊 Base"),
        (bundle.bear, COLORS["bear"], "🐻 Bear"),
    ]:
        bx = [str(hist.index[-1])] + list(scenario.forecast.index.astype(str))
        by = [hist.iloc[-1]]       + list(scenario.forecast.values)
        fig.add_trace(go.Scatter(
            x=bx, y=by, name=label,
            line=dict(color=color, width=2.5),
            mode="lines+markers", marker=dict(size=6)
        ))

    fig.update_layout(
        title="Scenario Analysis — Bear / Base / Bull",
        template="plotly_white", height=440,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def plot_model_comparison(results, series):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(series.index.astype(str)), y=list(series.values),
        name="Historical", line=dict(color=COLORS["historical"], width=3),
        mode="lines+markers", marker=dict(size=5)
    ))
    fig.add_vline(x=str(series.index[-1]), line_dash="dot", line_color="gray", opacity=0.5)
    for i, (name, result) in enumerate(results.items()):
        bx = [str(series.index[-1])] + list(result.forecast.index.astype(str))
        by = [series.iloc[-1]]       + list(result.forecast.values)
        fig.add_trace(go.Scatter(
            x=bx, y=by,
            name=f"{name} ({result.mape:.1f}%)",
            line=dict(color=COLORS["models"][i % 5], width=2, dash="dash"),
            mode="lines"
        ))
    fig.update_layout(
        title="All Models Comparison",
        template="plotly_white", height=440,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def plot_growth_bars(series):
    yoy = series.pct_change(min(4, len(series)//3)).dropna()
    colors = [COLORS["bull"] if v >= 0 else COLORS["bear"] for v in yoy.values]
    fig = go.Figure(go.Bar(
        x=list(yoy.index.astype(str)), y=list(yoy.values * 100),
        marker_color=colors, text=[f"{v:.1f}%" for v in yoy.values * 100],
        textposition="outside"
    ))
    fig.update_layout(
        title="Period-over-Period Growth %",
        template="plotly_white", height=350,
        yaxis_title="Growth %",
        margin=dict(l=20, r=20, t=50, b=60),
    )
    fig.add_hline(y=0, line_color="gray", line_width=1)
    return fig


def plot_mape_bar(results):
    names = list(results.keys())
    mapes = [r.mape for r in results.values()]
    colors = [COLORS["bull"] if m == min(mapes) else COLORS["models"][i % 5]
              for i, m in enumerate(mapes)]
    fig = go.Figure(go.Bar(
        x=mapes, y=names, orientation="h",
        marker_color=colors,
        text=[f"{m:.1f}%" for m in mapes], textposition="outside"
    ))
    fig.update_layout(
        title="Model Accuracy (MAPE — lower is better)",
        template="plotly_white", height=300,
        xaxis_title="MAPE %",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# ------------------------------------------------------------------ #
#  Main app                                                            #
# ------------------------------------------------------------------ #
st.title("📊 AI Financial Model")
st.caption("Upload your Excel file, pick a column, and get instant multi-model forecasts.")

if uploaded is None:
    st.info("👈 Upload an Excel file in the sidebar to get started, or use the demo below.")
    if st.button("🚀 Run Demo with Sample Data", type="primary", use_container_width=True):
        from sample_data_generator import generate_sample_data
        generate_sample_data("demo_data.xlsx")
        with open("demo_data.xlsx", "rb") as f:
            st.session_state["demo_bytes"] = f.read()
        st.rerun()

    if "demo_bytes" in st.session_state:
        uploaded = st.session_state["demo_bytes"]
    else:
        st.stop()

# Save upload to temp file
with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
    if isinstance(uploaded, bytes):
        tmp.write(uploaded)
    else:
        tmp.write(uploaded.read())
    tmp_path = tmp.name

# Load data
try:
    loader = FinancialDataLoader(tmp_path)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

info = loader.summary()

# Column picker
numeric_cols = loader.get_numeric_columns()
default_col  = loader.revenue_col or numeric_cols[0]
default_idx  = numeric_cols.index(default_col) if default_col in numeric_cols else 0

col_name = st.selectbox(
    "📌 Select column to forecast",
    numeric_cols, index=default_idx,
    help="Pick any numeric column from your Excel file"
)

series = loader.get_time_series(col_name)
series.name = col_name

st.divider()

# ── KPI row ──────────────────────────────────────────────────────── #
kpi = KPICalculator.full_report(series)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("📅 Periods",      str(kpi["periods"]))
c2.metric("🏁 Start Value",  fmt(kpi["start_value"]))
c3.metric("🎯 End Value",    fmt(kpi["end_value"]))
c4.metric("📈 Total Growth", f"{kpi['total_growth_pct']:+.1f}%")
c5.metric("⚡ CAGR",         f"{kpi['cagr_pct']:+.1f}%")

st.divider()

# ── Run models ───────────────────────────────────────────────────── #
with st.spinner("🤖 Running forecast models…"):
    forecaster = EnsembleForecaster()
    results = {}
    for ModelClass in EnsembleForecaster.MODELS:
        m = ModelClass()
        try:
            res = m.fit_predict(series, periods)
            results[m.name] = res
            forecaster.results[m.name] = res
        except Exception:
            pass

if not results:
    st.error("All models failed. Please check your data.")
    st.stop()

bundle = forecaster.scenarios(series, periods, bull_factor, bear_factor)
best   = min(results.values(), key=lambda r: r.mape)

# ── Tabs ─────────────────────────────────────────────────────────── #
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Best Forecast", "📊 All Models", "🎭 Scenarios", "📉 KPIs", "⬇️ Download"
])

with tab1:
    st.plotly_chart(plot_forecast(best, series), use_container_width=True)
    st.caption(f"Best model: **{best.model_name}** with MAPE {best.mape:.1f}%")

    # Forecast table
    fc_df = pd.DataFrame({
        "Period":       best.forecast.index.astype(str),
        "Forecast":     best.forecast.values.round(0),
        "Lower (80%)":  best.confidence_lo.values.round(0),
        "Upper (80%)":  best.confidence_hi.values.round(0),
    })
    st.dataframe(fc_df, use_container_width=True, hide_index=True)

with tab2:
    st.plotly_chart(plot_model_comparison(results, series), use_container_width=True)
    st.plotly_chart(plot_mape_bar(results), use_container_width=True)

    # Model table
    model_df = pd.DataFrame([{
        "Model":         name,
        "MAPE %":        f"{r.mape:.1f}%",
        "RMSE":          f"{r.rmse:,.0f}",
        "Forecast CAGR": f"{KPICalculator.compound_forecast_growth(r)*100:+.1f}%",
        "Best":          "⭐" if name == best.model_name else ""
    } for name, r in results.items()])
    st.dataframe(model_df, use_container_width=True, hide_index=True)

    st.subheader("Individual Model Charts")
    for name, result in results.items():
        with st.expander(f"{name}  —  MAPE {result.mape:.1f}%"):
            st.plotly_chart(plot_forecast(result, series, title=name),
                            use_container_width=True)

with tab3:
    st.plotly_chart(plot_scenarios(bundle), use_container_width=True)

    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("🐻 Bear (final)", fmt(bundle.bear.forecast.iloc[-1]),
               delta=f"{(bundle.bear.forecast.iloc[-1]/series.iloc[-1]-1)*100:+.1f}% vs today")
    sc2.metric("📊 Base (final)", fmt(bundle.base.forecast.iloc[-1]),
               delta=f"{(bundle.base.forecast.iloc[-1]/series.iloc[-1]-1)*100:+.1f}% vs today")
    sc3.metric("🐂 Bull (final)", fmt(bundle.bull.forecast.iloc[-1]),
               delta=f"{(bundle.bull.forecast.iloc[-1]/series.iloc[-1]-1)*100:+.1f}% vs today")

    sc_df = pd.DataFrame({
        "Period": bundle.base.forecast.index.astype(str),
        "Bear":   bundle.bear.forecast.values.round(0),
        "Base":   bundle.base.forecast.values.round(0),
        "Bull":   bundle.bull.forecast.values.round(0),
    })
    st.dataframe(sc_df, use_container_width=True, hide_index=True)

with tab4:
    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(plot_growth_bars(series), use_container_width=True)
    with col_r:
        roll = KPICalculator.rolling_avg(series, window=min(4, len(series)//2))
        fig_roll = go.Figure()
        fig_roll.add_trace(go.Scatter(
            x=list(series.index.astype(str)), y=list(series.values),
            name="Actual", line=dict(color=COLORS["historical"], width=1.5), opacity=0.5
        ))
        fig_roll.add_trace(go.Scatter(
            x=list(roll.index.astype(str)), y=list(roll.values),
            name="Rolling Avg", line=dict(color=COLORS["forecast"], width=3)
        ))
        fig_roll.update_layout(title="Rolling Average", template="plotly_white",
                               height=350, margin=dict(l=20,r=20,t=50,b=60))
        st.plotly_chart(fig_roll, use_container_width=True)

    st.subheader("Full KPI Summary")
    kpi_df = pd.DataFrame([
        {"Metric": k.replace("_", " ").title(), "Value": v}
        for k, v in kpi.items()
    ])
    st.dataframe(kpi_df, use_container_width=True, hide_index=True)

with tab5:
    st.subheader("⬇️ Download Excel Report")
    st.write("Export a full professional Excel workbook with all forecasts, scenarios, and charts.")

    with st.spinner("Building Excel report…"):
        report_path = tempfile.mktemp(suffix=".xlsx")
        try:
            export_report(series, results, bundle, output_path=report_path)
            with open(report_path, "rb") as f:
                report_bytes = f.read()
            st.download_button(
                label="📥 Download forecast_report.xlsx",
                data=report_bytes,
                file_name="forecast_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Report generation failed: {e}")

    st.subheader("⬇️ Download Forecast Data (CSV)")
    csv_df = pd.DataFrame({
        "Period":      best.forecast.index.astype(str),
        "Bear_Case":   bundle.bear.forecast.values.round(2),
        "Base_Case":   bundle.base.forecast.values.round(2),
        "Bull_Case":   bundle.bull.forecast.values.round(2),
        "Best_Model":  best.forecast.values.round(2),
        "CI_Lower":    best.confidence_lo.values.round(2),
        "CI_Upper":    best.confidence_hi.values.round(2),
    })
    st.download_button(
        label="📥 Download forecasts.csv",
        data=csv_df.to_csv(index=False),
        file_name="forecasts.csv",
        mime="text/csv",
        use_container_width=True,
    )
