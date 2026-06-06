# AI Financial Model

A Python-based financial forecasting tool that ingests Excel data and produces
multi-model forecasts, scenario analysis, KPI dashboards, and professional Excel reports.

---

## Features

| Feature | Details |
|---|---|
| **Data Ingestion** | Reads any Excel workbook; auto-detects date and revenue columns |
| **5 Forecast Models** | Linear Trend, Exponential Smoothing, ARIMA, Random Forest, Gradient Boosting |
| **Ensemble Forecast** | Weighted average of all models |
| **Scenario Analysis** | Bear / Base / Bull cases |
| **KPI Dashboard** | CAGR, growth rates, margins, rolling averages |
| **Charts** | PNG charts for every model, scenarios, comparison, and KPI dashboard |
| **Excel Export** | Industry-standard color-coded workbook with embedded charts |

---

## Setup

### 1. Prerequisites
- Python 3.10 or later
- PyCharm Community (free) or PyCharm Professional

### 2. Install dependencies

Open a terminal in the project folder and run:

```bash
pip install -r requirements.txt
```

Or in PyCharm: open `requirements.txt` → click **Install requirements** in the banner.

### 3. (Optional) Create a virtual environment first

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Quick Start

### Run the demo (generates sample data automatically)

```bash
python main.py --demo
```

This creates `demo_data.xlsx` with 24 quarters of realistic P&L, Balance Sheet,
Cash Flow, and KPI data, then runs all forecasts and saves output to `output/`.

### Use your own Excel file

```bash
# Interactive mode (you'll be prompted to pick a column)
python main.py --file your_data.xlsx

# Specify everything up front
python main.py --file your_data.xlsx --col Revenue --periods 8 --output results/
```

---

## Your Excel File Format

The tool auto-detects columns but works best if your Excel looks like this:

| Date       | Revenue    | COGS      | Gross_Profit | ... |
|------------|-----------|-----------|--------------|-----|
| 2021-01-01 | 1,000,000 | 420,000   | 580,000      | ... |
| 2021-04-01 | 1,050,000 | 441,000   | 609,000      | ... |

- **Date column**: any name containing "date", "period", "quarter", "year", or values that parse as dates
- **Numeric columns**: any column — you'll pick which to forecast interactively
- **Multiple sheets**: the tool picks the most relevant sheet (P&L / income statement preferred)

---

## Output Files

After running, the `output/` folder contains:

```
output/
├── forecast_report.xlsx        ← Main Excel report (open this!)
│   ├── Summary                 ← KPI table + model accuracy
│   ├── Linear Trend            ← Historical + fitted + forecast + chart
│   ├── Exponential Smoothing   ← Same structure
│   ├── ARIMA                   ← Same structure
│   ├── Random Forest           ← Same structure
│   ├── Gradient Boosting       ← Same structure
│   ├── Scenarios               ← Bear / Base / Bull
│   └── Ensemble_Forecast       ← Weighted ensemble
├── kpi_dashboard.png           ← Full KPI dashboard chart
├── model_comparison.png        ← All 5 models overlaid
├── scenarios.png               ← Bear/Base/Bull scenario chart
├── forecast_linear_trend.png
├── forecast_arima.png
└── ...
```

---

## Excel Color Coding (Industry Standard)

| Color | Meaning |
|-------|---------|
| **Blue text** | Hardcoded inputs |
| **Black text** | Formulas / calculations |
| **Green text** | Forecasted values (cross-sheet links) |
| **Green fill** | Forecast rows |
| **Blue fill** | Alternating historical rows |

---

## Project Structure

```
ai_financial_model/
├── main.py                   ← Entry point / CLI
├── data_loader.py            ← Excel reading & auto-detection
├── forecaster.py             ← All 5 models + KPI calculations
├── visualizer.py             ← Matplotlib charts
├── excel_reporter.py         ← Professional Excel report generator
├── sample_data_generator.py  ← Creates demo_data.xlsx
└── requirements.txt
```

---

## Extending the Model

### Add a new forecasting model

1. Open `forecaster.py`
2. Create a class with a `name` attribute and `fit_predict(series, periods)` method returning a `ForecastResult`
3. Add your class to `EnsembleForecaster.MODELS`

### Change scenario multipliers

```python
# In main.py, change the run() call:
bundle = forecaster.scenarios(series, periods, bull_factor=1.20, bear_factor=0.80)
```

### Forecast multiple columns at once

```python
for col in ["Revenue", "Gross_Profit", "EBITDA"]:
    run(filepath="data.xlsx", col_name=col, periods=8, output_dir=f"output/{col}")
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `No usable sheets found` | Ensure at least one sheet has 3+ rows of numeric data |
| ARIMA very slow | Normal for large datasets; reduce `periods` |
| Charts not opening | PNGs are saved to `output/`; open them from there |
