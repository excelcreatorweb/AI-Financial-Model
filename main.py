"""
main.py — AI Financial Model
==============================
Interactive CLI to load Excel data, run forecasts, and export reports.

Usage:
    python main.py                          # interactive prompts
    python main.py --file data.xlsx         # load specific file
    python main.py --file data.xlsx --col Revenue --periods 8
    python main.py --demo                   # generate & run on demo data
"""
import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table   import Table
from rich.panel   import Panel
from rich.prompt  import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from data_loader    import FinancialDataLoader
from forecaster     import EnsembleForecaster, KPICalculator
from visualizer     import save_all_charts
from excel_reporter import export_report

console = Console()


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #
def banner():
    console.print(Panel.fit(
        "[bold blue]AI Financial Model[/bold blue]\n"
        "[dim]Multi-model forecasting · Scenario analysis · Excel export[/dim]",
        border_style="blue"
    ))


def pick_column(loader: FinancialDataLoader) -> str:
    numeric = loader.get_numeric_columns()
    if not numeric:
        console.print("[red]No numeric columns found in primary sheet.[/red]")
        sys.exit(1)

    table = Table(title="Available Columns", show_lines=True)
    table.add_column("#",    style="cyan", width=4)
    table.add_column("Column", style="white")
    table.add_column("Sample Values", style="dim")
    for i, col in enumerate(numeric):
        sample = loader.primary_df[col].dropna().head(3).tolist()
        sample_str = " · ".join(f"{v:,.0f}" for v in sample)
        table.add_row(str(i + 1), col, sample_str)
    console.print(table)

    default_idx = 1
    if loader.revenue_col and loader.revenue_col in numeric:
        default_idx = numeric.index(loader.revenue_col) + 1

    choice = Prompt.ask(
        f"[bold]Select column to forecast[/bold] [dim](1-{len(numeric)})[/dim]",
        default=str(default_idx)
    )
    try:
        return numeric[int(choice) - 1]
    except (ValueError, IndexError):
        console.print("[red]Invalid choice, using first column.[/red]")
        return numeric[0]


# ------------------------------------------------------------------ #
#  Main pipeline                                                       #
# ------------------------------------------------------------------ #
def run(filepath: str, col_name: str = None, periods: int = 8,
        output_dir: str = "output"):
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    # 1. Load data
    console.rule("[bold blue]Step 1 · Loading Data")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  transient=True) as prog:
        prog.add_task("Reading Excel…", total=None)
        loader = FinancialDataLoader(filepath)

    info = loader.summary()
    console.print(f"  [green]✓[/green] File:          [white]{info['file']}[/white]")
    console.print(f"  [green]✓[/green] Sheets:        [white]{', '.join(info['sheets'])}[/white]")
    console.print(f"  [green]✓[/green] Primary sheet: [white]{info['primary_sheet']}[/white]")
    console.print(f"  [green]✓[/green] Rows:          [white]{info['rows']}[/white]")
    console.print(f"  [green]✓[/green] Date column:   [white]{info['date_col'] or 'none'}[/white]")

    # 2. Column selection
    if col_name is None:
        col_name = pick_column(loader)
    console.print(f"\n  [bold]Forecasting:[/bold] [cyan]{col_name}[/cyan]  "
                  f"for [cyan]{periods}[/cyan] periods\n")

    series = loader.get_time_series(col_name)
    series.name = col_name

    # 3. KPI summary
    console.rule("[bold blue]Step 2 · Historical KPIs")
    kpi = KPICalculator.full_report(series)
    kpi_table = Table(show_header=False, box=None, padding=(0, 2))
    kpi_table.add_column(style="dim")
    kpi_table.add_column(style="bold white")
    for label, val in [
        ("Periods",      str(kpi["periods"])),
        ("Start Value",  f"{kpi['start_value']:,.0f}"),
        ("End Value",    f"{kpi['end_value']:,.0f}"),
        ("Total Growth", f"{kpi['total_growth_pct']:+.1f}%"),
        ("CAGR",         f"{kpi['cagr_pct']:+.1f}%"),
        ("Mean",         f"{kpi['mean']:,.0f}"),
        ("Std Dev",      f"{kpi['std_dev']:,.0f}"),
    ]:
        kpi_table.add_row(label, val)
    console.print(kpi_table)

    # 4. Run models
    console.rule("[bold blue]Step 3 · Running Forecast Models")
    forecaster = EnsembleForecaster()
    results = {}

    model_names = ["Linear Trend", "Exponential Smoothing (Holt-Winters)",
                   "ARIMA", "Random Forest", "Gradient Boosting"]
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  transient=False) as prog:
        for ModelClass in EnsembleForecaster.MODELS:
            m    = ModelClass()
            task = prog.add_task(f"  Fitting {m.name}…", total=None)
            try:
                res = m.fit_predict(series, periods)
                results[m.name] = res
                forecaster.results[m.name] = res
                prog.update(task, description=f"  [green]✓[/green] {m.name}  "
                                              f"(MAPE {res.mape:.1f}%)")
            except Exception as e:
                prog.update(task, description=f"  [red]✗[/red] {m.name}  ({e})")
            prog.stop_task(task)

    if not results:
        console.print("[red]All models failed. Check your data.[/red]")
        sys.exit(1)

    # Model comparison table
    table = Table(title="Model Performance", show_lines=True)
    table.add_column("Model",         style="white")
    table.add_column("MAPE %",        style="cyan", justify="right")
    table.add_column("RMSE",          style="cyan", justify="right")
    table.add_column("Forecast CAGR", style="green", justify="right")
    table.add_column("Status",        justify="center")

    best = min(results.values(), key=lambda r: r.mape)
    for name, result in results.items():
        fc_cagr = KPICalculator.compound_forecast_growth(result) * 100
        is_best = name == best.model_name
        table.add_row(
            f"[bold]{name}[/bold]" if is_best else name,
            f"{result.mape:.1f}%",
            f"{result.rmse:,.0f}",
            f"{fc_cagr:+.1f}%",
            "⭐ Best" if is_best else "",
        )
    console.print(table)

    # 5. Scenario bundle
    bundle = forecaster.scenarios(series, periods)
    console.print(f"\n  [bold]Scenario ranges (final period):[/bold]")
    console.print(f"    Bear: [red]{bundle.bear.forecast.iloc[-1]:,.0f}[/red]  "
                  f"Base: [white]{bundle.base.forecast.iloc[-1]:,.0f}[/white]  "
                  f"Bull: [green]{bundle.bull.forecast.iloc[-1]:,.0f}[/green]")

    # 6. Save charts
    console.rule("[bold blue]Step 4 · Saving Charts")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  transient=True) as prog:
        prog.add_task("Rendering charts…", total=None)
        chart_paths = save_all_charts(series, results, bundle, output_dir=str(out))
    console.print(f"  [green]✓[/green] {len(chart_paths)} charts → [dim]{out}/[/dim]")

    # 7. Excel report
    console.rule("[bold blue]Step 5 · Excel Report")
    report_path = str(out / "forecast_report.xlsx")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  transient=True) as prog:
        prog.add_task("Building Excel workbook…", total=None)
        export_report(series, results, bundle, output_path=report_path)
    console.print(f"  [green]✓[/green] Report saved → [cyan]{report_path}[/cyan]")

    # 8. Done
    console.rule()
    console.print(Panel(
        f"[bold green]✅  All done![/bold green]\n\n"
        f"  Best model: [bold]{best.model_name}[/bold]  (MAPE {best.mape:.1f}%)\n"
        f"  Base forecast CAGR: [bold]{KPICalculator.compound_forecast_growth(best)*100:+.1f}%[/bold]\n"
        f"  Output: [cyan]{out.resolve()}[/cyan]",
        border_style="green"
    ))


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(description="AI Financial Model")
    parser.add_argument("--file",    "-f", help="Path to Excel file")
    parser.add_argument("--col",     "-c", help="Column to forecast")
    parser.add_argument("--periods", "-p", type=int, default=8,
                        help="Number of periods to forecast (default: 8)")
    parser.add_argument("--output",  "-o", default="output",
                        help="Output directory (default: output)")
    parser.add_argument("--demo",    action="store_true",
                        help="Generate and run on demo data")
    args = parser.parse_args()

    banner()

    if args.demo:
        console.print("[bold yellow]Running demo mode…[/bold yellow]")
        from sample_data_generator import generate_sample_data
        generate_sample_data("demo_data.xlsx")
        filepath = "demo_data.xlsx"
        col_name = "Revenue"
    elif args.file:
        filepath = args.file
        col_name = args.col
    else:
        console.print("\n[bold]No file specified.[/bold] Run with [cyan]--demo[/cyan] "
                      "to try sample data, or enter a path:\n")
        filepath = Prompt.ask("[bold]Excel file path[/bold]")
        col_name = None

    periods = args.periods

    if not col_name and args.col:
        col_name = args.col

    if col_name is None and not args.demo:
        # Will be interactively chosen inside run()
        pass

    run(filepath=filepath, col_name=col_name, periods=periods, output_dir=args.output)


if __name__ == "__main__":
    main()
