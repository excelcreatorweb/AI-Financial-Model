"""
data_loader.py
Handles reading, validating, and cleaning financial Excel data.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


class FinancialDataLoader:
    """Loads and validates financial data from Excel workbooks."""

    KNOWN_REVENUE_COLS = [
        "revenue", "sales", "net_revenue", "total_revenue",
        "net_sales", "turnover", "income"
    ]
    DATE_COLS = ["date", "period", "quarter", "year", "month", "fiscal_year"]

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        self.sheets: dict[str, pd.DataFrame] = {}
        self.primary_df: Optional[pd.DataFrame] = None
        self.date_col: Optional[str] = None
        self.revenue_col: Optional[str] = None
        self._load()

    # ------------------------------------------------------------------ #
    #  Loading                                                             #
    # ------------------------------------------------------------------ #
    def _load(self):
        xl = pd.ExcelFile(self.filepath)
        for sheet in xl.sheet_names:
            try:
                df = xl.parse(sheet)
                df = self._clean(df)
                if df is not None and len(df) >= 3:
                    self.sheets[sheet] = df
            except Exception:
                pass

        if not self.sheets:
            raise ValueError("No usable sheets found in the workbook.")

        self.primary_df, self.date_col, self.revenue_col = self._pick_primary()

    def _clean(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        df = df.dropna(how="all").dropna(axis=1, how="all")
        if df.empty or len(df.columns) < 2:
            return None
        df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
        df = df.reset_index(drop=True)
        return df

    # ------------------------------------------------------------------ #
    #  Sheet / column detection                                            #
    # ------------------------------------------------------------------ #
    def _pick_primary(self) -> tuple[pd.DataFrame, Optional[str], Optional[str]]:
        """Heuristically find the most useful sheet (P&L / revenue-bearing)."""
        best_sheet = None
        best_score = -1

        for name, df in self.sheets.items():
            score = 0
            lower_name = name.lower()
            if any(k in lower_name for k in ["p&l", "pnl", "income", "revenue", "financials"]):
                score += 10
            rev_col = self._find_column(df, self.KNOWN_REVENUE_COLS)
            if rev_col:
                score += 5
            date_col = self._find_date_col(df)
            if date_col:
                score += 3
            score += len(df) * 0.01
            if score > best_score:
                best_score = score
                best_sheet = name

        df        = self.sheets[best_sheet]
        date_col  = self._find_date_col(df)
        rev_col   = self._find_column(df, self.KNOWN_REVENUE_COLS)

        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

        return df, date_col, rev_col

    def _find_column(self, df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
        for col in df.columns:
            if col.lower() in candidates:
                return col
        for col in df.columns:
            for c in candidates:
                if c in col.lower():
                    return col
        return None

    def _find_date_col(self, df: pd.DataFrame) -> Optional[str]:
        for col in df.columns:
            if col.lower() in self.DATE_COLS:
                return col
        for col in df.columns:
            sample = df[col].dropna().head(5)
            try:
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().sum() >= 3:
                    return col
            except Exception:
                pass
        return None

    # ------------------------------------------------------------------ #
    #  Public helpers                                                      #
    # ------------------------------------------------------------------ #
    def get_time_series(self, value_col: Optional[str] = None) -> pd.Series:
        """Returns a clean numeric time-series indexed by date."""
        col = value_col or self.revenue_col
        if col is None:
            # Pick the first all-numeric column
            for c in self.primary_df.columns:
                if pd.api.types.is_numeric_dtype(self.primary_df[c]):
                    col = c
                    break

        if col is None:
            raise ValueError("No numeric column found.")

        df = self.primary_df[[self.date_col, col]].copy() if self.date_col else self.primary_df[[col]].copy()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna()

        if self.date_col:
            df = df.set_index(self.date_col)

        return df[col]

    def get_numeric_columns(self) -> list[str]:
        return [c for c in self.primary_df.columns
                if pd.api.types.is_numeric_dtype(self.primary_df[c])]

    def summary(self) -> dict:
        return {
            "file":          self.filepath.name,
            "sheets":        list(self.sheets.keys()),
            "primary_sheet": list(self.sheets.keys())[0] if self.sheets else None,
            "rows":          len(self.primary_df) if self.primary_df is not None else 0,
            "date_col":      self.date_col,
            "revenue_col":   self.revenue_col,
            "numeric_cols":  self.get_numeric_columns(),
        }
