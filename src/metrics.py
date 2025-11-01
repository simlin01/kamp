# src/metrics.py
# -*- coding: utf-8 -*-
"""
Metrics utilities for SCM pipeline:
- Forecast metrics: MAE, RMSE, WAPE, MAPE, sMAPE, Bias (ME/MPE)
- Planning  metrics: FillRate, BacklogRate, Utilization, Smoothness, InventoryTurnover
- Optional cluster-level metrics when feat_df (Product_Number, Cluster) provided
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------
# Helpers (safe numeric ops)
# ---------------------------

_EPS = 1e-9


def _to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce all columns to numeric where possible (errors→NaN)."""
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _error_metrics(yhat: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Compute common forecast error metrics on flat arrays."""
    yhat = np.asarray(yhat, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    den = np.clip(np.abs(y), _EPS, None)

    err = yhat - y
    abs_e = np.abs(err)
    sq_e = err ** 2

    mae = float(abs_e.mean())
    rmse = float(np.sqrt(sq_e.mean()))
    wape = float(abs_e.sum() / (np.abs(y).sum() + _EPS))
    mape = float((abs_e / den).mean())
    smape = float((2 * abs_e / (np.abs(yhat) + np.abs(y) + _EPS)).mean())
    me = float(err.mean())
    mpe = float((err / den).mean())
    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE": wape,
        "MAPE": mape,
        "sMAPE": smape,
        "Bias_ME": me,
        "Bias_MPE": mpe,
    }


# ------------------------------------------------
# Forecast metrics (two input shapes are supported)
# ------------------------------------------------
# 1) Wide-vs-Wide:
#    - pred_df: columns [product_col, *horizons]
#    - actuals_df: columns [product_col, *horizons]
#
# 2) Long vs Long (date-based):
#    - pred_df: [product_col, Date, y_hat]
#    - actuals_df: [product_col, Date, y_actual]
#    (Date column name can be 'Date' or 'DateTime'; time part ignored)


def _detect_long_form(df: pd.DataFrame) -> bool:
    cols = set(df.columns)
    return (
        (("y_hat" in cols) or ("y_pred" in cols) or ("pred" in cols))
        and (("y_actual" in cols) or ("actual" in cols))
    ) or (
        ("Date" in cols or "DateTime" in cols)
        and (("y_hat" in cols) or ("y_actual" in cols) or ("pred" in cols) or ("actual" in cols))
    )


def _normalize_date_col(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    date_col = None
    for c in ["Date", "DateTime"]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        return df
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    # normalize to date if datetime
    df["__Date__"] = df[date_col].dt.normalize()
    return df


def _align_wide(
    pred_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    product_col: str,
    horizons: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Align wide dataframes by product_col and horizons."""
    # keep only needed columns
    p = pred_df[[product_col] + horizons].copy()
    a = actuals_df[[product_col] + horizons].copy()

    # align products intersection
    common = p.merge(a[[product_col]], on=product_col, how="inner")[product_col]
    p = p[p[product_col].isin(common)]
    a = a[a[product_col].isin(common)]

    # sort same order by product
    p = p.sort_values(product_col).reset_index(drop=True)
    a = a.sort_values(product_col).reset_index(drop=True)

    # numeric coercion
    p[horizons] = _to_numeric_df(p[horizons])
    a[horizons] = _to_numeric_df(a[horizons])

    yhat = p[horizons].to_numpy().ravel()
    y = a[horizons].to_numpy().ravel()
    mask = ~np.isnan(yhat) & ~np.isnan(y)
    return yhat[mask], y[mask]


def _align_long(
    pred_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    product_col: str,
    pred_col_candidates: List[str] = ["y_hat", "y_pred", "pred"],
    act_col_candidates: List[str] = ["y_actual", "actual"],
) -> Tuple[np.ndarray, np.ndarray]:
    """Align long-format predictions and actuals by (product, date)."""
    p = _normalize_date_col(pred_df)
    a = _normalize_date_col(actuals_df)

    # pick column names
    pred_col = next((c for c in pred_col_candidates if c in p.columns), None)
    act_col = next((c for c in act_col_candidates if c in a.columns), None)
    if pred_col is None:
        raise KeyError("Prediction column not found in pred_df. Expected one of: " + str(pred_col_candidates))
    if act_col is None:
        raise KeyError("Actual column not found in actuals_df. Expected one of: " + str(act_col_candidates))

    # ensure date col
    if "__Date__" not in p.columns or "__Date__" not in a.columns:
        raise KeyError("Both pred_df and actuals_df must have 'Date' or 'DateTime' columns in long format.")

    key_cols = [product_col, "__Date__"]
    m = p[key_cols + [pred_col]].merge(a[key_cols + [act_col]], on=key_cols, how="inner")
    m[pred_col] = pd.to_numeric(m[pred_col], errors="coerce")
    m[act_col] = pd.to_numeric(m[act_col], errors="coerce")

    m = m.dropna(subset=[pred_col, act_col])
    return m[pred_col].to_numpy(), m[act_col].to_numpy()


def compute_forecast_metrics(
    pred_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    horizons: Optional[List[str]],
    product_col: str,
) -> Dict[str, float]:
    """
    Compute forecast error metrics.

    Parameters
    ----------
    pred_df : pd.DataFrame
        Either wide (product + horizons) or long (product, date, y_hat).
    actuals_df : pd.DataFrame
        Either wide (product + horizons) or long (product, date, y_actual).
    horizons : List[str] or None
        Required for wide-vs-wide evaluation. Ignored for long-vs-long.
    product_col : str
        Product identifier column name.

    Returns
    -------
    Dict[str, float]
    """
    is_long_pred = _detect_long_form(pred_df)
    is_long_act = _detect_long_form(actuals_df)

    if is_long_pred and is_long_act:
        yhat, y = _align_long(pred_df, actuals_df, product_col)
    else:
        if not horizons:
            raise ValueError("horizons must be provided for wide-vs-wide forecast metrics.")
        yhat, y = _align_wide(pred_df, actuals_df, product_col, horizons)

    return _error_metrics(yhat, y)


# ---------------------------------------
# Planning metrics (from production plan)
# ---------------------------------------

def compute_planning_metrics(
    plan_df: pd.DataFrame,
    daily_capacity: float,
    feat_df: Optional[pd.DataFrame] = None,
    product_col: str = "Product_Number",
) -> Dict[str, object]:
    """
    Compute planning KPI from production plan.
    Expects columns: ['day_idx', 'demand', 'produce', 'backlog', 'end_inventory'].

    If feat_df with [product_col, 'Cluster'] is provided, returns cluster-level backlog rates.

    Returns
    -------
    {
      "FillRate": float,
      "BacklogRate": float,
      "Utilization": float,
      "Smoothness": float,
      "InventoryTurnover": float,
      "ClusterBacklog": (optional) dict[cluster] -> {demand, backlog, backlog_rate}
    }
    """
    required_cols = {"day_idx", "demand", "produce", "backlog", "end_inventory"}
    missing = required_cols - set(plan_df.columns)
    if missing:
        raise KeyError(f"plan_df is missing columns: {missing}")

    df = plan_df.copy()
    for c in ["demand", "produce", "backlog", "end_inventory"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    total_demand = float(df["demand"].sum())
    total_backlog = float(df["backlog"].sum())
    total_produce = float(df["produce"].sum())

    # Service level
    fill_rate = 1.0 - (total_backlog / (total_demand + _EPS))
    backlog_rate = total_backlog / (total_demand + _EPS)

    # Capacity utilization (average day production vs capacity)
    daily_prod = df.groupby("day_idx")["produce"].sum().sort_index()
    utilization = float(daily_prod.mean() / (daily_capacity + _EPS))

    # Production smoothness (absolute day-to-day change)
    smoothness = float(daily_prod.diff().abs().dropna().mean()) if len(daily_prod) > 1 else 0.0

    # Inventory turnover (produce sum / avg inventory)
    inv_turnover = float(total_produce / (df["end_inventory"].mean() + _EPS))

    out: Dict[str, object] = {
        "FillRate": fill_rate,
        "BacklogRate": backlog_rate,
        "Utilization": utilization,
        "Smoothness": smoothness,
        "InventoryTurnover": inv_turnover,
    }

    # Optional: cluster-level backlog rates
    if feat_df is not None and product_col in df.columns:
        if (product_col in feat_df.columns) and ("Cluster" in feat_df.columns):
            m = df.merge(
                feat_df[[product_col, "Cluster"]],
                on=product_col,
                how="left",
            )
            grp = m.groupby("Cluster")[["demand", "backlog"]].sum()
            grp["backlog_rate"] = grp["backlog"] / (grp["demand"] + _EPS)
            out["ClusterBacklog"] = grp.round(6).to_dict(orient="index")

    return out


# --------------------
# Minimal CLI (optional)
# --------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute metrics (demo).")
    parser.add_argument("--pred_csv", type=str, default=None)
    parser.add_argument("--actuals_csv", type=str, default=None)
    parser.add_argument("--plan_csv", type=str, default=None)
    parser.add_argument("--feat_csv", type=str, default=None)
    parser.add_argument("--product_col", type=str, default="Product_Number")
    parser.add_argument("--horizons", nargs="*", default=None)
    parser.add_argument("--daily_capacity", type=float, default=10000)
    args = parser.parse_args()

    if args.pred_csv and args.actuals_csv:
        pred = pd.read_csv(args.pred_csv)
        act = pd.read_csv(args.actuals_csv)
        fm = compute_forecast_metrics(pred, act, args.horizons, args.product_col)
        print("[Forecast metrics]")
        for k, v in fm.items():
            print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

    if args.plan_csv:
        plan = pd.read_csv(args.plan_csv)
        feat = pd.read_csv(args.feat_csv) if args.feat_csv else None
        pm = compute_planning_metrics(plan, args.daily_capacity, feat_df=feat, product_col=args.product_col)
        print("\n[Planning metrics]")
        for k, v in pm.items():
            if k == "ClusterBacklog":
                print("ClusterBacklog:")
                for ck, row in v.items():
                    print(f"  {ck}: {row}")
            else:
                print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")