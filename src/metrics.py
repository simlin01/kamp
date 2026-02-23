# src/metrics.py
# -*- coding: utf-8 -*-
"""
Metrics utilities for SCM pipeline:
- Forecast metrics: MAE, RMSE, WAPE, sMAPE, Bias (ME/MPE)
- Planning  metrics: FillRate, ShortageRate, BacklogLevelRate, Utilization, Smoothness, InventoryTurnover
- Optional cluster-level metrics when feat_df (Product_Number, Cluster) provided

[IMPORTANT UPDATE]
- Planning BacklogRate/FillRate are computed from "shortage" = backlog increase (new unmet demand),
  consistent with planner_opt / evaluator definition:
    shortage_{t} = max(backlog_t - backlog_{t-1}, 0), shortage_0 = backlog_0
  TotalShortage = sum(shortage)
  ShortageRate  = TotalShortage / TotalDemand
  FillRate      = 1 - ShortageRate

- We still report backlog "level" totals for reference (can be useful, but not for service-rate).
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import pandas as pd


# ---------------------------
# Helpers (safe numeric ops)
# ---------------------------

_EPS = 1e-9


def _to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _error_metrics(yhat: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    yhat = np.asarray(yhat, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    den = np.clip(np.abs(y), _EPS, None)

    err = yhat - y
    abs_e = np.abs(err)
    sq_e = err ** 2

    return {
        "MAE": float(abs_e.mean()),
        "RMSE": float(np.sqrt(sq_e.mean())),
        "WAPE": float(abs_e.sum() / (np.abs(y).sum() + _EPS)),
        "sMAPE": float((2 * abs_e / (np.abs(yhat) + np.abs(y) + _EPS)).mean()),
        "Bias_ME": float(err.mean()),
        "Bias_MPE": float((err / den).mean()),
    }


# ------------------------------------------------
# Forecast metrics
# ------------------------------------------------

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
    for c in ["Date", "DateTime"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            df["__Date__"] = df[c].dt.normalize()
            break
    return df


def _align_wide(
    pred_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    product_col: str,
    horizons: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    p = pred_df[[product_col] + horizons].copy()
    a = actuals_df[[product_col] + horizons].copy()

    common = p.merge(a[[product_col]], on=product_col, how="inner")[product_col]
    p = p[p[product_col].isin(common)].sort_values(product_col)
    a = a[a[product_col].isin(common)].sort_values(product_col)

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
) -> Tuple[np.ndarray, np.ndarray]:
    p = _normalize_date_col(pred_df)
    a = _normalize_date_col(actuals_df)

    pred_col = next(c for c in ["y_hat", "y_pred", "pred"] if c in p.columns)
    act_col = next(c for c in ["y_actual", "actual"] if c in a.columns)

    m = p[[product_col, "__Date__", pred_col]].merge(
        a[[product_col, "__Date__", act_col]],
        on=[product_col, "__Date__"],
        how="inner",
    )
    m[pred_col] = pd.to_numeric(m[pred_col], errors="coerce")
    m[act_col] = pd.to_numeric(m[act_col], errors="coerce")
    m = m.dropna()
    return m[pred_col].to_numpy(), m[act_col].to_numpy()


def compute_forecast_metrics(
    pred_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    horizons: Optional[List[str]],
    product_col: str,
) -> Dict[str, float]:
    if _detect_long_form(pred_df) and _detect_long_form(actuals_df):
        yhat, y = _align_long(pred_df, actuals_df, product_col)
    else:
        if not horizons:
            raise ValueError("horizons required for wide-vs-wide evaluation")
        yhat, y = _align_wide(pred_df, actuals_df, product_col, horizons)
    return _error_metrics(yhat, y)


# ---------------------------------------
# Planning metrics (UPDATED)
# ---------------------------------------

def _ensure_plan_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    planner_opt 출력이 버전마다 컬럼명이 조금씩 다를 수 있어서,
    최소 요구 컬럼을 안전하게 맞춰준다.
    """
    df = df.copy()

    # day index
    if "day_idx" not in df.columns:
        if "day" in df.columns:
            df["day_idx"] = df["day"]
        else:
            raise KeyError("plan_df missing 'day_idx' (or 'day')")

    # demand / produce
    if "demand" not in df.columns:
        raise KeyError("plan_df missing 'demand'")
    if "produce" not in df.columns:
        # 가끔 'production'으로 오는 경우
        if "production" in df.columns:
            df["produce"] = df["production"]
        else:
            raise KeyError("plan_df missing 'produce' (or 'production')")

    # backlog
    if "backlog" not in df.columns:
        # shortage만 있는 경우 backlog level이 없을 수 있음
        # 그 경우 backlog level totals는 0으로 처리
        df["backlog"] = 0.0

    # inventory
    if "end_inventory" not in df.columns:
        if "inventory" in df.columns:
            df["end_inventory"] = df["inventory"]
        elif "inv" in df.columns:
            df["end_inventory"] = df["inv"]
        else:
            # inventory가 아예 없으면 0으로
            df["end_inventory"] = 0.0

    return df


def _compute_shortage_from_backlog(
    df: pd.DataFrame,
    product_col: str,
) -> pd.Series:
    """
    shortage = backlog 증가분 (신규 미충족), shortage_0 = backlog_0
    df는 (product, day_idx) 기준으로 유일해야 가장 깔끔하지만,
    중복이 있어도 sum으로 묶어서 처리한다.
    """
    # (product, day) 단위로 backlog 레벨을 하나로 만들기
    g = (df.groupby([product_col, "day_idx"], dropna=False, as_index=False)["backlog"]
           .sum(numeric_only=True))

    g = g.sort_values([product_col, "day_idx"])
    # diff of backlog level per product
    g["prev_backlog"] = g.groupby(product_col)["backlog"].shift(1).fillna(0.0)
    inc = (g["backlog"] - g["prev_backlog"]).clip(lower=0.0)
    # day0에서 prev_backlog=0이므로 backlog_0이 그대로 shortage로 들어감
    g["shortage"] = inc
    # 원래 df index 길이에 맞출 필요는 없고, 총합/집계에 쓸 Series면 충분
    return g["shortage"]


def compute_planning_metrics(
    plan_df: pd.DataFrame,
    daily_capacity: float,
    feat_df: Optional[pd.DataFrame] = None,
    product_col: str = "Product_Number",
) -> Dict[str, object]:

    df = _ensure_plan_cols(plan_df)

    # normalize product id ('.0' 제거 등)
    if product_col in df.columns:
        df[product_col] = df[product_col].astype(str).str.replace(r"\.0$", "", regex=True)
    else:
        # product_col 없으면 shortage는 day-level로만 계산(제한적)
        # 그래도 파이프라인은 안 죽게 처리
        df[product_col] = "__ALL__"

    # numeric
    for c in ["demand", "produce", "backlog", "end_inventory"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["day_idx"] = pd.to_numeric(df["day_idx"], errors="coerce").fillna(0).astype(int)

    # totals
    total_demand = float(df["demand"].sum())
    total_produce = float(df["produce"].sum())
    backlog_level_total = float(df["backlog"].sum())  # (참고용, 레벨 합이라 과대해질 수 있음)

    # shortage-based service metrics (✅ evaluator와 정합)
    shortage = _compute_shortage_from_backlog(df, product_col=product_col)
    total_shortage = float(shortage.sum())
    shortage_rate = float(total_shortage / (total_demand + _EPS))
    fill_rate = float(1.0 - shortage_rate)

    # utilization / smoothness
    daily_prod = df.groupby("day_idx")["produce"].sum().sort_index()
    n_days = int(daily_prod.shape[0]) if daily_prod is not None else 0
    util_total = float(total_produce / ((daily_capacity + _EPS) * max(n_days, 1)))
    util_mean = float(daily_prod.mean() / (daily_capacity + _EPS)) if n_days > 0 else 0.0
    smoothness = float(daily_prod.diff().abs().dropna().mean()) if n_days > 1 else 0.0

    inv_mean = float(df["end_inventory"].mean())
    inv_turn = float(total_produce / (inv_mean + _EPS))

    out: Dict[str, object] = {
        # --- service (shortage-based)
        "FillRate": fill_rate,
        "ShortageRate": shortage_rate,
        "TotalShortage": total_shortage,

        # --- reference only (level-based; NOT for service evaluation)
        "BacklogLevelTotal": backlog_level_total,
        "BacklogLevelRate": float(backlog_level_total / (total_demand + _EPS)),

        # --- capacity / stability
        "Utilization_mean": util_mean,
        "Utilization_total": util_total,
        "Smoothness": smoothness,

        # --- inventory
        "AvgInventory": inv_mean,
        "InventoryTurnover": inv_turn,

        # --- totals
        "TotalDemand": total_demand,
        "TotalProduction": total_produce,
        "n_days": n_days,
    }

    # optional: per-day utilization series (debugging/plots)
    try:
        out["DailyUtilization"] = (daily_prod / (daily_capacity + _EPS)).round(6).to_dict()
    except Exception:
        pass

    # cluster-level backlog/shortage (safe merge)
    if feat_df is not None:
        if {"Cluster", product_col}.issubset(feat_df.columns):
            feat = feat_df[[product_col, "Cluster"]].copy()
            feat[product_col] = feat[product_col].astype(str).str.replace(r"\.0$", "", regex=True)
            feat = feat.drop_duplicates(subset=[product_col], keep="first")

            m = df.merge(feat, on=product_col, how="left")
            m["Cluster"] = pd.to_numeric(m["Cluster"], errors="coerce")
            m = m.dropna(subset=["Cluster"]).copy()
            m["Cluster"] = m["Cluster"].astype(int)

            if not m.empty:
                # cluster totals (demand, backlog level)
                grp = m.groupby("Cluster")[["demand", "backlog"]].sum(numeric_only=True)
                grp["backlog_level_rate"] = grp["backlog"] / (grp["demand"] + _EPS)

                # cluster shortage rate (by product within cluster)
                # compute shortage on (Cluster, Product, day)
                g2 = (m.groupby(["Cluster", product_col, "day_idx"], as_index=False)["backlog"].sum())
                g2 = g2.sort_values(["Cluster", product_col, "day_idx"])
                g2["prev"] = g2.groupby(["Cluster", product_col])["backlog"].shift(1).fillna(0.0)
                g2["shortage"] = (g2["backlog"] - g2["prev"]).clip(lower=0.0)

                dem2 = m.groupby("Cluster")["demand"].sum()
                sh2 = g2.groupby("Cluster")["shortage"].sum()
                cluster_short_rate = (sh2 / (dem2 + _EPS)).replace([np.inf, -np.inf], np.nan)

                out["ClusterKPI"] = {
                    int(k): {
                        "demand": float(grp.loc[k, "demand"]),
                        "backlog_level": float(grp.loc[k, "backlog"]),
                        "backlog_level_rate": float(grp.loc[k, "backlog_level_rate"]),
                        "total_shortage": float(sh2.get(k, 0.0)),
                        "shortage_rate": float(cluster_short_rate.get(k, np.nan)),
                    }
                    for k in grp.index.tolist()
                }

    return out


# --------------------
# CLI
# --------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute SCM metrics.")
    parser.add_argument("--pred_csv", type=str)
    parser.add_argument("--actuals_csv", type=str)
    parser.add_argument("--plan_csv", type=str)
    parser.add_argument("--feat_csv", type=str)
    parser.add_argument("--product_col", type=str, default="Product_Number")
    parser.add_argument("--horizons", nargs="*", default=None)
    parser.add_argument("--daily_capacity", type=float, default=10000)
    parser.add_argument("--out_json", type=str, default=None)

    # ✅ compatibility: when only planning metrics are computed, write "Planning" block only
    parser.add_argument(
        "--flat_planning",
        action="store_true",
        help="If set and only planning metrics are computed, save the Planning dict directly (not wrapped).",
    )

    args = parser.parse_args()
    results: Dict[str, object] = {}

    if args.pred_csv and args.actuals_csv:
        pred = pd.read_csv(args.pred_csv)
        act = pd.read_csv(args.actuals_csv)
        results["Forecast"] = compute_forecast_metrics(
            pred, act, args.horizons, args.product_col
        )

    if args.plan_csv:
        plan = pd.read_csv(args.plan_csv)
        feat = pd.read_csv(args.feat_csv) if args.feat_csv else None
        results["Planning"] = compute_planning_metrics(
            plan, args.daily_capacity, feat_df=feat, product_col=args.product_col
        )

    for sec, metrics in results.items():
        print(f"\n[{sec} metrics]")
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                print(f"{k}: {v}")
        else:
            print(metrics)

    if args.out_json:
        to_save = results
        if args.flat_planning and ("Forecast" not in results) and ("Planning" in results):
            to_save = results["Planning"]  # type: ignore

        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(to_save, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] Metrics saved to {args.out_json}")