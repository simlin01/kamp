# src/metrics.py
# -*- coding: utf-8 -*-
"""
Metrics utilities for SCM pipeline:
- Forecast metrics: MAE, RMSE, WAPE, sMAPE, Bias (ME/MPE)
- Planning  metrics: FillRate, ShortageRate, BacklogLevelRate, Utilization, Smoothness, InventoryTurnover
- Optional cluster-level metrics when feat_df (Product_Number, Cluster) provided

[IMPORTANT]
- Planning FillRate is computed from "shortage" = backlog increase (new unmet demand),
  consistent with planner_opt / evaluator definition:
    shortage_t = max(backlog_t - backlog_{t-1}, 0), shortage_0 = backlog_0
  TotalShortage = sum(shortage)
  ShortageRate  = TotalShortage / TotalDemand
  FillRate      = 1 - ShortageRate

[이번 수정 핵심]
1) Forecast metrics wide-form에서 horizons 자동탐지(미지정 시) 지원
2) Plan 컬럼 표준화 강화: shortage 컬럼이 있으면 backlog 없이도 service 계산 가능
3) CVaR mean plan(시나리오 평균 plan)에서 FillRate 왜곡 방지:
   - shortage 컬럼이 있으면 backlog diff로 다시 계산하지 말고 shortage를 우선 사용
   - shortage 합산은 항상 (product, day) 단위로 집계 후 총합으로 계산
4) Cluster KPI 계산 안전화
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import json
import re

import numpy as np
import pandas as pd

_EPS = 1e-9


# ---------------------------
# Helpers (safe numeric ops)
# ---------------------------

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


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def _extract_horizon_index(col: str) -> Optional[int]:
    c = _norm(col)
    if re.fullmatch(r"T", c):
        return 0
    m = re.fullmatch(r"T\+(\d+)", c)
    if m:
        return int(m.group(1))
    if "T일" in c:
        return 0
    m2 = re.search(r"T\+(\d+)\s*일", c)
    if m2:
        return int(m2.group(1))
    m3 = re.search(r"T\+(\d+)", c)
    if m3:
        return int(m3.group(1))
    return None


def detect_horizons_from_df(df: pd.DataFrame, product_col: str) -> List[str]:
    cands: List[str] = []
    for c in df.columns:
        if c == product_col:
            continue
        cc = _norm(c)
        if ("예상" in cc or "예정" in cc or "수주" in cc) and (("T일" in cc) or ("T+" in cc)):
            cands.append(c)
        elif re.fullmatch(r"T(\+\d+)?", cc):
            cands.append(c)
        elif re.search(r"T\+\d+", cc):
            cands.append(c)

    with_h = []
    no_h = []
    for c in cands:
        h = _extract_horizon_index(c)
        if h is not None:
            with_h.append((h, c))
        else:
            no_h.append((None, c))

    out = [c for _, c in sorted(with_h, key=lambda x: x[0])] + [c for _, c in no_h]
    out = list(dict.fromkeys(out))
    return out


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
            horizons = detect_horizons_from_df(pred_df, product_col=product_col)
        if not horizons:
            raise ValueError("horizons required for wide-vs-wide evaluation (auto-detect failed)")
        yhat, y = _align_wide(pred_df, actuals_df, product_col, horizons)
    return _error_metrics(yhat, y)


# ---------------------------------------
# Planning metrics
# ---------------------------------------

def _ensure_plan_cols(df: pd.DataFrame, product_col: str) -> pd.DataFrame:
    df = df.copy()

    # product col
    if product_col not in df.columns:
        for cand in ["Product_Number", "product", "SKU", "품번"]:
            if cand in df.columns:
                df[product_col] = df[cand]
                break
    if product_col not in df.columns:
        df[product_col] = "__ALL__"

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
        if "production" in df.columns:
            df["produce"] = df["production"]
        else:
            raise KeyError("plan_df missing 'produce' (or 'production')")

    # optional columns
    if "backlog" not in df.columns:
        df["backlog"] = np.nan  # 없다는 정보를 유지
    if "shortage" not in df.columns:
        df["shortage"] = np.nan
    if "end_inventory" not in df.columns:
        if "inventory" in df.columns:
            df["end_inventory"] = df["inventory"]
        elif "inv" in df.columns:
            df["end_inventory"] = df["inv"]
        else:
            df["end_inventory"] = 0.0

    return df


def _compute_shortage_from_backlog(df: pd.DataFrame, product_col: str) -> float:
    """
    backlog가 있을 때 shortage = backlog 증가분.
    반환: total_shortage (float)
    """
    g = (
        df.groupby([product_col, "day_idx"], dropna=False, as_index=False)["backlog"]
          .sum(numeric_only=True)
          .sort_values([product_col, "day_idx"])
    )
    g["prev_backlog"] = g.groupby(product_col)["backlog"].shift(1).fillna(0.0)
    g["shortage"] = (g["backlog"] - g["prev_backlog"]).clip(lower=0.0)
    return float(g["shortage"].sum())


def _compute_shortage_from_shortage_col(df: pd.DataFrame, product_col: str) -> float:
    """
    shortage 컬럼이 이미 plan_df에 있을 때(CVaR mean plan 등).
    (product, day) 중복이 있을 수 있으니 합친 뒤 총합.
    반환: total_shortage (float)
    """
    g = (
        df.groupby([product_col, "day_idx"], dropna=False, as_index=False)["shortage"]
          .sum(numeric_only=True)
    )
    return float(g["shortage"].sum())


def compute_planning_metrics(
    plan_df: pd.DataFrame,
    daily_capacity: float,
    feat_df: Optional[pd.DataFrame] = None,
    product_col: str = "Product_Number",
) -> Dict[str, object]:

    df = _ensure_plan_cols(plan_df, product_col=product_col)

    # normalize ids
    df[product_col] = df[product_col].astype(str).str.replace(r"\.0$", "", regex=True)
    df["day_idx"] = pd.to_numeric(df["day_idx"], errors="coerce").fillna(0).astype(int)

    # numeric
    for c in ["demand", "produce", "backlog", "shortage", "end_inventory"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["demand"] = df["demand"].fillna(0.0)
    df["produce"] = df["produce"].fillna(0.0)
    df["end_inventory"] = df["end_inventory"].fillna(0.0)

    # backlog/shortage는 없을 수도 있으므로 그대로 둠(na 유지)
    total_demand = float(df["demand"].sum())
    total_produce = float(df["produce"].sum())

    # reference only (level-based)
    backlog_level_total = float(df["backlog"].fillna(0.0).sum())

    # ✅ service metrics
    if df["shortage"].notna().any():
        total_shortage = _compute_shortage_from_shortage_col(df.fillna({"shortage": 0.0}), product_col=product_col)
    else:
        if df["backlog"].notna().any():
            total_shortage = _compute_shortage_from_backlog(df.fillna({"backlog": 0.0}), product_col=product_col)
        else:
            # shortage도 backlog도 없으면 service를 계산할 근거가 없음 → 0으로 두되 표시
            total_shortage = 0.0

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
        "FillRate": fill_rate,
        "ShortageRate": shortage_rate,
        "TotalShortage": total_shortage,

        "BacklogLevelTotal": backlog_level_total,
        "BacklogLevelRate": float(backlog_level_total / (total_demand + _EPS)),

        "Utilization_mean": util_mean,
        "Utilization_total": util_total,
        "Smoothness": smoothness,

        "AvgInventory": inv_mean,
        "InventoryTurnover": inv_turn,

        "TotalDemand": total_demand,
        "TotalProduction": total_produce,
        "n_days": n_days,
    }

    try:
        out["DailyUtilization"] = (daily_prod / (daily_capacity + _EPS)).round(6).to_dict()
    except Exception:
        pass

    # cluster-level KPI
    if feat_df is not None and isinstance(feat_df, pd.DataFrame):
        if {product_col, "Cluster"}.issubset(feat_df.columns):
            feat = feat_df[[product_col, "Cluster"]].copy()
            feat[product_col] = feat[product_col].astype(str).str.replace(r"\.0$", "", regex=True)
            feat = feat.drop_duplicates(subset=[product_col], keep="first")

            m = df.merge(feat, on=product_col, how="left")
            m["Cluster"] = pd.to_numeric(m["Cluster"], errors="coerce")
            m = m.dropna(subset=["Cluster"]).copy()
            m["Cluster"] = m["Cluster"].astype(int)

            if not m.empty:
                grp = m.groupby("Cluster")[["demand", "backlog"]].sum(numeric_only=True)
                grp["backlog_level_rate"] = grp["backlog"].fillna(0.0) / (grp["demand"] + _EPS)

                # cluster shortage totals
                if m["shortage"].notna().any():
                    sh = (
                        m.fillna({"shortage": 0.0})
                         .groupby("Cluster")["shortage"].sum(numeric_only=True)
                    )
                else:
                    # backlog diff within cluster/product
                    g2 = (
                        m.fillna({"backlog": 0.0})
                         .groupby(["Cluster", product_col, "day_idx"], as_index=False)["backlog"].sum()
                         .sort_values(["Cluster", product_col, "day_idx"])
                    )
                    g2["prev"] = g2.groupby(["Cluster", product_col])["backlog"].shift(1).fillna(0.0)
                    g2["shortage"] = (g2["backlog"] - g2["prev"]).clip(lower=0.0)
                    sh = g2.groupby("Cluster")["shortage"].sum(numeric_only=True)

                dem = m.groupby("Cluster")["demand"].sum(numeric_only=True)
                cluster_short_rate = (sh / (dem + _EPS)).replace([np.inf, -np.inf], np.nan)

                out["ClusterKPI"] = {
                    int(k): {
                        "demand": float(grp.loc[k, "demand"]),
                        "backlog_level": float(grp.loc[k, "backlog"]) if "backlog" in grp.columns else 0.0,
                        "backlog_level_rate": float(grp.loc[k, "backlog_level_rate"]),
                        "total_shortage": float(sh.get(k, 0.0)),
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
        results["Forecast"] = compute_forecast_metrics(pred, act, args.horizons, args.product_col)

    if args.plan_csv:
        plan = pd.read_csv(args.plan_csv)
        feat = pd.read_csv(args.feat_csv) if args.feat_csv else None
        results["Planning"] = compute_planning_metrics(plan, args.daily_capacity, feat_df=feat, product_col=args.product_col)

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