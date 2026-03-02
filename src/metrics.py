# src/metrics.py
# -*- coding: utf-8 -*-
"""
Metrics utilities for SCM pipeline:
- Forecast metrics: MAE, RMSE, WAPE, sMAPE, Bias (ME/MPE)
- Planning  metrics: FillRate, ShortageRate, BacklogLevelRate, Utilization, Smoothness, InventoryTurnover
- Optional cluster-level metrics when feat_df (Product_Number, Cluster) provided
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import json
import re

import numpy as np
import pandas as pd

_EPS = 1e-9
PAST_MARKERS = ["작년", "전년", "지난해"]


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

    # normalize product ids
    p[product_col] = p[product_col].astype(str).str.replace(r"\.0$", "", regex=True)
    a[product_col] = a[product_col].astype(str).str.replace(r"\.0$", "", regex=True)

    common = sorted(set(p[product_col]).intersection(set(a[product_col])))
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

    # normalize product ids
    p[product_col] = p[product_col].astype(str).str.replace(r"\.0$", "", regex=True)
    a[product_col] = a[product_col].astype(str).str.replace(r"\.0$", "", regex=True)

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


def _is_past_ref(col: str) -> bool:
    cc = _norm(col)
    return any(m in cc for m in PAST_MARKERS)


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


def _within_key(col: str) -> int:
    c = _norm(col)
    if "예정" in c:
        return 0
    if "예상" in c:
        return 1
    return 9


def _filter_horizons_by_kind(cols: List[str], horizon_kind: str) -> List[str]:
    hk = str(horizon_kind).lower().strip()
    if hk not in ("planned", "expected", "both"):
        raise ValueError("--horizon_kind must be one of: planned, expected, both")

    cols = [_norm(c) for c in cols]
    cols = [c for c in cols if c and (not _is_past_ref(c))]

    def is_planned(c: str) -> bool:
        return "예정" in c

    def is_expected(c: str) -> bool:
        return "예상" in c

    if hk == "planned":
        return [c for c in cols if is_planned(c)]
    if hk == "expected":
        return [c for c in cols if is_expected(c)]

    by_h: Dict[int, List[str]] = {}
    for c in cols:
        h = _extract_horizon_index(c)
        if h is None:
            continue
        if ("예정" in c) or ("예상" in c):
            by_h.setdefault(h, []).append(c)

    out: List[str] = []
    for h in sorted(by_h.keys()):
        day_cols = sorted(by_h[h], key=lambda x: (_within_key(x), x))
        chosen = None
        for cc in day_cols:
            if "예정" in cc:
                chosen = cc
                break
        if chosen is None:
            chosen = day_cols[0]
        out.append(chosen)
    return out


def detect_horizons_from_df(
    df: pd.DataFrame,
    product_col: str,
    horizon_kind: str = "planned",
) -> List[str]:
    cands: List[str] = []
    for c in df.columns:
        if c == product_col:
            continue
        cc = _norm(c)
        if _is_past_ref(cc):
            continue

        # 한국어 기반
        if (("예상" in cc) or ("예정" in cc) or ("수주" in cc)) and (("T일" in cc) or ("T+" in cc)):
            cands.append(c)
            continue

        # 축약형 / 포함형
        if re.fullmatch(r"T(\+\d+)?", cc):
            cands.append(c)
            continue
        if re.search(r"T\+\d+", cc):
            cands.append(c)
            continue

    # horizon index로 정렬
    with_h: List[Tuple[int, str]] = []
    no_h: List[str] = []
    for c in cands:
        h = _extract_horizon_index(c)
        if h is not None:
            with_h.append((h, c))
        else:
            no_h.append(c)

    # 같은 horizon 내 planned→expected 우선(정렬 안정화)
    with_h_sorted = sorted(with_h, key=lambda x: (x[0], _within_key(x[1]), _norm(x[1])))
    ordered = [c for _, c in with_h_sorted] + [c for c in no_h]
    ordered = list(dict.fromkeys(ordered))

    # kind 필터
    filtered = _filter_horizons_by_kind(ordered, horizon_kind=horizon_kind)

    # kind 필터 결과가 비면, fallback으로 "kind 무시"하고라도 반환(단, 비과거만)
    if not filtered:
        # 그래도 최소한 과거는 제외된 ordered를 반환
        return ordered

    return filtered


def compute_forecast_metrics(
    pred_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    horizons: Optional[List[str]],
    product_col: str,
    horizon_kind: str = "planned",
) -> Dict[str, float]:
    if _detect_long_form(pred_df) and _detect_long_form(actuals_df):
        yhat, y = _align_long(pred_df, actuals_df, product_col)
    else:
        if not horizons:
            horizons = detect_horizons_from_df(pred_df, product_col=product_col, horizon_kind=horizon_kind)
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
        df["backlog"] = np.nan
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
    g = (
        df.groupby([product_col, "day_idx"], dropna=False, as_index=False)["backlog"]
          .sum(numeric_only=True)
          .sort_values([product_col, "day_idx"])
    )
    g["prev_backlog"] = g.groupby(product_col)["backlog"].shift(1).fillna(0.0)
    g["shortage"] = (g["backlog"] - g["prev_backlog"]).clip(lower=0.0)
    return float(g["shortage"].sum())


def _compute_shortage_from_shortage_col(df: pd.DataFrame, product_col: str) -> float:
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

    total_demand = float(df["demand"].sum())
    total_produce = float(df["produce"].sum())

    backlog_level_total = float(df["backlog"].fillna(0.0).sum())

    # service metrics
    if df["shortage"].notna().any():
        total_shortage = _compute_shortage_from_shortage_col(df.fillna({"shortage": 0.0}), product_col=product_col)
    else:
        if df["backlog"].notna().any():
            total_shortage = _compute_shortage_from_backlog(df.fillna({"backlog": 0.0}), product_col=product_col)
        else:
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

    parser.add_argument("--horizon_kind", type=str, default="planned", choices=["planned", "expected", "both"],
                        help="Wide-form forecast evaluation에서 자동탐지 시 사용할 horizon 종류. 기본 planned 권장.")
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

        horizons = args.horizons if args.horizons else None
        results["Forecast"] = compute_forecast_metrics(
            pred_df=pred,
            actuals_df=act,
            horizons=horizons,
            product_col=args.product_col,
            horizon_kind=args.horizon_kind,
        )

    if args.plan_csv:
        plan = pd.read_csv(args.plan_csv)
        feat = pd.read_csv(args.feat_csv) if args.feat_csv else None
        results["Planning"] = compute_planning_metrics(
            plan_df=plan,
            daily_capacity=args.daily_capacity,
            feat_df=feat,
            product_col=args.product_col,
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