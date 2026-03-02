# src/report_llm.py
# -*- coding: utf-8 -*-
"""
report_llm.py (revised + fixed + executive-friendly)

- production_plan.csv / forecast_by_product.csv / metrics_final.csv 기반 주간 리포트 생성
- Facts(정량 집계)를 LLM에 제공 + Verifier Agent로 JSON 정합성 검증
- MC 검증 결과(mc_validation.json)를 Facts에 포함해 Markdown에 항상 표시
"""

from __future__ import annotations

import os
import json
import time
import argparse
import requests
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import numpy as np
import re
from pathlib import Path

# optional dependency
try:
    import markdown  # type: ignore
except Exception:
    markdown = None

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# =========================================================
# 유틸
# =========================================================
WEIRD_SPACES = ["\ufeff", "\u200b", "\u200c", "\u200d", "\xa0"]
_EPS = 1e-9


def _clean_str(s: str) -> str:
    s2 = str(s).strip()
    for w in WEIRD_SPACES:
        s2 = s2.replace(w, "")
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2


def _dedup_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()
    df.columns = [_clean_str(c) for c in df.columns]
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()
    return df


def _exists(path: str) -> bool:
    return bool(path) and os.path.exists(path)


def _read_clip_csv(path: str, max_rows: int = 50, max_chars: int = 8000) -> str:
    if not _exists(path):
        return f"[MISSING] {path}"
    df = pd.read_csv(path)
    head_txt = df.head(max_rows).to_csv(index=False)
    if len(head_txt) > max_chars:
        head_txt = head_txt[:max_chars] + f"\n...[truncated to {max_chars} chars]"
    return head_txt


def _topn(series: pd.Series, n: int = 5, largest: bool = True) -> List[Tuple[str, float]]:
    if series is None or getattr(series, "empty", True):
        return []
    ser = series.copy()
    ser = ser[~ser.isna()]
    if ser.empty:
        return []
    ser = ser.sort_values(ascending=not largest).iloc[:n]
    return [(str(idx), float(val)) for idx, val in ser.items()]


def _pick(cols_map: Dict[str, str], cands: List[str]) -> Optional[str]:
    """
    컬럼 선택 우선순위:
    1) 대소문자 무시 정확 일치
    2) 단어 경계(\b) 일치
    3) 부분문자열 일치 (prod/product_number 오인방지)
    """
    keys = list(cols_map.keys())
    lcands = [str(c).lower() for c in cands]

    # 1) exact
    for c in lcands:
        for k in keys:
            if k == c:
                return cols_map[k]

    # 2) word boundary
    for c in lcands:
        pat = re.compile(rf"\b{re.escape(c)}\b")
        for k in keys:
            if pat.search(k):
                return cols_map[k]

    # 3) substring
    for c in lcands:
        for k in keys:
            if c in k:
                if c in {"prod"} and "product_number" in k:
                    continue
                return cols_map[k]
    return None


def _load_json_if_exists(path: str) -> Optional[dict]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# =========================================================
# MC dict 호환 유틸
# =========================================================
def _mc_stat(s: dict, metric: str) -> Optional[dict]:
    if not isinstance(s, dict):
        return None
    v = s.get(metric)
    return v if isinstance(v, dict) else None


def _mc_get(s: dict, metric: str, stat: str, default=None):
    """
    Supports both:
      1) nested: s[metric][stat]
      2) flat:   s[f"{metric}_{stat_lower}"]  e.g., ShortageRate_var
    """
    if not isinstance(s, dict):
        return default

    # 1) nested form
    v = s.get(metric, None)
    if isinstance(v, dict):
        if stat in v:
            return v.get(stat, default)
        # allow lowercase aliases
        stat_alias = {"VaR": "var", "worst": "worst", "CVaR": "cvar", "mean": "mean"}.get(stat, None)
        if stat_alias and stat_alias in v:
            return v.get(stat_alias, default)

    # 2) flat form: Metric_stat
    stat_key = {
        "mean": "mean",
        "VaR": "var",
        "worst": "worst",
        "CVaR": "cvar",
    }.get(stat, stat.lower())

    flat_key = f"{metric}_{stat_key}"
    if flat_key in s:
        return s.get(flat_key, default)

    # 3) also support lowercase metric just in case
    flat_key2 = f"{metric.lower()}_{stat_key}"
    if flat_key2 in s:
        return s.get(flat_key2, default)

    return default


def _mc_pick_service_metric(summary: dict) -> Tuple[str, Optional[dict]]:
    if _mc_stat(summary, "ShortageRate"):
        return "ShortageRate", _mc_stat(summary, "ShortageRate")
    if _mc_stat(summary, "BacklogRate"):
        return "BacklogRate", _mc_stat(summary, "BacklogRate")
    return "ShortageRate", None


def _trim_mc_validation(mc: Optional[dict], keep_per_scenario: bool) -> Optional[dict]:
    """
    현업 보고서에서는 per_scenario(원시 리스트)가 너무 크고 노이즈가 될 수 있음.
    - 기본: summary만 유지
    - keep_per_scenario=True 일 때만 per_scenario 유지
    """
    if not isinstance(mc, dict):
        return mc
    out = dict(mc)
    # 보통 {"summary":..., "per_scenario":[...]} 형태
    if not keep_per_scenario:
        if "per_scenario" in out:
            out.pop("per_scenario", None)
    return out

def _normalize_mc_validation(mc: Optional[dict]) -> Optional[dict]:
    """Normalize MC validation JSON into a consistent shape used by the report.

    We support:
      - Old style: summary has nested metrics: summary[Metric] = {mean, VaR, worst, CVaR?}, and summary has FailRate.
      - New style (current evaluator): summary is mostly flat keys like ShortageRate_mean/var/cvar,
        and per_scenario rows contain raw values and fail flags.

    This function:
      1) Ensures mc['summary'] contains nested metrics dicts for the report table.
      2) Fills missing metrics (e.g., TotalDemand, InventoryRate) from per_scenario.
      3) Computes 'worst'(max) and 'FailRate' from per_scenario when missing.
    """
    if not isinstance(mc, dict) or not mc:
        return mc

    summary = mc.get("summary")
    per = mc.get("per_scenario")
    cfg = mc.get("config") or {}

    if summary is None and isinstance(mc, dict):
        # allow passing summary-only as mc
        summary = mc

    # If already nested and has FailRate, we still may want to backfill worst if missing.
    out_summary = dict(summary) if isinstance(summary, dict) else {}

    def _ensure_nested(metric: str) -> dict:
        d = out_summary.get(metric)
        if isinstance(d, dict):
            return d
        d = {}
        out_summary[metric] = d
        return d

    def _percentile(xs: list[float], q: float) -> float:
        if not xs:
            return float("nan")
        arr = np.asarray(xs, dtype=float)
        return float(np.quantile(arr, q))

    # 1) Flat -> nested for key metrics where evaluator writes *_mean/var/cvar
    # Map evaluator's 'var' to report's 'VaR', 'cvar' to 'CVaR'.
    flat_map = [
        ("ShortageRate", "ShortageRate"),
        ("WShortageRate", "WShortageRate"),
        ("loss", "Loss"),
        ("w_loss", "WLoss"),
    ]
    for flat_prefix, metric_name in flat_map:
        mean_k = f"{flat_prefix}_mean"
        var_k = f"{flat_prefix}_var"
        cvar_k = f"{flat_prefix}_cvar"
        if mean_k in out_summary or var_k in out_summary or cvar_k in out_summary:
            d = _ensure_nested(metric_name)
            if "mean" not in d and mean_k in out_summary:
                d["mean"] = out_summary.get(mean_k)
            if "VaR" not in d and var_k in out_summary:
                d["VaR"] = out_summary.get(var_k)
            if "CVaR" not in d and cvar_k in out_summary:
                d["CVaR"] = out_summary.get(cvar_k)

    # 2) Fill from per_scenario (mean/VaR/worst and FailRate)
    alpha = cfg.get("alpha", None)
    try:
        alpha = float(alpha)
    except Exception:
        alpha = None
    q = alpha if (alpha is not None and 0 < alpha < 1) else 0.9

    def _collect(key: str) -> list[float]:
        if not isinstance(per, list):
            return []
        xs = []
        for r in per:
            if not isinstance(r, dict):
                continue
            v = r.get(key, None)
            if v is None:
                continue
            try:
                xs.append(float(v))
            except Exception:
                continue
        return xs

    # Metrics present in per_scenario
    per_metrics = {
        "TotalDemand": "TotalDemand",
        "TotalShortage": "TotalShortage",
        "ShortageRate": "ShortageRate",
        "InventoryRate": "InventoryRate",
        "loss": "Loss",
        "WTotalDemand": "WTotalDemand",
        "WTotalShortage": "WTotalShortage",
        "WShortageRate": "WShortageRate",
        "WInventoryRate": "WInventoryRate",
        "w_loss": "WLoss",
    }

    for per_key, metric_name in per_metrics.items():
        xs = _collect(per_key)
        if not xs:
            continue
        d = _ensure_nested(metric_name)
        if "mean" not in d or d.get("mean") is None:
            d["mean"] = float(np.mean(xs))
        if "VaR" not in d or d.get("VaR") is None:
            d["VaR"] = _percentile(xs, q)
        if "worst" not in d or d.get("worst") is None:
            d["worst"] = float(np.max(xs))

    # 3) FailRate
    if "FailRate" not in out_summary or out_summary.get("FailRate") is None:
        if isinstance(per, list) and per:
            fail_flags = []
            # Prefer non-weighted fail if exists, else weighted fail.
            for r in per:
                if not isinstance(r, dict):
                    continue
                v = r.get("fail", None)
                if v is None:
                    v = r.get("w_fail", None)
                if v is None:
                    continue
                try:
                    fail_flags.append(float(v))
                except Exception:
                    continue
            if fail_flags:
                # if flags are 0/1, mean is rate. If they are booleans, float conversion works.
                out_summary["FailRate"] = float(np.mean(fail_flags))

    # 4) Convenience keys for downstream (backward compat)
    # Some code paths look for TotalDemand_mean style.
    for metric_name in ["TotalDemand", "TotalShortage", "AvgInventory", "InventoryRate"]:
        d = out_summary.get(metric_name)
        if isinstance(d, dict):
            for k, alias in [("mean", "mean"), ("VaR", "var"), ("worst", "worst"), ("CVaR", "cvar")]:
                if k in d:
                    out_summary[f"{metric_name}_{alias}"] = d[k]

    # Put back
    out = dict(mc)
    out["summary"] = out_summary
    return out




# =========================================================
# 1) Plan 요약 (제품별)
# =========================================================
def summarize_by_product(plan_csv: str, product_col_candidates=("product_number", "product", "제품")) -> Dict:
    """
    production_plan.csv를 제품 단위로 집계:
      - sum(produce, demand, backlog)
      - end_inventory: 제품별 마지막 day_idx 재고
      - BacklogRate = backlog / (demand + eps)
      - top_low_coverage: backlog가 0이면 coverage(재고/수요) 하위
      - top_overprod: end_inventory 상위
    """
    if not _exists(plan_csv):
        return {"missing": True, "path": plan_csv}

    df = _dedup_columns(pd.read_csv(plan_csv))
    cols = {c.lower(): c for c in df.columns}

    col_prod = _pick(cols, list(product_col_candidates))
    col_prodqty = _pick(cols, ["produce", "production", "생산"])
    col_dem = _pick(cols, ["demand", "수요"])
    col_back = _pick(cols, ["backlog", "백로그"])
    col_inv = _pick(cols, ["end_inventory", "inventory", "inv", "재고"])
    col_day = _pick(cols, ["day_idx", "day"])

    required = [col_prod, col_prodqty, col_dem]
    if any(x is None for x in required):
        return {
            "missing": False,
            "schema_error": True,
            "columns": list(df.columns),
            "picked": {
                "product": col_prod,
                "produce": col_prodqty,
                "demand": col_dem,
                "backlog": col_back,
                "inventory": col_inv,
                "day": col_day,
            },
        }

    df[col_prod] = df[col_prod].astype(str)

    for c in [col_prodqty, col_dem, col_back, col_inv]:
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")

    # 제품별 마지막 재고(end_inventory)
    inv_last_df = None
    if col_inv and col_day and col_inv in df.columns and col_day in df.columns:
        tmp = df[[col_prod, col_day, col_inv]].copy()
        tmp[col_day] = pd.to_numeric(tmp[col_day], errors="coerce")
        tmp = tmp.dropna(subset=[col_day])
        if not tmp.empty:
            tmp = tmp.sort_values([col_prod, col_day])
            last = tmp.groupby(col_prod, as_index=False).tail(1)
            inv_last_df = last[[col_prod, col_inv]].rename(
                columns={col_prod: "Product_Number", col_inv: "end_inventory"}
            )

    agg = {col_prodqty: "sum", col_dem: "sum"}
    if col_back:
        agg[col_back] = "sum"

    grp = df.groupby(col_prod, dropna=False, as_index=False).agg(agg)
    grp = grp.rename(columns={col_prod: "Product_Number", col_prodqty: "produce", col_dem: "demand"})
    if col_back:
        grp = grp.rename(columns={col_back: "backlog"})
    if "backlog" not in grp.columns:
        grp["backlog"] = 0.0

    if inv_last_df is not None:
        grp = grp.merge(inv_last_df, on="Product_Number", how="left")
    else:
        grp["end_inventory"] = 0.0

    grp["BacklogRate"] = grp["backlog"].fillna(0.0) / (grp["demand"].fillna(0.0) + _EPS)

    top_backlog_df = grp.sort_values("backlog", ascending=False).head(5).copy()
    top_backlog = top_backlog_df[["Product_Number", "backlog", "BacklogRate"]].to_dict(orient="records")
           
    min_coverage_value = None
    try:
        tmp_cov = grp.copy()
        tmp_cov = tmp_cov[tmp_cov["demand"].fillna(0.0) > 0].copy()
        tmp_cov["InvCoverage"] = tmp_cov["end_inventory"].fillna(0.0) / (tmp_cov["demand"].fillna(0.0) + _EPS)
        if not tmp_cov.empty:
            min_coverage_value = float(tmp_cov["InvCoverage"].min())
    except Exception:
        min_coverage_value = None

    top_low_coverage = []
    if len(top_backlog) > 0 and float(pd.Series([r.get("backlog", 0.0) for r in top_backlog]).fillna(0.0).sum()) == 0.0:
        tmp = grp.copy()
        tmp = tmp[tmp["demand"].fillna(0.0) > 0].copy()
        tmp["InvCoverage"] = tmp["end_inventory"].fillna(0.0) / (tmp["demand"].fillna(0.0) + _EPS)
        top_low_df = tmp.sort_values("InvCoverage", ascending=True).head(5).copy()
        top_low_coverage = top_low_df[["Product_Number", "InvCoverage", "end_inventory", "demand"]].to_dict(orient="records")

    grp["_over_score"] = grp["end_inventory"].fillna(0.0).astype(float)
    top_overprod_df = grp.sort_values("_over_score", ascending=False).head(5).copy()
    top_overprod = top_overprod_df[["Product_Number", "_over_score"]].rename(columns={"_over_score": "over_score"}).to_dict(orient="records")

    preview_cols = ["Product_Number", "produce", "demand", "backlog", "end_inventory", "BacklogRate"]
    table_preview = grp.sort_values("backlog", ascending=False).head(40)[preview_cols].to_dict(orient="records")

    return {
        "missing": False,
        "schema_error": False,
        "table_head": table_preview,
        "top_backlog": top_backlog,
        "top_overprod": top_overprod,
        "top_low_coverage": top_low_coverage,
        "min_coverage": min_coverage_value, 
    }


# =========================================================
# 2) Plan 요약 (단일 plan)
# =========================================================
def _summarize_single_plan(plan_csv: str) -> Dict:
    if not _exists(plan_csv):
        return {"missing": True, "path": plan_csv}

    df = _dedup_columns(pd.read_csv(plan_csv))
    cols = {c.lower(): c for c in df.columns}

    col_prod = _pick(cols, ["product_number", "product", "제품"])
    col_day = _pick(cols, ["day_idx", "day"])
    col_date_like = _pick(cols, ["date", "날짜", "horizon"])
    col_prod_qty = _pick(cols, ["produce", "production", "생산"])
    col_inv = _pick(cols, ["end_inventory", "inv", "inventory", "재고"])
    col_backlog = _pick(cols, ["backlog", "백로그"])
    col_capa = _pick(cols, ["capa", "capacity"])

    required = [col_prod, col_prod_qty]
    if any(x is None for x in required):
        return {"missing": False, "schema_error": True, "columns": list(df.columns)}

    df[col_prod] = df[col_prod].astype(str)
    for c in [col_prod_qty, col_inv, col_backlog, col_capa, col_day]:
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")

    # period
    period = {}
    if col_date_like and col_date_like in df.columns:
        s = df[col_date_like].astype(str)
        period = {"min": str(s.min()), "max": str(s.max()), "n_points": int(len(s))}
    elif col_day and col_day in df.columns and df[col_day].notna().any():
        period = {"min": str(int(df[col_day].min())), "max": str(int(df[col_day].max())), "n_points": int(len(df))}

    total_prod = float(df[col_prod_qty].fillna(0.0).sum())

    # inventory: 제품별 마지막 day_idx 재고 합
    total_inv = None
    avg_inv = None
    if col_inv and col_inv in df.columns:
        avg_inv = float(df[col_inv].fillna(0.0).mean())
        if col_day and col_day in df.columns and df[col_day].notna().any():
            tmp = df[[col_prod, col_day, col_inv]].dropna(subset=[col_day]).copy()
            tmp = tmp.sort_values([col_prod, col_day])
            last = tmp.groupby(col_prod, as_index=False).tail(1)
            total_inv = float(last[col_inv].fillna(0.0).sum())
        if total_inv is None:
            total_inv = float(avg_inv * df[col_prod].nunique())

    total_backlog = float(df[col_backlog].fillna(0.0).sum()) if (col_backlog and col_backlog in df.columns) else 0.0
    total_capa = None
    if col_capa and col_capa in df.columns:
        tc = float(df[col_capa].fillna(0.0).sum())
        total_capa = tc if tc > 0 else None

    # n_days
    n_days = None
    if col_day and col_day in df.columns:
        n_days = int(df[col_day].nunique(dropna=True))
    elif col_date_like and col_date_like in df.columns:
        n_days = int(df[col_date_like].nunique(dropna=True))

    avg_daily_capa = float(total_capa / n_days) if (n_days and total_capa is not None) else None

    # variability
    prod_variability = None
    if col_day and col_day in df.columns and df[col_day].notna().any():
        day_prod = df.groupby(col_day, dropna=False)[col_prod_qty].sum().sort_index()
        if len(day_prod) > 1:
            prod_variability = float(np.nanstd(day_prod.values))

    avg_utilization = float(total_prod / total_capa) if (total_capa is not None and total_capa > 0) else None
    util_target = 0.9
    util_deviation = float(abs(avg_utilization - util_target)) if avg_utilization is not None else None

    # top5
    g = df.groupby(col_prod, dropna=False)
    prod_backlog_sum = g[col_backlog].sum() if (col_backlog and col_backlog in df.columns) else pd.Series(dtype=float)

    prod_inv_last = None
    if col_inv and col_day and col_inv in df.columns and col_day in df.columns and df[col_day].notna().any():
        tmp = df[[col_prod, col_day, col_inv]].dropna(subset=[col_day]).copy()
        tmp = tmp.sort_values([col_prod, col_day])
        last = tmp.groupby(col_prod, as_index=False).tail(1)
        prod_inv_last = last.set_index(col_prod)[col_inv]

    top_increase = _topn(prod_backlog_sum, n=5, largest=True) if not prod_backlog_sum.empty else []
    if prod_inv_last is not None and not prod_inv_last.empty:
        top_overprod = _topn(prod_inv_last, n=5, largest=True)
    else:
        prod_prod_sum = g[col_prod_qty].sum()
        approx = prod_prod_sum - (prod_backlog_sum if not prod_backlog_sum.empty else 0.0)
        top_overprod = _topn(approx, n=5, largest=True)

    return {
        "missing": False,
        "schema_error": False,
        "period": period,
        "totals": {
            "total_production": total_prod,
            "total_inventory": total_inv,
            "avg_inventory": avg_inv,
            "total_backlog": total_backlog,
            "total_capa": total_capa,
            "avg_daily_capa": avg_daily_capa,
            "n_days": n_days,
        },
        "timeline": {
            "production_variability": prod_variability,
            "avg_utilization": avg_utilization,
            "util_target": util_target,
            "util_deviation": util_deviation,
        },
        "top5_increase_needed": [{"product": p, "sum_backlog": v} for p, v in top_increase],
        "top5_overproduction": [{"product": p, "score": v} for p, v in top_overprod],
        "columns": list(df.columns),
    }


def _pareto_frontier(items: List[Dict]) -> List[int]:
    if not items:
        return []
    dominated = set()
    for i, a in enumerate(items):
        if i in dominated:
            continue
        for j, b in enumerate(items):
            if i == j or j in dominated:
                continue
            conds = [
                (b["backlog"] <= a["backlog"]) if (a["backlog"] is not None and b["backlog"] is not None) else False,
                (b["variability"] <= a["variability"]) if (a["variability"] is not None and b["variability"] is not None) else False,
                (b["util_dev"] <= a["util_dev"]) if (a["util_dev"] is not None and b["util_dev"] is not None) else False,
            ]
            strict = [
                (b["backlog"] < a["backlog"]) if (a["backlog"] is not None and b["backlog"] is not None) else False,
                (b["variability"] < a["variability"]) if (a["variability"] is not None and b["variability"] is not None) else False,
                (b["util_dev"] < a["util_dev"]) if (a["util_dev"] is not None and b["util_dev"] is not None) else False,
            ]
            if all(conds) and any(strict):
                dominated.add(i)
                break
    return [i for i in range(len(items)) if i not in dominated]


def summarize_plans(plans: List[str], names: Optional[List[str]] = None) -> Dict:
    names = names or [f"scenario_{i+1}" for i in range(len(plans))]
    per = []
    for p, nm in zip(plans, names):
        s = _summarize_single_plan(p)
        per.append({"name": nm, "path": p, "summary": s})

    pts = []
    for it in per:
        s = it["summary"]
        backlog = s.get("totals", {}).get("total_backlog")
        variability = s.get("timeline", {}).get("production_variability")
        util_dev = s.get("timeline", {}).get("util_deviation")
        pts.append({"backlog": backlog, "variability": variability, "util_dev": util_dev})

    pareto_idx = _pareto_frontier(pts)
    for i, it in enumerate(per):
        it["pareto_frontier"] = (i in pareto_idx)

    return {"scenarios": per}


# =========================================================
# 3) Forecast metrics 요약
# =========================================================
def summarize_metrics(metrics_csv: str) -> Dict:
    if not _exists(metrics_csv):
        return {"missing": True, "path": metrics_csv}

    df = _dedup_columns(pd.read_csv(metrics_csv))
    cols = {c.lower(): c for c in df.columns}
    col_h = _pick(cols, ["horizon", "target", "label", "기간", "예상"])
    col_mae = _pick(cols, ["mae"])
    col_r2 = _pick(cols, ["r2"])

    if col_h and col_mae and col_r2:
        tmp = df[[col_h, col_mae, col_r2]].copy()
        tmp.columns = ["horizon", "mae", "r2"]
    else:
        first = df.columns[0]
        if first.lower().startswith("unnamed") or first.strip() == "":
            df = df.rename(columns={first: "horizon"})
            col_h = "horizon"
        else:
            col_h = first

        cols2 = {c.lower(): c for c in df.columns}
        col_mae = cols2.get("mae", _pick(cols2, ["mae"]))
        col_r2 = cols2.get("r2", _pick(cols2, ["r2"]))
        if not (col_h and col_mae and col_r2):
            return {"missing": False, "schema_error": True, "columns": list(df.columns)}

        tmp = df[[col_h, col_mae, col_r2]].copy()
        tmp.columns = ["horizon", "mae", "r2"]

    tmp["mae"] = pd.to_numeric(tmp["mae"], errors="coerce")
    tmp["r2"] = pd.to_numeric(tmp["r2"], errors="coerce")

    out = {
        "by_horizon": tmp.sort_values("horizon").to_dict(orient="records"),
        "avg_mae": float(tmp["mae"].mean(skipna=True)),
        "avg_r2": float(tmp["r2"].mean(skipna=True)),
        "best_horizon_by_r2": None,
        "best_horizon_by_mae": None,
    }
    try:
        out["best_horizon_by_r2"] = tmp.loc[tmp["r2"].idxmax(), "horizon"]
    except Exception:
        pass
    try:
        out["best_horizon_by_mae"] = tmp.loc[tmp["mae"].idxmin(), "horizon"]
    except Exception:
        pass
    return out


def summarize_forecast_by_product(forecast_csv: str) -> Dict:
    if not _exists(forecast_csv):
        return {"missing": True, "path": forecast_csv}
    df = _dedup_columns(pd.read_csv(forecast_csv))
    n_rows, n_cols = df.shape
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    stats = {}
    for c in numeric_cols[:20]:
        s = df[c]
        stats[c] = {"mean": float(s.mean(skipna=True)), "p50": float(s.quantile(0.5)), "p90": float(s.quantile(0.9))}
    return {"shape": [int(n_rows), int(n_cols)], "numeric_stats": stats, "columns": list(df.columns)}


# =========================================================
# 4) Prompt / LLM 호출
# =========================================================
SYS_PROMPT = (
    "You are a senior operations strategy analyst writing for executives.\n"
    "You must write concise, decision-oriented interpretation in Korean.\n"
    "\n"
    "NON-NEGOTIABLE RULES:\n"
    "1) Do NOT rewrite or repeat KPI tables, Top5 lists, Monte Carlo summaries, scenario tables, or any numeric blocks already in FACTS.\n"
    "2) Only write interpretation. Do not add report title or duplicate headings already present in the canonical report.\n"
    "3) When you reference any number, copy it verbatim from FACTS (no rounding, no recalculation).\n"
    "4) If deterministic totals show 'total_backlog = 0' BUT Monte Carlo indicates risk (ShortageRate/BacklogRate > 0 or FailRate > 0),\n"
    "   you MUST state that the point-estimate plan satisfies demand but under uncertainty there is unmet-demand risk.\n"
    "5) Never conclude '문제 없음/안정적' unless BOTH deterministic AND MC risk indicators support it.\n"
    "6) Avoid generic advice. Every action must cite at least one metric or product evidence from FACTS.\n"
    "\n"
    "OUTPUT FORMAT (strict):\n"
    "- First output ONLY JSON with the exact schema.\n"
    "- Then output a line with '---'\n"
    "- Then output ONLY these Markdown sections (and nothing else):\n"
    "  ## Executive Interpretation\n"
    "  ## Operational Risk Interpretation\n"
    "  ## Managerial Action Implications\n"
)

USER_TASK = """다음 'Facts'는 CSV/JSON에서 계산된 정량 사실입니다.
수치는 반드시 Facts의 값을 **그대로 복사**하여 사용하세요. (재계산/반올림/보정 금지)

- **JSON 파트(스키마 출력)**는 수치를 Facts에서 그대로 복사하세요. (재계산 금지)
- **Markdown 해석 파트**는 사람이 읽기 좋게 '표기'만 다듬어도 됩니다.
  - 큰 수: 천 단위 콤마 사용, 소수점 최대 2자리
  - 비율: 소수점 4자리 또는 % 표기(소수점 2자리) 중 가독성 좋은 방식
  - 단, 의미가 바뀌는 재계산/임의 보정은 금지

톤/강도 규칙(중요):
- Facts의 `traffic_light.overall`이 **GREEN**이면 전반적으로 '양호/안정적'으로 시작하고, 리스크는 **미세 조정/모니터링** 톤으로만 서술하세요. 과장/경고 단어(예: '관리 필요', '주의가 필요', '불확실성이 크다')는 금지하거나 반드시 '경미/제한적' 수식어를 붙이세요.
- **YELLOW**면 주의/개선 필요 표현은 가능하되, 과장 금지.
- **RED**일 때만 강한 경고/즉시 조치 표현을 사용하세요.

반드시 포함:
- Backlog는 절대량과 함께 `BacklogRate = total_backlog / TotalDemand(mean)`를 함께 언급하세요(가능하면 %로).

당신이 작성할 내용은 이미 출력되는 정량 리포트(표/나열)를 '다시 쓰는' 것이 아니라,
관리자가 바로 의사결정할 수 있도록 **의미-영향-조치**로 연결하는 해석 코멘터리입니다.

[Facts(JSON)]
{facts_json}

[샘플 미리보기]
{samples}

[TASK]
1) 아래 JSON 스키마를 **정확히** 출력 (값은 Facts에서 복사)
2) 이어서 '---' 이후에 아래 3개 섹션만 포함한 해석 Markdown 작성
   - ## Executive Interpretation
   - ## Operational Risk Interpretation
   - ## Managerial Action Implications

[JSON Schema - keys only]
{
  "summary": {
    "period_min": "string|null",
    "period_max": "string|null",
    "total_production": "number",
    "total_inventory": "number",
    "total_backlog": "number",
    "avg_daily_capa": "number|null",
    "key_takeaways": ["string", "..."]
  },
  "top5": {
    "increase_needed": [{"product":"string","sum_backlog":"number"}],
    "overproduction": [{"product":"string","score":"number"}]
  },
  "forecast_metrics": {
    "by_horizon": [{"horizon":"string|number","mae":"number","r2":"number"}],
    "avg_mae":"number",
    "avg_r2":"number",
    "best_horizon_by_r2":"string|number|null",
    "best_horizon_by_mae":"string|number|null"
  },
  "scenario_compare": {
    "table": [{"name":"string","total_backlog":"number","prod_variability":"number|null","avg_utilization":"number|null","pareto":true}]
  },
  "actions": ["string","string","string"],
  "risks": ["string","string"]
}
"""

REFLECT_PROMPT = """당신은 Verifier Agent입니다.
아래는 모델이 생성한 JSON과, 참조해야 하는 Facts입니다.
JSON/Markdown 해석이 Facts와 상충하거나 품질 이슈가 있으면 문제 목록을 한국어 bullet로 반환하세요.
문제 없으면 "OK"만 반환하세요.
[FACTS]
{facts_json}
[MODEL_JSON]
{model_json}
"""


@dataclass
class LLMConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_retries: int = 4
    retry_backoff_sec: float = 2.5

    backend: str = os.getenv("LLM_BACKEND", "openai")
    internal_url: str = os.getenv("INTERNAL_LLM_URL", "").strip()
    internal_base_url: str = os.getenv("INTERNAL_LLM_BASE_URL", "").strip()
    internal_api_key: str = os.getenv("INTERNAL_LLM_API_KEY", "").strip()


def _call_llm(messages, cfg: LLMConfig) -> str:
    def _to_openai_messages(msgs):
        out = []
        for m in msgs:
            role = getattr(m, "type", None) or getattr(m, "role", None) or ""
            role = role.lower()
            if role in ("system", "systemmessage"):
                r = "system"
            elif role in ("human", "humanmessage", "user"):
                r = "user"
            elif role in ("ai", "aimessage", "assistant"):
                r = "assistant"
            else:
                r = "user"
            content = getattr(m, "content", None)
            if content is None:
                content = str(m)
            out.append({"role": r, "content": content})
        return out

    last_err = None
    for i in range(cfg.max_retries):
        try:
            backend = (cfg.backend or os.getenv("LLM_BACKEND", "openai")).strip().lower()

            if backend == "internal":
                url = (cfg.internal_url or os.getenv("INTERNAL_LLM_URL", "")).strip()
                base = (cfg.internal_base_url or os.getenv("INTERNAL_LLM_BASE_URL", "")).strip()
                if not url:
                    if not base:
                        raise RuntimeError("INTERNAL_LLM_URL 또는 INTERNAL_LLM_BASE_URL 환경변수를 설정하세요.")
                    url = base.rstrip("/") + "/v1/chat/completions"

                headers = {"Content-Type": "application/json"}
                api_key = (cfg.internal_api_key or os.getenv("INTERNAL_LLM_API_KEY", "")).strip()
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"

                payload = {"model": cfg.model, "temperature": cfg.temperature, "messages": _to_openai_messages(messages)}
                resp = requests.post(url, headers=headers, json=payload, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY 환경변수를 설정하세요.")
            llm = ChatOpenAI(model=cfg.model, temperature=cfg.temperature)
            resp = llm.invoke(messages)
            return resp.content if hasattr(resp, "content") else str(resp)

        except Exception as e:
            last_err = e
            time.sleep(cfg.retry_backoff_sec * (i + 1))

    raise RuntimeError(f"LLM 호출 실패: {last_err}")


def _split_json_markdown(raw: str) -> Tuple[Optional[dict], str]:
    json_obj, md = None, ""
    try:
        start = raw.find("{")
        end = -1
        depth = 0
        for i, ch in enumerate(raw[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if start != -1 and end != -1:
            json_txt = raw[start : end + 1]
            json_obj = json.loads(json_txt)
            md_split = raw[end + 1 :].split("\n---\n", 1)
            md = md_split[1].strip() if len(md_split) == 2 else raw[end + 1 :].strip()
        else:
            md = raw
    except Exception:
        md = raw
    return json_obj, md


def _extract_llm_exec_commentary(md: str) -> str:
    if not md:
        return ""
    text = md.strip()

    def _grab(title: str) -> str:
        m = re.search(rf"^##\s*{re.escape(title)}\s*$([\s\S]*?)(?=^##\s|\Z)", text, flags=re.MULTILINE)
        if not m:
            return ""
        body = m.group(1).strip()
        if not body:
            return ""
        return f"## {title}\n\n{body}\n"

    parts = []
    for t in ["Executive Interpretation", "Operational Risk Interpretation", "Managerial Action Implications"]:
        sec = _grab(t)
        if sec:
            parts.append(sec)

    return "\n".join(parts).strip()

def _get_failrate(mc: dict, summary: dict) -> float:
    for k in ["FailRate", "fail_rate", "FailRate_mean", "failrate"]:
        v = summary.get(k, None) if isinstance(summary, dict) else None
        if v is not None:
            return float(v)
        v = mc.get(k, None) if isinstance(mc, dict) else None
        if v is not None:
            return float(v)
    return 0.0

# =========================================================
# HTML / Charts
# =========================================================
def _charts_from_facts_base64(facts: dict, charts_mode: str) -> List[Tuple[str, str]]:
    """
    charts_mode:
      - none: []
      - summary: Top5 bar 위주 (분포(hist) 없음)
      - dist: 분포(hist) 포함 (디버깅용)
    """

    COVERAGE_COLOR = "#3A7DFF"   # 시그니처 블루
    OVERPROD_COLOR = "#FF6B6B"   # 소프트 레드

    charts_mode = (charts_mode or "summary").strip().lower()
    if charts_mode == "none":
        return []

    try:
        import io
        import base64
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return []

    charts: List[Tuple[str, str]] = []

    def _fig_to_b64(fig) -> str:
        import io, base64
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _beautify_ax(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.25)
        ax.tick_params(axis="both", labelsize=9)

    # 1) MC 분포: dist일 때만
    if charts_mode == "dist":
        mc = facts.get("mc_validation") or {}
        per = mc.get("per_scenario") if isinstance(mc, dict) else None
        if isinstance(per, list) and len(per) > 0:
            try:
                sr = [float(x.get("ShortageRate")) for x in per if x.get("ShortageRate") is not None]
                if not sr:
                    sr = [float(x.get("BacklogRate")) for x in per if x.get("BacklogRate") is not None]
                if sr:
                    plt.figure()
                    plt.hist(sr, bins=20)
                    plt.title("MC ServiceRisk distribution")
                    plt.xlabel("ShortageRate (or BacklogRate)")
                    plt.ylabel("count")
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png", bbox_inches="tight", dpi=160)
                    plt.close()
                    charts.append(("MC 서비스 리스크 분포(Shortage/Backlog)", base64.b64encode(buf.getvalue()).decode("ascii")))
            except Exception:
                pass

            try:
                loss = [float(x.get("Loss")) for x in per if x.get("Loss") is not None]
                if loss:
                    plt.figure()
                    plt.hist(loss, bins=20)
                    plt.title("MC Loss distribution")
                    plt.xlabel("Loss")
                    plt.ylabel("count")
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png", bbox_inches="tight", dpi=160)
                    plt.close()
                    charts.append(("MC Loss 분포", base64.b64encode(buf.getvalue()).decode("ascii")))
            except Exception:
                pass

    # 2) 제품 Top5 bar (summary/dist 공통)
    ps = facts.get("product_summary") or {}
    low_cov = ps.get("top_low_coverage") or []
    if low_cov:
        try:
            labels = [str(d.get("Product_Number", "?")) for d in low_cov[:5]]
            vals = [float(d.get("InvCoverage", 0.0)) for d in low_cov[:5]]

            fig, ax = plt.subplots(figsize=(7.4, 3.2), constrained_layout=True)
            y = list(range(len(labels)))

            ax.barh(y, vals, color=COVERAGE_COLOR, alpha=0.9)
            ax.set_yticks(y)
            ax.set_yticklabels(labels)
            ax.invert_yaxis()
            ax.set_facecolor("#f8f9fb")
            ax.set_title("Top 5 Low Inventory Coverage (Inv/Demand)", fontsize=11, loc="left")
            ax.set_xlabel("Coverage", fontsize=10)

            # 값 라벨
            for i, v in enumerate(vals):
                ax.text(v, i, f" {v:.3f}", va="center", fontsize=9)

            _beautify_ax(ax)
            
            charts.append(("재고 커버리지 하위 Top5", _fig_to_b64(fig)))
        except Exception:
            pass

    top_over = ps.get("top_overprod") or []
    if top_over:
        try:
            labels = [str(d.get("Product_Number", "?")) for d in top_over[:5]]
            vals = [float(d.get("over_score", 0.0)) for d in top_over[:5]]

            fig, ax = plt.subplots(figsize=(7.4, 3.2), constrained_layout=True)
            y = list(range(len(labels)))

            ax.barh(y, vals, color=OVERPROD_COLOR, alpha=0.9)
            ax.set_yticks(y)
            ax.set_yticklabels(labels)
            ax.invert_yaxis()
            ax.set_facecolor("#f8f9fb")
            ax.set_title("Top 5 Overproduction Score", fontsize=11, loc="left")
            ax.set_xlabel("over_score", fontsize=10)

            for i, v in enumerate(vals):
                ax.text(v, i, f" {v:,.1f}", va="center", fontsize=9)

            _beautify_ax(ax)
            charts.append(("과잉 생산(재고) 상위 Top5", _fig_to_b64(fig)))
        except Exception:
            pass

    return charts


def md_to_html_with_charts(md_path: str, html_path: str, facts: Optional[dict] = None, title: str = "주간 운영 계획 보고서"):
    if markdown is None:
        raise RuntimeError("패키지 'markdown'이 필요합니다. pip install markdown")

    md_text = Path(md_path).read_text(encoding="utf-8")
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "toc"])

    charts_mode = "summary"
    if isinstance(facts, dict):
        charts_mode = (facts.get("_charts_mode") or "summary")

    chart_html = ""
    if isinstance(facts, dict) and facts:
        charts = _charts_from_facts_base64(facts, charts_mode=charts_mode)
        if charts:
            parts = ["<section>", "<h2>자동 생성 그래프</h2>"]
            for t, b64 in charts:
                parts.append(f"<h3>{t}</h3>")
                parts.append(
                    f'<img src="data:image/png;base64,{b64}" '
                    'style="max-width:100%;height:auto;border:1px solid #eaeaea;'
                    'border-radius:10px;box-shadow:0 6px 18px rgba(0,0,0,0.06);'
                    'padding:8px;background:#fff;margin:8px 0 18px 0;" />'
                )
            parts.append("</section>")
            chart_html = "\n".join(parts)

    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Malgun Gothic", sans-serif;
           max-width: 980px; margin: 40px auto; padding: 0 16px; line-height: 1.6; }}
    h1 {{ margin-top: 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ background: #f6f6f6; text-align: left; }}
    code, pre {{ background: #f3f3f3; padding: 2px 4px; border-radius: 4px; }}
    pre {{ padding: 12px; overflow: auto; }}
    hr {{ margin: 24px 0; }}
  </style>
</head>
<body>
{chart_html}
{body}
</body>
</html>
"""
    Path(html_path).write_text(html, encoding="utf-8")


# =========================================================
# Verifier
# =========================================================
def verify_report(model_json: dict, facts: dict, cfg: LLMConfig) -> Dict:
    sys = SystemMessage(content="You are a strict QA verifier. Follow the checklist and be unforgiving.")
    user = HumanMessage(
        content=REFLECT_PROMPT.format(
            facts_json=json.dumps(facts, ensure_ascii=False, indent=2),
            model_json=json.dumps(model_json, ensure_ascii=False, indent=2),
        )
    )
    out = _call_llm([sys, user], cfg)
    ok = out.strip().upper() == "OK"
    return {"ok": ok, "report": out.strip()}


# =========================================================
# Canonical renderer
# =========================================================
def _grade_traffic_light(facts: dict) -> dict:
    rep = facts.get("plan_summary_rep") or {}
    totals = rep.get("totals") or {}
    tl = rep.get("timeline") or {}
    pm = facts.get("planning_metrics") or {}

    total_backlog = float(totals.get("total_backlog", 0.0) or 0.0)

    avg_util = pm.get("Utilization_mean", pm.get("Utilization", None))
    if avg_util is None:
        avg_util = tl.get("avg_utilization")
    avg_util = float(avg_util) if avg_util is not None else None

    mc = facts.get("mc_validation") or {}
    s = mc.get("summary", mc) if isinstance(mc, dict) else {}
    mc = facts.get("mc_validation") or {}
    fail = _get_failrate(mc, s)

    service_metric_name, _ = _mc_pick_service_metric(s)
    sr_v = _mc_get(s, service_metric_name, "VaR", None)
    sr_worst = _mc_get(s, service_metric_name, "worst", None)

    def _lvl(v: str) -> int:
        return {"green": 0, "yellow": 1, "red": 2}.get(v, 1)

    total_demand = _mc_get(s, "TotalDemand", "mean", None)

    if total_demand is None:
        total_demand = _mc_get(mc, "TotalDemand", "mean", None)
    total_demand = float(total_demand) if total_demand is not None else None
    backlog_rate = (total_backlog / total_demand) if (total_demand and total_demand > 0) else None

    # 1) 1차 판정: BacklogRate 기준
    #    GREEN < 1%, YELLOW 1~3%, RED >= 3%
    if backlog_rate is not None:
        if backlog_rate >= 0.03:
            service = "red"
        elif backlog_rate >= 0.01:
            service = "yellow"
        else:
            service = "green"
    else:
        # fallback: TotalDemand 없을 때는 ShortageRate tail 기준으로만 판정(기존보다 완화)
        if sr_v is not None and float(sr_v) >= 0.10:
            service = "red"
        elif (sr_v is not None and float(sr_v) >= 0.075) or (sr_worst is not None and float(sr_worst) >= 0.10) or fail >= 0.30:
            service = "yellow"
        else:
            service = "green"

    # 2) MC tail risk 안전장치 블록 제거

    ps = facts.get("product_summary") or {}
    low_cov = ps.get("top_low_coverage") or []

    if low_cov:
        min_cov = float(low_cov[0].get("InvCoverage", 1.0))
    else:
        # top_low_coverage가 비어도 전체 품목 기준 min coverage 사용
        min_cov = ps.get("min_coverage", None)
        min_cov = float(min_cov) if min_cov is not None else None
    top_over = ps.get("top_overprod") or []
    top_over_1 = float(top_over[0].get("over_score", 0.0)) if top_over else 0.0

    if min_cov is not None and min_cov < 0.05:
        inventory = "red"
    elif (min_cov is not None and min_cov < 0.12) or top_over_1 >= 2500:
        inventory = "yellow"
    else:
        inventory = "green"

    if avg_util is None:
        capacity = "yellow"
    elif avg_util >= 0.98:
        capacity = "green"
    elif avg_util >= 0.92:
        capacity = "yellow"
    elif avg_util >= 0.80:
        capacity = "yellow"
    else:
        capacity = "red"

    ms = facts.get("metrics_summary") or {}
    by_h = ms.get("by_horizon") or []
    r2_t = None
    for r in by_h:
        h = str(r.get("horizon", ""))
        if "T일" in h:
            try:
                r2_t = float(r.get("r2"))
            except Exception:
                r2_t = None
            break
    if r2_t is None:
        forecast = "yellow"
    elif r2_t < 0.80:
        forecast = "red"
    elif r2_t < 0.88:
        forecast = "yellow"
    else:
        forecast = "green"

    dims = [service, inventory, capacity, forecast]
    n = len(dims)
    red_cnt = sum(1 for d in dims if d == "red")
    yellow_cnt = sum(1 for d in dims if d == "yellow")
    green_cnt = sum(1 for d in dims if d == "green")

    # 종합등급: 비율/분포 기반 산정
    avg_score = (2.0 * red_cnt + 1.0 * yellow_cnt) / max(n, 1)

    if avg_score >= 1.25:
        overall = "red"
    elif avg_score >= 0.50:
        overall = "yellow"
    else:
        overall = "green"

    return {
        "overall": overall,
        "dimensions": {"service": service, "inventory": inventory, "capacity": capacity, "forecast": forecast},
        "signals": {
            "total_backlog": total_backlog,
            "total_demand_mean": total_demand,
            "backlog_rate": backlog_rate,
            "avg_utilization": avg_util,
            "fail_rate": fail,
            "service_metric": service_metric_name,
            "service_VaR": sr_v,
            "service_worst": sr_worst,
            "min_coverage": min_cov,
            "top_over_1": top_over_1,
            "r2_t": r2_t,
            "grade_counts": {"red": red_cnt, "yellow": yellow_cnt, "green": green_cnt},
            "overall_avg_score": avg_score,
        },
    }


def _render_canonical_md(facts: dict) -> str:
    import math

    def _fmt(x, nd=1):
        try:
            if x is None or (isinstance(x, float) and math.isnan(x)):
                return "N/A"
            if isinstance(x, (int,)):
                return f"{x:,d}"
            return f"{float(x):,.{nd}f}"
        except Exception:
            return str(x)

    def _emoji(level: str) -> str:
        return {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(level, "🟡")

    rep = facts.get("plan_summary_rep") or {}
    totals = rep.get("totals") or {}
    period = rep.get("period") or {}
    tl = rep.get("timeline") or {}
    pm = facts.get("planning_metrics") or {}

    grade = _grade_traffic_light(facts)
    sig = grade.get("signals", {})

    md: List[str] = []
    md.append("# 주간 운영 계획 보고서")
    md.append("")
    md.append("## 1) Executive Summary")
    md.append("")
    md.append(f"- **종합 등급**: {_emoji(grade['overall'])} **{grade['overall'].upper()}**")
    md.append(f"- **총 생산량**: {_fmt(totals.get('total_production'), 1)}")
    if totals.get("total_capa") is not None:
        md.append(f"- **총 CAPA**: {_fmt(totals.get('total_capa'), 1)}")
    md.append(f"- **총 재고(제품별 마지막일 합)**: {_fmt(totals.get('total_inventory'), 1)}")
    md.append(f"- **총 백로그**: {_fmt(totals.get('total_backlog'), 1)}")
    if period.get("min") or period.get("max"):
        md.append(f"- **기간**: {period.get('min')} ~ {period.get('max')}")
    if totals.get("avg_daily_capa") is not None:
        md.append(f"- **평균 일일 CAPA**: {_fmt(totals.get('avg_daily_capa'), 1)}")

    util_mean = pm.get("Utilization_mean", pm.get("Utilization", None))
    if util_mean is None:
        util_mean = tl.get("avg_utilization")
    if util_mean is not None:
        md.append(f"- **설비 활용률(평균)**: {_fmt(util_mean, 4)}")
    md.append("")

    md.append("## 2) 신호등(traffic light) 진단")
    md.append("")
    md.append("| 구분 | 등급 | 근거(핵심 신호) |")
    md.append("|---|---|---|")
    md.append(
        f"| 서비스(납기/미충족) | {_emoji(grade['dimensions']['service'])} {grade['dimensions']['service'].upper()} | "
        f"BacklogRate={_fmt(sig.get('backlog_rate'),4)}, "
        f"{sig.get('service_metric','ShortageRate')}(VaR)={_fmt(sig.get('service_VaR'),4)}, "
        f"worst={_fmt(sig.get('service_worst'),4)}, FailRate={_fmt(sig.get('fail_rate'),4)}, "
        f"TotalBacklog={_fmt(sig.get('total_backlog'),1)} |"
        )
    md.append(
        f"| 재고(커버리지/과잉) | {_emoji(grade['dimensions']['inventory'])} {grade['dimensions']['inventory'].upper()} | "
        f"MinCoverage={_fmt(sig.get('min_coverage'),4)}, TopOver={_fmt(sig.get('top_over_1'),1)} |"
    )
    md.append(
        f"| 설비(활용률) | {_emoji(grade['dimensions']['capacity'])} {grade['dimensions']['capacity'].upper()} | "
        f"AvgUtil={_fmt(sig.get('avg_utilization'),4)} |"
    )
    md.append(
        f"| 예측(신뢰도) | {_emoji(grade['dimensions']['forecast'])} {grade['dimensions']['forecast'].upper()} | "
        f"R2(T일)={_fmt(sig.get('r2_t'),4)} |"
    )
    md.append("")

    mc = facts.get("mc_validation") or None

    if isinstance(mc, dict) and mc:
        s = mc.get("summary", mc)
        md.append("## 3) 수요 변동성 리스크 (Monte Carlo)")
        md.append("")
        md.append("| 지표 | mean | p90(VaR) | max(worst) |")
        md.append("|---|---:|---:|---:|")
        svc_name, _ = _mc_pick_service_metric(s)
        if _mc_stat(s, svc_name):
            md.append(f"| {svc_name} | {_fmt(_mc_get(s, svc_name, 'mean'),4)} | {_fmt(_mc_get(s, svc_name, 'VaR'),4)} | {_fmt(_mc_get(s, svc_name, 'worst'),4)} |")
        if _mc_stat(s, "InventoryRate"):
            md.append(f"| InventoryRate | {_fmt(_mc_get(s,'InventoryRate','mean'),4)} | {_fmt(_mc_get(s,'InventoryRate','VaR'),4)} | {_fmt(_mc_get(s,'InventoryRate','worst'),4)} |")
        if _mc_stat(s, "Loss"):
            md.append(f"| Loss | {_fmt(_mc_get(s,'Loss','mean'),4)} | {_fmt(_mc_get(s,'Loss','VaR'),4)} | {_fmt(_mc_get(s,'Loss','worst'),4)} |")
            cvar = _mc_get(s, "Loss", "CVaR", None)
            if cvar is not None:
                md.append(f"\n- **Loss CVaR(alpha)**: {_fmt(cvar,4)}")
        md.append("")
        md.append(f"- **FailRate**: {_fmt(s.get('FailRate', 0.0),4)}  (정의: 임계 기준을 넘는 시나리오 비율)")
        md.append("")

    ps = facts.get("product_summary") or {}
    low_cov = ps.get("top_low_coverage", []) or []
    top_over = ps.get("top_overprod", []) or []
    top_inc = ps.get("top_backlog", []) or []

    md.append("## 4) 품목(SKU) 불균형 포인트")
    md.append("")
    if low_cov:
        md.append("### 4.1 재고 커버리지 낮은 품목 (주의)")
        md.append("")
        md.append("| 품목 | 커버리지(재고/수요) | 재고 | 수요 |")
        md.append("|---|---:|---:|---:|")
        for d in low_cov[:5]:
            md.append(f"| {d.get('Product_Number','?')} | {_fmt(d.get('InvCoverage'),3)} | {_fmt(d.get('end_inventory'),1)} | {_fmt(d.get('demand'),1)} |")
        md.append("")
    else:
        md.append("### 4.1 증가 필요 품목(백로그 상위)")
        md.append("")
        md.append("| 품목 | 백로그 합 |")
        md.append("|---|---:|")
        for d in top_inc[:5]:
            md.append(f"| {d.get('Product_Number','?')} | {_fmt(d.get('backlog', d.get('sum_backlog',0.0)),1)} |")
        md.append("")

    md.append("### 4.2 과잉 생산(재고) 상위 품목")
    md.append("")
    md.append("| 품목 | 과잉 점수(over_score) |")
    md.append("|---|---:|")
    for d in top_over[:5]:
        md.append(f"| {d.get('Product_Number','?')} | {_fmt(d.get('over_score',0.0),1)} |")
    md.append("")

    ms = facts.get("metrics_summary") or {}
    rows = ms.get("by_horizon") or []
    if rows:
        md.append("## 5) 수요 예측 품질 (Forecast)")
        md.append("")
        md.append("| Horizon | MAE | R2 |")
        md.append("|---|---:|---:|")
        for r in rows:
            md.append(f"| {r.get('horizon')} | {_fmt(r.get('mae'),4)} | {_fmt(r.get('r2'),4)} |")
        md.append("")

    if isinstance(pm, dict) and pm:
        md.append("## 6) Planning KPI (metrics.py)")
        md.append("")
        for k in ["FillRate", "ShortageRate", "TotalShortage", "Utilization_mean", "Smoothness", "AvgInventory", "InventoryTurnover"]:
            if k in pm:
                nd = 6 if ("Rate" in k) else 2
                md.append(f"- **{k}**: {_fmt(pm.get(k), nd)}")
        md.append("")

    md.append("## 7) 권고 조치(이번 주/다음 주)")
    md.append("")
    acts = facts.get("rule_based_actions", []) or []
    if not acts:
        acts = ["주요 지표 이상 없음 — 계획 유지 및 예측 모니터링 지속"]
    md.append("### 7.1 이번 주 즉시 조치")
    md.append("")
    for a in acts[:3]:
        md.append(f"- {a}")
    md.append("")
    md.append("### 7.2 다음 주 개선(정책/모델)")
    md.append("")
    md.append("- 커버리지 하한선(최소 재고) 정책 도입 또는 planner 제약/파라미터 재검토")
    md.append("- MC 기반 운영 기준(mean→VaR 또는 mean+k·std) 비교 후 기준 방식 확정")
    md.append("")

    return "\n".join(md)


def _enforce_facts_on_json(js: dict, facts: dict) -> dict:
    if not isinstance(js, dict):
        return js

    rep = facts.get("plan_summary_rep") or {}
    totals = rep.get("totals") or {}
    js.setdefault("summary", {})
    for k in ["total_production", "total_inventory", "total_backlog", "avg_daily_capa"]:
        v = totals.get(k, None)
        if v is not None:
            js["summary"][k] = v

    period = rep.get("period") or {}
    js["summary"]["period_min"] = period.get("min")
    js["summary"]["period_max"] = period.get("max")

    ps = facts.get("product_summary") or {}
    inc = ps.get("top_backlog") or []
    over = ps.get("top_overprod") or []
    js.setdefault("top5", {})
    js["top5"]["increase_needed"] = [{"product": d.get("Product_Number"), "sum_backlog": d.get("backlog", d.get("sum_backlog"))} for d in inc]
    js["top5"]["overproduction"] = [{"product": d.get("Product_Number"), "score": d.get("over_score")} for d in over]

    sc = facts.get("plan_scenarios", {}).get("scenarios", [])
    table = []
    for it in sc:
        s = it.get("summary") or {}
        t = s.get("totals") or {}
        tl = s.get("timeline") or {}
        table.append({
            "name": it.get("name"),
            "total_backlog": t.get("total_backlog"),
            "prod_variability": tl.get("production_variability"),
            "avg_utilization": tl.get("avg_utilization"),
            "pareto": bool(it.get("pareto_frontier", False)),
        })
    js.setdefault("scenario_compare", {})
    js["scenario_compare"]["table"] = table

    ms = facts.get("metrics_summary") or {}
    js.setdefault("forecast_metrics", {})
    if ms and not ms.get("missing"):
        js["forecast_metrics"] = ms

    return js


def generate_rule_based_actions(facts: dict) -> List[str]:
    actions: List[str] = []
    rep = facts.get("plan_summary_rep", {})
    totals = rep.get("totals", {})
    tl = rep.get("timeline", {})
    pm = facts.get("planning_metrics", {}) or {}
    mc = facts.get("mc_validation", {}) or {}
    mc = facts.get("mc_validation") or {}
    summary = mc.get("summary", {}) if isinstance(mc, dict) else {}
    s = mc.get("summary", mc) if isinstance(mc, dict) else {}

    avg_util = pm.get("Utilization_mean", tl.get("avg_utilization"))
    smooth = pm.get("Smoothness", None)
    shortage_rate = pm.get("ShortageRate", None)
    total_inv = totals.get("total_inventory", None)

    svc_name, _ = _mc_pick_service_metric(s)
    svc_var = _mc_get(s, svc_name, "VaR", None)

    if avg_util is not None and float(avg_util) < 0.8:
        actions.append("라인 가동률 제고를 위해 CAPA 재배분 또는 잔업 계획 검토")

    if smooth is not None and float(smooth) > 500:
        actions.append("일별 생산 변동성(Smoothness) 완화를 위해 스무딩 파라미터/캠페인 조정")

    if svc_var is not None and float(svc_var) >= 0.10:
        actions.append(f"MC 기준 {svc_name} VaR이 높아(불확실성 하 미충족 리스크) 안전재고/커버리지 하한 정책 우선 적용")

    if shortage_rate is not None and float(shortage_rate) >= 0.20:
        actions.append("ShortageRate가 높아 공급 부족 리스크 → 클러스터 2/3 중심 생산 우선순위 재정렬")

    if total_inv is not None and float(total_inv) > 10000:
        actions.append("과잉 재고 감축 및 프로모션 전략 검토")

    if not actions:
        actions.append("주요 지표 이상 없음 — 계획 유지 및 예측 모니터링 지속")

    return actions


# =========================================================
# build_report
# =========================================================
def build_report_with_llm(
    plan_csv: str = "",
    forecast_csv: str = "",
    metrics_csv: str = "",
    mc_json: str = "",
    planning_metrics_json: str = "",
    plan_csvs: Optional[List[str]] = None,
    scenario_names: Optional[List[str]] = None,
    model_name: str = "gpt-4o-mini",
    feat_csv: str = "",
    max_head_rows: int = 40,
    max_chars: int = 6000,
    auto_regen_on_fail: bool = True,
    charts_mode: str = "summary",          
    keep_mc_per_scenario: bool = False,
) -> Dict:
    cfg = LLMConfig(model=model_name)

    if plan_csvs and len(plan_csvs) > 0:
        plans = plan_csvs
    elif plan_csv:
        plans = [plan_csv]
    else:
        plans = []

    plans_summary = summarize_plans(plans, names=scenario_names) if plans else {"scenarios": []}
    rep_sum = plans_summary["scenarios"][0]["summary"] if plans_summary["scenarios"] else {}

    metrics_sum = summarize_metrics(metrics_csv) if metrics_csv else {}
    forecast_sum = summarize_forecast_by_product(forecast_csv) if forecast_csv else {}
    product_summary = summarize_by_product(plans[0]) if plans else {}

    pm = _load_json_if_exists(planning_metrics_json) if planning_metrics_json else None
    mc_raw = _load_json_if_exists(mc_json) if mc_json else None
    mc_raw = _normalize_mc_validation(mc_raw)
    mc_summary = _trim_mc_validation(mc_raw, keep_per_scenario=keep_mc_per_scenario)

    facts = {
        "plan_scenarios": plans_summary,
        "plan_summary_rep": rep_sum,
        "metrics_summary": metrics_sum,
        "forecast_summary": forecast_sum,
        "product_summary": product_summary,
        "mc_validation": mc_summary,
        "planning_metrics": pm,
        "_charts_mode": charts_mode,
    }
    facts["rule_based_actions"] = generate_rule_based_actions(facts)

    samples: List[str] = []
    if plans:
        use_names = scenario_names or [f"scenario_{i+1}" for i in range(len(plans))]
        for nm, pth in zip(use_names, plans):
            samples.append(f"[PRODUCTION_PLAN: {nm}]\n{_read_clip_csv(pth, max_rows=max_head_rows, max_chars=max_chars)}")
    if forecast_csv:
        samples.append(f"[FORECAST_BY_PRODUCT]\n{_read_clip_csv(forecast_csv, max_rows=max_head_rows, max_chars=max_chars)}")
    if metrics_csv:
        samples.append(f"[FORECAST_METRICS]\n{_read_clip_csv(metrics_csv, max_rows=max_head_rows, max_chars=max_chars)}")

    sys = SystemMessage(content=SYS_PROMPT)
    facts_json_str = json.dumps(facts, ensure_ascii=False, indent=2)
    samples_str = "\n\n".join(samples) if samples else "[NO SAMPLES]"
    user_content = USER_TASK.replace("{facts_json}", facts_json_str).replace("{samples}", samples_str)
    user = HumanMessage(content=user_content)

    raw = _call_llm([sys, user], cfg)
    js, md = _split_json_markdown(raw)
    if js is not None:
        js = _enforce_facts_on_json(js, facts)

    canonical = _render_canonical_md(facts)
    llm_commentary = _extract_llm_exec_commentary(md)
    if llm_commentary:
        md = canonical + "\n\n---\n\n# 추가 해석(LLM)\n\n" + llm_commentary
    else:
        md = canonical

    verification = {"ok": True, "report": "OK"}
    regen = False
    if js is not None:
        verification = verify_report(js, facts, cfg)
        if auto_regen_on_fail and not verification["ok"]:
            reflect_user = HumanMessage(
                content=(user_content + "\n\n[Verifier Issues]\n" + verification["report"] + "\n\n위 문제를 모두 수정하여 다시 출력하세요.")
            )
            raw2 = _call_llm([sys, reflect_user], cfg)
            js2, md2 = _split_json_markdown(raw2)
            raw, js = raw2, js2
            if js is not None:
                js = _enforce_facts_on_json(js, facts)
            llm_commentary2 = _extract_llm_exec_commentary(md2 or "")
            md = canonical + ("\n\n---\n\n# 추가 해석(LLM)\n\n" + llm_commentary2 if llm_commentary2 else "")
            regen = True
            verification = verify_report(js if js else {}, facts, cfg)

    return {"json": js, "markdown": md, "raw": raw, "verify": verification, "regen": regen, "facts": facts}


# =========================================================
# CLI
# =========================================================
def _ensure_parent_dir(path: str):
    p = Path(path)
    if p.parent:
        p.parent.mkdir(parents=True, exist_ok=True)


def main():
    p = argparse.ArgumentParser(description="Weekly report generator (LLM-augmented)")
    p.add_argument("--plan", help="production_plan.csv 경로 (단일)")
    p.add_argument("--plans", help="쉼표(,)로 구분된 production_plan.csv 경로 목록")
    p.add_argument("--scenario_names", help="쉼표(,)로 구분된 시나리오 이름 목록 (plans와 동일 길이)")
    p.add_argument("--forecast", help="forecast_by_product.csv 경로")
    p.add_argument("--metrics", help="forecast_metrics.csv 경로")
    p.add_argument("--planning_metrics_json", default=None, help="metrics.py가 저장한 planning_metrics.json 경로(선택)")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--out_md", default="weekly_report.md")
    p.add_argument("--out_json", default="weekly_report.json")
    p.add_argument("--out_verify", default="weekly_report.verify.txt")
    p.add_argument("--no_regen", action="store_true", help="검증 실패 시 재생성 비활성화")
    p.add_argument("--feat", default="./data/feat.csv", help="(유지) features.py가 만든 feat.csv 경로")
    p.add_argument("--mc_json", default=None, help="evaluator가 저장한 MC 요약 JSON 경로")
    p.add_argument("--charts_mode", default="summary", choices=["none", "summary", "dist"],
                   help="HTML에 포함할 차트 수준: none(없음) | summary(Top5) | dist(분포까지)")
    p.add_argument("--keep_mc_per_scenario", action="store_true",
                   help="MC JSON의 per_scenario를 Facts에 유지(기본은 trim하여 summary만 사용)")

    args = p.parse_args()

    plans = [s.strip() for s in args.plans.split(",") if s.strip()] if args.plans else []
    names = [s.strip() for s in args.scenario_names.split(",") if s.strip()] if args.scenario_names else None

    out = build_report_with_llm(
        plan_csv=args.plan or "",
        plan_csvs=plans or None,
        scenario_names=names,
        forecast_csv=args.forecast or "",
        metrics_csv=args.metrics or "",
        model_name=args.model,
        feat_csv=args.feat or "",
        auto_regen_on_fail=not args.no_regen,
        mc_json=args.mc_json or "",
        planning_metrics_json=args.planning_metrics_json or "",
        charts_mode=args.charts_mode,
        keep_mc_per_scenario=bool(args.keep_mc_per_scenario),
    )

    if out.get("markdown") is not None:
        _ensure_parent_dir(args.out_md)
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write(out["markdown"])

        out_html = str(Path(args.out_md).with_suffix(".html"))
        try:
            _ensure_parent_dir(out_html)
            md_to_html_with_charts(args.out_md, out_html, facts=out.get("facts"), title="주간 운영 계획 보고서")
            print(f"[OK] Saved HTML:\n- {out_html}")
        except Exception as e:
            print("[ERROR] md_to_html_with_charts failed:", repr(e))

    if out.get("json") is not None:
        _ensure_parent_dir(args.out_json)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out["json"], f, ensure_ascii=False, indent=2)

    if out.get("verify") is not None:
        _ensure_parent_dir(args.out_verify)
        with open(args.out_verify, "w", encoding="utf-8") as f:
            v = out["verify"]
            f.write(("OK" if v.get("ok") else "NG") + "\n\n")
            f.write(v.get("report", ""))

    print(f"[OK] Saved:\n- {args.out_md}\n- {args.out_json}\n- {args.out_verify}\n(re-generated: {out.get('regen')})")


if __name__ == "__main__":
    main()
