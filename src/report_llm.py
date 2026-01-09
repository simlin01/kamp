# src/report_llm.py
# -*- coding: utf-8 -*-
"""
report_llm.py
- production_plan.csv / forecast_by_product.csv / metrics_final.csv 기반 주간 리포트 생성
- Facts(정량 집계)를 LLM에 제공 + Verifier Agent로 JSON 정합성 검증
- (NEW) MC 검증 결과(mc_validation.json)를 Facts에 포함해 Canonical Markdown에 항상 표시

중요 수정 사항
1) _pick 함수가 중복 정의되어 기존 로직이 덮어씌워지는 버그 제거 (단일 정의로 통합)
2) plan/feat/metrics 컬럼 선택 로직을 '정확일치→단어경계→부분문자열'로 안정화
3) summarize_metrics: metrics_final.csv가 'horizon' 컬럼이 없고 index가 horizon일 수 있음 → 대응
4) CLI에서 불필요하게 다시 facts 만들고 md를 덧붙이던 코드 제거 (build_report_with_llm 결과만 저장)
5) mc_json이 summary만 주는/전체 dict 주는 케이스 모두 안전하게 처리
"""

import os
import json
import time
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# =========================================================
# 유틸
# =========================================================
WEIRD_SPACES = ["\ufeff", "\u200b", "\u200c", "\u200d", "\xa0"]

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

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0

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

# =========================================================
# 1) Plan 요약 (제품별)
# =========================================================
def summarize_by_product(plan_csv: str, product_col_candidates=("product_number","product","제품")) -> Dict:
    """
    production_plan.csv를 제품 단위로 집계:
      - sum(produce, demand, backlog, end_inventory)
      - BacklogRate = backlog / (demand + 1e-9)
      - Top5 backlog, Top5 overproduction(= end_inventory 상위 또는 (produce - demand)+)
    반환: { "table_head": [...], "top_backlog": [...], "top_overprod": [...] }
    """
    if not _exists(plan_csv):
        return {"missing": True, "path": plan_csv}

    df = pd.read_csv(plan_csv)
    df = _dedup_columns(df)
    cols = {c.lower(): c for c in df.columns}

    col_prodname = _pick(cols, list(product_col_candidates))
    col_prodqty  = _pick(cols, ["produce","production","생산"])
    col_dem      = _pick(cols, ["demand","수요"])
    col_back     = _pick(cols, ["backlog","백로그"])
    col_inv      = _pick(cols, ["end_inventory","inventory","inv","재고"])

    print("[DEBUG] plan(by_product) columns:", list(df.columns))
    print("[DEBUG] picked(by_product) -> product:", col_prodname,
          "/ qty:", col_prodqty, "/ demand:", col_dem,
          "/ back:", col_back, "/ inv:", col_inv)

    required = [col_prodname, col_prodqty, col_dem]
    if any(x is None for x in required):
        return {
            "missing": False,
            "schema_error": True,
            "columns": list(df.columns),
            "picked": {
                "product": col_prodname, "produce": col_prodqty, "demand": col_dem,
                "backlog": col_back, "inventory": col_inv
            }
        }

    for c in [col_prodqty, col_dem, col_back, col_inv]:
        if c and c in df.columns and not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")

    agg_dict = {col_prodqty: "sum", col_dem: "sum"}
    if col_back: agg_dict[col_back] = "sum"
    if col_inv:  agg_dict[col_inv]  = "sum"

    grp = df.groupby(col_prodname, dropna=False, as_index=False).agg(agg_dict)

    rename_map = {col_prodname: "Product_Number", col_prodqty: "produce", col_dem: "demand"}
    if col_back: rename_map[col_back] = "backlog"
    if col_inv:  rename_map[col_inv]  = "end_inventory"
    grp = grp.rename(columns=rename_map)

    grp["Product_Number"] = grp["Product_Number"].astype(str)
    if "backlog" not in grp.columns:
        grp["backlog"] = 0.0
    if "end_inventory" not in grp.columns:
        grp["end_inventory"] = 0.0

    grp["BacklogRate"] = grp["backlog"] / (grp["demand"] + 1e-9)

    top_backlog_df = grp.sort_values("backlog", ascending=False).head(5).copy()
    top_backlog = top_backlog_df[["Product_Number", "backlog", "BacklogRate"]].to_dict(orient="records")

    # overproduction score
    over_col = "_over_score"
    over_score = grp["end_inventory"].fillna(0.0)
    if float(over_score.sum()) == 0.0 and {"produce","demand"} <= set(grp.columns):
        over_score = (grp["produce"].fillna(0.0) - grp["demand"].fillna(0.0)).clip(lower=0.0)
    grp[over_col] = over_score.fillna(0.0).astype(float)

    top_overprod_df = grp.sort_values(over_col, ascending=False).head(5).copy()
    top_overprod = (top_overprod_df[["Product_Number", over_col]]
                    .rename(columns={over_col: "over_score"})
                    .to_dict(orient="records"))

    preview_cols = [c for c in ["Product_Number","produce","demand","backlog","end_inventory","BacklogRate"] if c in grp.columns]
    table_preview = (grp.sort_values("backlog", ascending=False)
                       .head(40)[preview_cols]
                       .to_dict(orient="records"))

    return {
        "missing": False,
        "schema_error": False,
        "table_head": table_preview,
        "top_backlog": top_backlog,
        "top_overprod": top_overprod
    }

# =========================================================
# 2) Plan 요약 (단일 plan)
# =========================================================
def _summarize_single_plan(plan_csv: str) -> Dict:
    if not _exists(plan_csv):
        return {"missing": True, "path": plan_csv}

    df = pd.read_csv(plan_csv)
    df = _dedup_columns(df)
    cols = {c.lower(): c for c in df.columns}

    col_prod = _pick(cols, ["product_number", "product", "제품"])
    col_date = _pick(cols, ["date", "날짜", "horizon", "day", "day_idx"])
    col_prod_qty = _pick(cols, ["produce", "production", "생산"])
    col_inv = _pick(cols, ["end_inventory", "inv", "inventory", "재고"])
    col_backlog = _pick(cols, ["backlog", "백로그"])
    col_capa = _pick(cols, ["capa", "capacity"])

    print("[DEBUG] plan columns:", list(df.columns))
    print("[DEBUG] picked -> product:", col_prod, "/ date:", col_date,
          "/ qty:", col_prod_qty, "/ inv:", col_inv,
          "/ back:", col_backlog, "/ capa:", col_capa)

    required = [col_prod, col_prod_qty]
    if any(x is None for x in required):
        return {"missing": False, "schema_error": True, "columns": list(df.columns)}

    df[col_prod] = df[col_prod].astype(str)

    for c in [col_prod_qty, col_inv, col_backlog, col_capa]:
        if c and c in df.columns and not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].apply(_safe_float)

    # 기간 정보
    period = {}
    if col_date and col_date in df.columns:
        s = df[col_date].astype(str)
        period = {"min": str(s.min()), "max": str(s.max()), "n_points": int(len(s))}

    total_prod = float(df[col_prod_qty].sum())
    total_inv = float(df[col_inv].sum()) if col_inv else 0.0
    total_backlog = float(df[col_backlog].sum()) if col_backlog else 0.0
    total_capa = float(df[col_capa].sum()) if col_capa else 0.0

    if col_date:
        n_days = int(df[col_date].nunique())
    elif "day_idx" in df.columns:
        n_days = int(df["day_idx"].nunique())
    else:
        n_days = None

    avg_daily_capa = float(total_capa / n_days) if (n_days and total_capa) else None

    # 변동성
    if col_date:
        prod_timeline = df.groupby(col_date, dropna=False)[col_prod_qty].sum().sort_index()
        prod_variability = float(np.nanstd(prod_timeline.values)) if len(prod_timeline) else None
    else:
        prod_variability = None

    avg_utilization = float(total_prod / total_capa) if total_capa > 0 else None
    util_target = 0.9
    util_deviation = float(abs(avg_utilization - util_target)) if avg_utilization is not None else None

    # top5
    g = df.groupby(col_prod, dropna=False)
    prod_backlog_sum = g[col_backlog].sum() if col_backlog else pd.Series(dtype=float)
    prod_inv_sum = g[col_inv].sum() if col_inv else pd.Series(dtype=float)
    prod_prod_sum = g[col_prod_qty].sum()

    top_increase = _topn(prod_backlog_sum, n=5, largest=True) if not prod_backlog_sum.empty else []
    if not prod_inv_sum.empty:
        top_overprod = _topn(prod_inv_sum, n=5, largest=True)
    else:
        approx = (prod_prod_sum - (prod_backlog_sum if not prod_backlog_sum.empty else 0.0))
        top_overprod = _topn(approx, n=5, largest=True)

    capa_conflict_ratio = None
    if col_capa and col_date:
        day_prod = df.groupby(col_date)[col_prod_qty].sum()
        day_capa = df.groupby(col_date)[col_capa].sum()
        align = pd.concat([day_prod, day_capa], axis=1).dropna()
        if not align.empty:
            capa_conflict_ratio = float((align.iloc[:, 0] > align.iloc[:, 1]).mean())

    print("[DEBUG] totals -> prod:", total_prod, "inv:", total_inv,
          "backlog:", total_backlog, "capa:", total_capa, "avg_daily_capa:", avg_daily_capa)

    return {
        "missing": False,
        "schema_error": False,
        "period": period,
        "totals": {
            "total_production": total_prod,
            "total_inventory": total_inv,
            "total_backlog": total_backlog,
            "total_capa": total_capa,
            "avg_daily_capa": avg_daily_capa,
            "n_days": n_days
        },
        "timeline": {
            "production_variability": prod_variability,
            "avg_utilization": avg_utilization,
            "util_target": util_target,
            "util_deviation": util_deviation,
            "capa_conflict_ratio": capa_conflict_ratio,
        },
        "top5_increase_needed": [{"product": p, "sum_backlog": v} for p, v in top_increase],
        "top5_overproduction": [{"product": p, "score": v} for p, v in top_overprod],
        "columns": list(df.columns),
    }

def summarize_cluster_kpi(plan_csv: str, feat_csv: str) -> dict:
    """
    feat.csv의 cluster(0~3)와 production_plan.csv를 Product_Number 기준으로 매칭해
    클러스터별 KPI(총 생산량, 재고, 백로그, CAPA, 평균 활용도)를 계산한다.
    """
    if not (_exists(plan_csv) and _exists(feat_csv)):
        return {"missing": True, "reason": "file not found", "paths": [plan_csv, feat_csv]}

    plan = _dedup_columns(pd.read_csv(plan_csv))
    feat = _dedup_columns(pd.read_csv(feat_csv))

    def _pick_key(df):
        lc = {c.lower(): c for c in df.columns}
        for k in ["product_number", "product", "제품"]:
            if k in lc:
                return lc[k]
        for c in lc:
            if "product" in c:
                return lc[c]
        return None

    key_plan = _pick_key(plan)
    key_feat = _pick_key(feat)
    if key_plan is None or key_feat is None:
        return {"missing": False, "schema_error": True, "reason": "key not found",
                "plan_cols": list(plan.columns), "feat_cols": list(feat.columns)}

    cl_col = next((c for c in feat.columns if c.lower() == "cluster"), None)
    if cl_col is None:
        return {"missing": False, "schema_error": True, "reason": "cluster col not found",
                "feat_cols": list(feat.columns)}

    plan[key_plan] = plan[key_plan].astype(str)
    feat[key_feat] = feat[key_feat].astype(str)

    for c in ["produce", "end_inventory", "backlog", "capa"]:
        if c in plan.columns and not pd.api.types.is_numeric_dtype(plan[c]):
            plan[c] = pd.to_numeric(plan[c].astype(str).str.replace(",", ""), errors="coerce")

    merged = plan.merge(feat[[key_feat, cl_col]], left_on=key_plan, right_on=key_feat, how="left")
    merged[cl_col] = pd.to_numeric(merged[cl_col], errors="coerce")
    merged = merged[~merged[cl_col].isna()].copy()
    merged[cl_col] = merged[cl_col].astype(int)

    if merged.empty:
        return {"missing": True, "reason": "no rows after merge"}

    grp = (merged.groupby(cl_col)[["produce", "end_inventory", "backlog", "capa"]]
           .sum(numeric_only=True).reset_index()
           .rename(columns={cl_col: "cluster"}))

    grp["utilization"] = grp["produce"] / grp["capa"].replace({0: np.nan}) if "capa" in grp.columns else np.nan

    labels = {
        0: "비활발(저수요/저생산)",
        1: "소극적(생산 적음)",
        2: "보통(균형형)",
        3: "매우 활발(생산/수요 집중)"
    }
    grp["label"] = grp["cluster"].map(labels).fillna("N/A")

    return {"missing": False, "schema_error": False, "table": grp.to_dict(orient="records")}

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
# 3) Metrics / Forecast 요약
# =========================================================
def summarize_metrics(metrics_csv: str) -> Dict:
    """
    forecast.py에서 저장되는 metrics_final.csv는 보통:
      - index가 horizon(예: 'T일 예상 수주량')이고
      - columns: MAE, RMSE, R2, SMAPE, ...
    일 수도 있고,
      - horizon 컬럼이 따로 있을 수도 있음.
    둘 다 지원.
    """
    if not _exists(metrics_csv):
        return {"missing": True, "path": metrics_csv}

    df = pd.read_csv(metrics_csv)
    df = _dedup_columns(df)

    # case A: horizon 컬럼이 존재
    cols = {c.lower(): c for c in df.columns}
    col_h = _pick(cols, ["horizon", "target", "label", "기간", "예상"])
    col_mae = _pick(cols, ["mae"])
    col_r2 = _pick(cols, ["r2"])

    if col_h and col_mae and col_r2:
        tmp = df[[col_h, col_mae, col_r2]].copy()
        tmp.columns = ["horizon", "mae", "r2"]
    else:
        # case B: horizon이 index로 저장된 형태 (첫 컬럼이 index였던 경우)
        # 흔한 케이스: 첫 컬럼 이름이 Unnamed: 0 또는 비어있음
        first = df.columns[0]
        if first.lower().startswith("unnamed") or first.strip() == "":
            df = df.rename(columns={first: "horizon"})
            col_h = "horizon"
        else:
            col_h = first  # 그래도 첫 컬럼을 horizon으로 가정

        # mae/r2 컬럼 찾기
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
        stats[c] = {
            "mean": float(s.mean(skipna=True)),
            "p50": float(s.quantile(0.5)),
            "p90": float(s.quantile(0.9)),
        }
    return {"shape": [int(n_rows), int(n_cols)], "numeric_stats": stats, "columns": list(df.columns)}

# =========================================================
# 4) Prompt / LLM 호출
# =========================================================
SYS_PROMPT = (
    "You are an operations planning analyst AI.\n"
    "STRICTLY FOLLOW FACTS PROVIDED. If any number appears in FACTS, you MUST copy it verbatim.\n"
    "Do NOT invent or round to zero. Never replace a positive FACT with 0.\n"
    "Summarize supply-chain production plan and forecasting quality for a weekly executive report.\n"
    "Be concise, numeric, and actionable. Use bullet points, Korean language.\n"
    "First output ONLY JSON with the exact schema, then after a line with '---', output a Markdown block.\n"
    "**In the Markdown block, you MUST include an '행동 계획' section with 3-5 items.**\n"
)

USER_TASK = """다음의 '사전 정량 요약(Facts)'은 CSV에서 직접 계산된 사실입니다.
반드시 Facts의 수치를 **그대로 복사하여** 사용하세요. 추론/보정/생략 금지.
총 생산량, 총 백로그, 총 재고, 평균 일일 CAPA 등 핵심 KPI는 Facts에서 제공되는 값을 **그대로** 사용하세요.
제품별 Top5는 반드시 Facts.product_summary의 값을 그대로 사용하세요(제품명/숫자 그대로).
추가로 cluster_summary는 생산량/백로그/재고를 클러스터(0~3) 단위로 집계한 내용입니다.
**또한 cluster_summary가 제공되면 클러스터(0~3)별 KPI 표를 포함하고 간단히 해석하세요.**
Markdown 보고서에는 반드시 '행동 계획' 섹션(3-5개)을 포함하세요. 

[금지사항]
- Facts에 양(+)의 값이 있는데 0 또는 N/A로 표기하는 행위 금지
- 제품명을 임의로 0.0 등의 숫자로 치환 금지
- 스키마 키 이름/계층 변경 금지

[Facts(JSON)]
{facts_json}

[샘플 미리보기]
{samples}

[TASK]
1) 아래 스키마(JSON)를 **정확히** 출력 (값은 Facts를 복사)
2) 이어서 '---' 이후에 **Markdown 보고서** 작성 (KPI/Top5도 Facts 그대로)
3) 복수 시나리오가 제공되면 KPI 비교 표 + Pareto 프론티어(표/리스트) 포함

[JSON Schema - keys only]
{{
  "summary": {{
    "period_min": "string|null",
    "period_max": "string|null",
    "total_production": "number",
    "total_inventory": "number",
    "total_backlog": "number",
    "avg_daily_capa": "number|null",
    "key_takeaways": ["string", "..."]
  }},
  "top5": {{
    "increase_needed": [{{"product":"string","sum_backlog":"number"}}],
    "overproduction": [{{"product":"string","score":"number"}}]
  }},
  "forecast_metrics": {{
    "by_horizon": [{{"horizon":"string|number","mae":"number","r2":"number"}}],
    "avg_mae":"number",
    "avg_r2":"number",
    "best_horizon_by_r2":"string|number|null",
    "best_horizon_by_mae":"string|number|null"
  }},
  "scenario_compare": {{
    "table": [{{"name":"string","total_backlog":"number","prod_variability":"number|null","avg_utilization":"number|null","pareto":true}}]
  }},
  "actions": ["string","string","string"],
  "risks": ["string","string"]
}}
"""

REFLECT_PROMPT = """당신은 Verifier Agent입니다.
아래는 모델이 생성한 JSON과, 참조해야 하는 Facts입니다.
JSON이 Facts와 상충하거나 품질 이슈가 있으면 문제 목록을 한국어 bullet로 반환하세요.
문제 없으면 "OK"만 반환하세요.

[FACTS]
{facts_json}

[MODEL_JSON]
{model_json}

검증 체크리스트:
- 수치 일관성(총합/단위/음수 여부)
- Top5 선정 근거(백로그/재고 합 상위와 일치?)
- CAPA 충돌 여부 언급 누락?
- 시나리오 비교 표에 Pareto 표시가 Facts와 일치?
"""

@dataclass
class LLMConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_retries: int = 4
    retry_backoff_sec: float = 2.5

def _call_llm(messages, cfg: LLMConfig) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 환경변수를 설정하세요.")
    llm = ChatOpenAI(model=cfg.model, temperature=cfg.temperature)
    last_err = None
    for i in range(cfg.max_retries):
        try:
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
            json_txt = raw[start:end+1]
            json_obj = json.loads(json_txt)
            md_split = raw[end+1:].split("\n---\n", 1)
            md = md_split[1].strip() if len(md_split) == 2 else raw[end+1:].strip()
        else:
            md = raw
    except Exception:
        md = raw
    return json_obj, md

# =========================================================
# 5) Verifier Agent
# =========================================================
def verify_report(model_json: dict, facts: dict, cfg: LLMConfig) -> Dict:
    sys = SystemMessage(content="You are a strict QA verifier.")
    user = HumanMessage(content=REFLECT_PROMPT.format(
        facts_json=json.dumps(facts, ensure_ascii=False, indent=2),
        model_json=json.dumps(model_json, ensure_ascii=False, indent=2),
    ))
    out = _call_llm([sys, user], cfg)
    ok = out.strip().upper() == "OK"
    return {"ok": ok, "report": out.strip()}

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

    md: List[str] = []

    rep = facts.get("plan_summary_rep") or {}
    totals = rep.get("totals") or {}
    period = rep.get("period") or {}
    period_min = period.get("min")
    period_max = period.get("max")

    total_prod = totals.get("total_production", 0.0)
    total_inv  = totals.get("total_inventory", 0.0)
    total_back = totals.get("total_backlog", 0.0)
    avg_capa   = totals.get("avg_daily_capa", None)
    total_capa = totals.get("total_capa", 0.0)

    md.append("### 주간 운영 계획 보고서")
    md.append("")
    md.append("- **KPI 요약**")
    if period_min is not None or period_max is not None:
        md.append(f"  - 기간: {period_min} ~ {period_max}")
    md.append(f"  - 총 생산량: {_fmt(total_prod, 1)}")
    md.append(f"  - 총 재고: {_fmt(total_inv, 1)}")
    md.append(f"  - 총 백로그: {_fmt(total_back, 1)}")
    if avg_capa is not None:
        md.append(f"  - 평균 일일 CAPA: {_fmt(avg_capa, 1)}")
    if total_capa:
        md.append(f"  - 총 CAPA(합계): {_fmt(total_capa, 1)}")
    md.append("")

    mc = facts.get("mc_validation") or None
    if isinstance(mc, dict) and mc:
        s = mc.get("summary", mc)
        md.append("## Monte Carlo 검증 요약")
        br = s.get("BacklogRate")
        if isinstance(br, dict):
            md.append(
                f"- BacklogRate(mean/p90/max): "
                f"{float(br.get('mean', 0.0)):.4f} / {float(br.get('p90', 0.0)):.4f} / {float(br.get('max', 0.0)):.4f}"
            )
        md.append(f"- FailRate: {float(s.get('FailRate', 0.0)):.4f}")
        if "n_scenarios" in s:
            md.append(f"- 시나리오 수: {int(s.get('n_scenarios', 0))}")
        md.append("")

    md.append("### 행동 계획 (정량 기반)")
    for act in facts.get("rule_based_actions", []):
        md.append(f"- {act}")
    md.append("")

    ps = facts.get("product_summary") or {}
    top_inc  = ps.get("top_backlog", []) or []
    top_over = ps.get("top_overprod", []) or []

    md.append("- **Top 5 증가 필요 제품**")
    if top_inc:
        for d in top_inc:
            name = d.get("Product_Number", "?")
            val  = d.get("backlog", d.get("sum_backlog", 0))
            md.append(f"  - {name}: {_fmt(val, 1)}")
    else:
        md.append("  - (데이터 없음)")
    md.append("")

    md.append("- **Top 5 과잉 생산 제품**")
    if top_over:
        for d in top_over:
            name = d.get("Product_Number", "?")
            val  = d.get("over_score", 0)
            md.append(f"  - {name}: {_fmt(val, 1)}")
    else:
        md.append("  - (데이터 없음)")
    md.append("")

    ms = facts.get("metrics_summary") or {}
    rows = []
    for r in (ms.get("by_horizon") or []):
        rows.append((r.get("horizon"), r.get("mae"), r.get("r2")))
    if rows:
        md.append("- **예상 수주량 지표**")
        for h, mae, r2 in rows:
            md.append(f"  - {h} MAE: {_fmt(mae, 4)}, R2: {_fmt(r2, 4)}")
        md.append("")

    scs = facts.get("plan_scenarios", {}).get("scenarios", [])
    if scs:
        md.append("- **시나리오 비교**")
        md.append("  | 시나리오 이름 | 총 백로그 | 생산 변동성 | 평균 활용도 | 파레토 프론티어 |")
        md.append("  |----------------|------------|--------------|--------------|------------------|")
        for it in scs:
            name = it.get("name", "")
            s = it.get("summary", {}) or {}
            tt = s.get("totals", {}) or {}
            tl = s.get("timeline", {}) or {}
            md.append(
                f"  | {name} | {_fmt(tt.get('total_backlog'), 1)} | {_fmt(tl.get('production_variability'), 2)} | "
                f"{_fmt(tl.get('avg_utilization'), 2)} | {'예' if it.get('pareto_frontier') else '아니오'} |"
            )
        md.append("")

    cluster = facts.get("cluster_summary") or {}
    if cluster and not cluster.get("missing") and not cluster.get("schema_error"):
        try:
            dfc = pd.DataFrame(cluster.get("table", []))
            if not dfc.empty:
                md.append("- **클러스터별 생산 특성**")
                md.append("  | 클러스터 | 유형 | 총 생산량 | 총 재고 | 총 백로그 | 평균 활용도 |")
                md.append("  |---:|---|---:|---:|---:|---:|")
                for _, r in dfc.iterrows():
                    md.append(
                        f"  | {int(r['cluster'])} | {r.get('label','')} | {_fmt(r.get('produce',0),1)} | "
                        f"{_fmt(r.get('end_inventory',0),1)} | {_fmt(r.get('backlog',0),1)} | {_fmt(r.get('utilization',float('nan')),2)} |"
                    )
                md.append("")
        except Exception as e:
            md.append(f"_클러스터 표 생성 중 오류: {e}_")

    return "\n".join(md)

def _enforce_facts_on_json(js: dict, facts: dict) -> dict:
    if not isinstance(js, dict):
        return js

    rep = (facts.get("plan_summary_rep") or {})
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
    js["top5"]["increase_needed"] = [
        {"product": d.get("Product_Number"), "sum_backlog": d.get("backlog", d.get("sum_backlog"))}
        for d in inc
    ]
    js["top5"]["overproduction"] = [
        {"product": d.get("Product_Number"), "score": d.get("over_score")}
        for d in over
    ]

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
            "pareto": bool(it.get("pareto_frontier", False))
        })
    js.setdefault("scenario_compare", {})
    js["scenario_compare"]["table"] = table

    return js

def generate_rule_based_actions(facts: dict) -> List[str]:
    actions: List[str] = []
    plan = facts.get("plan_summary_rep", {}).get("timeline", {})
    totals = facts.get("plan_summary_rep", {}).get("totals", {})
    product_sum = facts.get("product_summary", {})

    avg_util = plan.get("avg_utilization")
    prod_var = plan.get("production_variability")
    total_backlog = totals.get("total_backlog", 0.0)
    total_inv = totals.get("total_inventory", 0.0)

    if avg_util is not None and avg_util < 0.8:
        actions.append("라인 가동률 제고를 위해 CAPA 재배분 또는 잔업 계획 검토")

    if product_sum and not product_sum.get("missing"):
        top5 = product_sum.get("top_backlog", [])
        if top5 and total_backlog:
            ratio = sum(float(item.get("backlog", 0)) for item in top5) / float(total_backlog)
            if ratio > 0.6:
                actions.append("상위 소수 품목에 대한 집중 생산 전략 수립")

    if prod_var is not None and prod_var > 500:
        actions.append("일별 생산 변동성을 완화하기 위한 생산 스무딩/캠페인 조정")

    if total_inv > 10000:
        actions.append("과잉 재고 감축 및 프로모션 전략 검토")

    if not actions:
        actions.append("주요 지표 이상 없음 — 계획 유지 및 예측 모니터링 지속")

    return actions

def _autoresolve_feat_path(feat_csv: str, plan_path: Optional[str]) -> str:
    if feat_csv and os.path.exists(feat_csv):
        return feat_csv
    cands = []
    if plan_path:
        plan_dir = os.path.dirname(os.path.abspath(plan_path))
        cands += [
            os.path.join(plan_dir, "feat.csv"),
            os.path.join(plan_dir, "..", "feat.csv"),
        ]
    cands += [
        "./outputs/outputs/feat.csv",
        "./outputs/feat.csv",
        "./data/feat.csv",
    ]
    for p in cands:
        if os.path.exists(p):
            print(f"[DEBUG] auto-picked feat.csv -> {p}")
            return p
    print("[DEBUG] feat.csv not found in common locations:", cands)
    return ""

# =========================================================
# 6) 메인 엔드포인트
# =========================================================
def build_report_with_llm(
    plan_csv: str = "",
    forecast_csv: str = "",
    metrics_csv: str = "",
    mc_json: str = "",
    plan_csvs: Optional[List[str]] = None,
    scenario_names: Optional[List[str]] = None,
    model_name: str = "gpt-4o-mini",
    feat_csv: str = "",
    max_head_rows: int = 40,
    max_chars: int = 6000,
    auto_regen_on_fail: bool = True
) -> Dict:
    """
    반환: {"json": <구조화 결과 or None>, "markdown": <문서>, "raw": <LLM원문>, "verify": {...}, "regen": bool}
    """
    cfg = LLMConfig(model=model_name)

    # Plans
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
    feat_path = _autoresolve_feat_path(feat_csv, plans[0] if plans else None)
    cluster_summary = summarize_cluster_kpi(plans[0], feat_path) if (plans and feat_path) else {"missing": True, "reason": "feat path unresolved"}
    print("[DEBUG] cluster_summary.flags:", {k: cluster_summary.get(k) for k in ["missing","schema_error","reason"]})

    mc_summary = None
    if mc_json and os.path.exists(mc_json):
        with open(mc_json, "r", encoding="utf-8") as f:
            mc_summary = json.load(f)

    facts = {
        "plan_scenarios": plans_summary,
        "plan_summary_rep": rep_sum,
        "metrics_summary": metrics_sum,
        "forecast_summary": forecast_sum,
        "product_summary": product_summary,
        "cluster_summary": cluster_summary,
        "mc_validation": mc_summary,
    }
    facts["rule_based_actions"] = generate_rule_based_actions(facts)

    print("[DEBUG] actions:", facts.get("rule_based_actions"))

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
    user = HumanMessage(content=USER_TASK.format(
        facts_json=json.dumps(facts, ensure_ascii=False, indent=2),
        samples="\n\n".join(samples) if samples else "[NO SAMPLES]"
    ))

    raw = _call_llm([sys, user], cfg)
    js, md = _split_json_markdown(raw)
    if js is not None:
        js = _enforce_facts_on_json(js, facts)

    canonical = _render_canonical_md(facts)
    md = canonical + ("\n\n---\n\n" + md if md else "")

    verification = {"ok": True, "report": "OK"}
    regen = False
    if js is not None:
        verification = verify_report(js, facts, cfg)
        if auto_regen_on_fail and not verification["ok"]:
            reflect_user = HumanMessage(content=(
                USER_TASK.format(
                    facts_json=json.dumps(facts, ensure_ascii=False, indent=2),
                    samples="\n\n".join(samples) if samples else "[NO SAMPLES]"
                )
                + "\n\n[Verifier Issues]\n"
                + verification["report"]
                + "\n\n위 문제를 모두 수정하여 다시 출력하세요."
            ))
            raw2 = _call_llm([sys, reflect_user], cfg)
            js2, md2 = _split_json_markdown(raw2)
            raw, js = raw2, js2
            if js is not None:
                js = _enforce_facts_on_json(js, facts)
            md = canonical + ("\n\n---\n\n" + (md2 or "") if (md2 or "") else "")
            regen = True
            verification = verify_report(js if js else {}, facts, cfg)

    return {"json": js, "markdown": md, "raw": raw, "verify": verification, "regen": regen}

# =========================================================
# 7) CLI
# =========================================================
from pathlib import Path

def _ensure_parent_dir(path: str):
    p = Path(path)
    if p.parent:
        p.parent.mkdir(parents=True, exist_ok=True)

def main():
    p = argparse.ArgumentParser(description="Weekly report generator (LLM-augmented, with Verifier & Scenarios)")
    p.add_argument("--plan", help="production_plan.csv 경로 (단일)")
    p.add_argument("--plans", help="쉼표(,)로 구분된 production_plan.csv 경로 목록")
    p.add_argument("--scenario_names", help="쉼표(,)로 구분된 시나리오 이름 목록 (plans와 동일 길이)")
    p.add_argument("--forecast", help="forecast_by_product.csv 경로")
    p.add_argument("--metrics", help="forecast_metrics.csv 경로")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--out_md", default="weekly_report.md")
    p.add_argument("--out_json", default="weekly_report.json")
    p.add_argument("--out_verify", default="weekly_report.verify.txt")
    p.add_argument("--no_regen", action="store_true", help="검증 실패 시 재생성 비활성화")
    p.add_argument("--feat", default="./data/feat.csv", help="features.py가 만든 feat.csv 경로")
    p.add_argument("--mc_json", default=None, help="evaluator가 저장한 MC 요약 JSON 경로")
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
    )

    if out.get("markdown") is not None:
        _ensure_parent_dir(args.out_md)
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write(out["markdown"])

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