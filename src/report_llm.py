# # src/report_llm.py

# [FILES]
# [PRODUCTION_PLAN_CSV]
# {plan_txt}

# [FORECAST_BY_PRODUCT_CSV]
# {forecast_txt}

# [FORECAST_METRICS_CSV]
# {metrics_txt}

# [TASK]
# 1) 요약(기간, CAPA, 총 수요/생산/백로그)
# 2) 제품 Top5: 증산 필요 / 과다생산
# 3) 예측성능: Horizon별 MAE/R2 요약
# 4) 액션아이템: 3~5개 (CAPA 재배분, 로트/안전재고 조정 등)
# 5) 리스크/가정: 데이터 품질/제약 가정 간단 표기
# """)

"""

CLI - 단일 플랜 (하위호환)
python -m src.report_llm \
  --plan ./outputs/production_plan.csv \
  --forecast ./outputs/pred_final.csv \
  --metrics ./outputs/metrics_final.csv \
  --model gpt-4o-mini \
  --out_md ./reports/weekly_report.md \
  --out_json ./reports/weekly_report.json
  --out_verify ./reports/weekly_report.verify.txt
"""


import os
import json
import time
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# =========================================================
# 유틸
# =========================================================
def _dedup_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()
    clean = []
    for c in df.columns:
        s = str(c).strip().replace("\ufeff","").replace("\u200b","").replace("\u200c","").replace("\u200d","").replace("\xa0","")
        clean.append(s)
    df.columns = clean
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

def _topn(series: pd.Series, n: int = 5, largest=True) -> List[Tuple[str, float]]:
    if series.empty:
        return []
    ser = series.copy()
    ser = ser[~ser.isna()]
    if ser.empty:
        return []
    ser = ser.sort_values(ascending=not largest)
    ser = ser.iloc[:n]
    return [(str(idx), float(val)) for idx, val in ser.items()]

def _pick(cols_map: Dict[str, str], cands: List[str]) -> Optional[str]:
    # 부분일치(대소문자 무시)
    for k in cols_map:
        if any(name.lower() in k for name in cands):
            return cols_map[k]
    return None

def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    real = [c for c in cols if c and c in df.columns]
    for c in real:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def summarize_by_product(plan_csv: str, product_col_candidates=("product_number","product","제품")) -> Dict:
    print("[DEBUG] plan.csv columns:", list(pd.read_csv(plan_csv).columns))
    """
    production_plan.csv를 제품 단위로 집계:
      - sum(produce, demand, backlog, end_inventory)
      - BacklogRate = backlog / (demand + 1e-9)
      - Top5 backlog, Top5 overproduction(= end_inventory 상위 또는 대안 점수)
    반환: { "table_head": [...], "top_backlog": [...], "top_overprod": [...] }
    """
    if not _exists(plan_csv):
        return {"missing": True, "path": plan_csv}

    df = pd.read_csv(plan_csv)
    cols = {c.lower(): c for c in df.columns}

    # 느슨한 매칭으로 컬럼 찾기
    col_prodname = _pick(cols, list(product_col_candidates))
    col_prod     = _pick(cols, ["prod","produce","production","생산"])
    col_dem      = _pick(cols, ["demand","수요"])
    col_back     = _pick(cols, ["backlog","백로그"])
    col_inv      = _pick(cols, ["inv","inventory","재고"])

    if not col_prodname or not col_prod or not col_dem:
        return {"missing": False, "schema_error": True, "columns": list(df.columns)}

    # 숫자형 변환(실존 컬럼만)
    df = _coerce_numeric(df, [col_prod, col_dem, col_back, col_inv])

    # 그룹핑 (as_index=False로 인덱스 충돌 회피)
    agg_dict = {col_prod: "sum", col_dem: "sum"}
    if col_back: agg_dict[col_back] = "sum"
    if col_inv:  agg_dict[col_inv]  = "sum"

    grp = df.groupby(col_prodname, dropna=False, as_index=False).agg(agg_dict)

    # 열명 표준화 (가능한 경우만)
    rename_map = {col_prodname: "Product_Number", col_prod: "produce", col_dem: "demand"}
    if col_back: rename_map[col_back] = "backlog"
    if col_inv:  rename_map[col_inv]  = "end_inventory"
    grp = grp.rename(columns=rename_map)

    # 만약 rename 후에도 Product_Number가 없다면, 자동 탐색으로 대체
    prod_col_final = "Product_Number" if "Product_Number" in grp.columns else \
                     _first_existing(grp, [col_prodname] + list(grp.columns))
    if prod_col_final != "Product_Number" and prod_col_final in grp.columns:
        grp = grp.rename(columns={prod_col_final: "Product_Number"})
        prod_col_final = "Product_Number"
    
    # 누락 컬럼 채움
    if "backlog" not in grp.columns:
        grp["backlog"] = 0.0
    if "end_inventory" not in grp.columns:
        grp["end_inventory"] = 0.0

    # 지표
    grp["BacklogRate"] = grp["backlog"] / (grp["demand"] + 1e-9)

    # Top 5 — 증산 필요(백로그 상위)
    # 실제 존재하는 컬럼만 안전하게 선택
    top_backlog_df = grp.sort_values("backlog", ascending=False).head(5).copy()
    top_backlog_df = _dedup_columns(top_backlog_df)
    safe_cols_back = [c for c in ["Product_Number", "backlog", "BacklogRate"] if c in top_backlog_df.columns]
    top_backlog = top_backlog_df[safe_cols_back].to_dict(orient="records")

    # Top 5 — 과다 생산(재고 상위; 재고 없으면 (produce - demand)+ 근사)
    # 1) 안전하게 숫자형 보정
    for c in ["produce", "demand", "end_inventory"]:
        if c in grp.columns and not pd.api.types.is_numeric_dtype(grp[c]):
            grp[c] = pd.to_numeric(grp[c], errors="coerce")

    # 2) 항상 _over_score를 생성 (존재하지 않으면 0.0으로라도 채움)
    over_col = "_over_score"
    if "end_inventory" in grp.columns:
        over_score = grp["end_inventory"].fillna(0.0).copy()
    else:
        over_score = pd.Series(0.0, index=grp.index)

    # 재고가 전부 0/NaN이면 (produce - demand)+ 로 대체
    if float(over_score.fillna(0).sum()) == 0.0 and {"produce","demand"} <= set(grp.columns):
        approx = (grp["produce"].fillna(0.0) - grp["demand"].fillna(0.0)).clip(lower=0.0)
        over_score = approx

    # 반드시 붙인다 (길이 맞춰서)
    if len(over_score) != len(grp):
        over_score = pd.Series(0.0, index=grp.index)
    grp[over_col] = over_score.fillna(0.0).astype(float)

    # 3) 정렬 전에 혹시 모를 중복 컬럼 제거
    if grp.columns.duplicated().any():
        grp = grp.loc[:, ~grp.columns.duplicated()].copy()
    # (여기까지 오면 _over_score가 반드시 존재)
    top_overprod_df = grp.sort_values(over_col, ascending=False).head(5).copy()

    # 4) 안전 선택 + rename
    safe_cols_over = [c for c in ["Product_Number", over_col] if c in top_overprod_df.columns]
    if "Product_Number" not in safe_cols_over:
        # 제품 컬럼이 표준명 아닌 경우 대비
        prod_fallback = next((c for c in top_overprod_df.columns if c.lower() in {"product_number","product","제품"}), None)
        if prod_fallback:
            safe_cols_over = [prod_fallback, over_col]
            top_overprod = (top_overprod_df[safe_cols_over]
                            .rename(columns={prod_fallback: "Product_Number", over_col: "over_score"})
                            .to_dict(orient="records"))
        else:
            # 제품 컬럼이 없다면 index를 이름으로 대체
            top_overprod_df = top_overprod_df.reset_index().rename(columns={"index":"Product_Number"})
            safe_cols_over = ["Product_Number", over_col]
            top_overprod = (top_overprod_df[safe_cols_over]
                            .rename(columns={over_col: "over_score"})
                            .to_dict(orient="records"))
    else:
        top_overprod = (top_overprod_df[safe_cols_over]
                        .rename(columns={over_col: "over_score"})
                        .to_dict(orient="records"))
    # 프리뷰 테이블(상위 40행) — 존재하는 컬럼만
    preview_cols = [c for c in ["Product_Number","produce","demand","backlog","end_inventory","BacklogRate"] if c in grp.columns]
    table_preview_df = grp.sort_values("backlog", ascending=False).head(40)[preview_cols].copy()
    table_preview_df = _dedup_columns(table_preview_df)
    table_preview = table_preview_df.to_dict(orient="records")

    return {
        "missing": False,
        "schema_error": False,
        "table_head": table_preview,
        "top_backlog": top_backlog,
        "top_overprod": top_overprod
    }

# =========================================================
# 1) Plan 요약 + (신규) 다중 시나리오 KPI / Pareto
# =========================================================
def _summarize_single_plan(plan_csv: str) -> Dict:
    if not _exists(plan_csv):
        return {"missing": True, "path": plan_csv}

    df = pd.read_csv(plan_csv)
    cols = {c.lower(): c for c in df.columns}

    col_prod = _pick(cols, ["product_number", "product", "제품"])
    col_date = _pick(cols, ["date", "날짜", "horizon", "day"])
    col_prod_qty = _pick(cols, ["생산", "prod", "production"])
    col_inv = _pick(cols, ["재고", "inv", "inventory"])
    col_backlog = _pick(cols, ["백로그", "backlog"])
    col_capa = _pick(cols, ["capa", "capacity"])

    required = [col_prod, col_prod_qty]
    if any(x is None for x in required):
        return {"missing": False, "schema_error": True, "columns": list(df.columns)}

    # 숫자형 보정
    for c in [col_prod_qty, col_inv, col_backlog, col_capa]:
        if c and not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].apply(_safe_float)

    # 기간
    period = {}
    if col_date:
        s = df[col_date].astype(str)
        period = {"min": str(s.min()), "max": str(s.max()), "n_points": int(len(s))}

    # 총합 KPI
    total_prod = float(df[col_prod_qty].sum())
    total_inv = float(df[col_inv].sum()) if col_inv else 0.0
    total_backlog = float(df[col_backlog].sum()) if col_backlog else 0.0
    total_capa = float(df[col_capa].sum()) if col_capa else 0.0

    # 타임라인 총생산 & 변동성
    prod_timeline = None
    if col_date:
        prod_timeline = df.groupby(col_date, dropna=False)[col_prod_qty].sum().sort_index()
        prod_variability = float(np.nanstd(prod_timeline.values)) if len(prod_timeline) else None
    else:
        prod_variability = None

    # 평균 가동률(= 총생산/총CAPA), 목표(0.9)에 대한 편차
    avg_utilization = float(total_prod / total_capa) if total_capa > 0 else None
    util_target = 0.9
    util_deviation = float(abs(avg_utilization - util_target)) if avg_utilization is not None else None

    # 제품별 집계 (Top5)
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

    # CAPA 충돌(생산 > CAPA) 비율
    capa_conflict_ratio = None
    if col_capa and col_date:
        day_prod = df.groupby(col_date)[col_prod_qty].sum()
        day_capa = df.groupby(col_date)[col_capa].sum()
        align = pd.concat([day_prod, day_capa], axis=1).dropna()
        if not align.empty:
            conflict = (align.iloc[:, 0] > align.iloc[:, 1]).mean()
            capa_conflict_ratio = float(conflict)

    return {
        "missing": False,
        "schema_error": False,
        "period": period,
        "totals": {
            "total_production": total_prod,
            "total_inventory": total_inv,
            "total_backlog": total_backlog,
            "total_capa": total_capa,
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

def _pareto_frontier(items: List[Dict]) -> List[int]:
    """
    items: [{"backlog": float, "variability": float, "util_dev": float}, ...]
    최소화 기준 3개(backlog, variability, util_dev)로 파레토 비지배 해 찾기.
    반환: 파레토 인덱스 리스트(원본 순서 기준)
    """
    if not items:
        return []
    dominated = set()
    for i, a in enumerate(items):
        if i in dominated:
            continue
        for j, b in enumerate(items):
            if i == j or j in dominated:
                continue
            # b가 a를 엄격히 지배하는지(모두 <=, 하나는 <)
            conds = [
                b["backlog"] <= a["backlog"] if a["backlog"] is not None and b["backlog"] is not None else False,
                b["variability"] <= a["variability"] if a["variability"] is not None and b["variability"] is not None else False,
                b["util_dev"] <= a["util_dev"] if a["util_dev"] is not None and b["util_dev"] is not None else False,
            ]
            strict = [
                b["backlog"] < a["backlog"] if a["backlog"] is not None and b["backlog"] is not None else False,
                b["variability"] < a["variability"] if a["variability"] is not None and b["variability"] is not None else False,
                b["util_dev"] < a["util_dev"] if a["util_dev"] is not None and b["util_dev"] is not None else False,
            ]
            if all(conds) and any(strict):
                dominated.add(i)
                break
    return [i for i in range(len(items)) if i not in dominated]

def summarize_plans(plans: List[str], names: Optional[List[str]] = None) -> Dict:
    """
    복수 시나리오의 production_plan.csv 요약 + Pareto.
    """
    names = names or [f"scenario_{i+1}" for i in range(len(plans))]
    per = []
    for p, nm in zip(plans, names):
        s = _summarize_single_plan(p)
        per.append({"name": nm, "path": p, "summary": s})

    # Pareto용 포인트 구성
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
# 2) Metrics / Forecast 요약 (기존)
# =========================================================
def summarize_metrics(metrics_csv: str) -> Dict:
    if not _exists(metrics_csv):
        return {"missing": True, "path": metrics_csv}
    df = pd.read_csv(metrics_csv)
    cols = {c.lower(): c for c in df.columns}
    def pick(name):
        for k in cols:
            if name in k:
                return cols[k]
        return None
    col_h = pick("horizon")
    col_mae = pick("mae")
    col_r2 = pick("r2")
    if not col_h or not col_mae or not col_r2:
        return {"missing": False, "schema_error": True, "columns": list(df.columns)}

    df = df[[col_h, col_mae, col_r2]].copy()
    df.columns = ["horizon", "mae", "r2"]
    df["mae"] = pd.to_numeric(df["mae"], errors="coerce")
    df["r2"] = pd.to_numeric(df["r2"], errors="coerce")

    out = {
        "by_horizon": df.sort_values("horizon").to_dict(orient="records"),
        "avg_mae": float(df["mae"].mean(skipna=True)),
        "avg_r2": float(df["r2"].mean(skipna=True)),
        "best_horizon_by_r2": None,
        "best_horizon_by_mae": None,
    }
    try:
        out["best_horizon_by_r2"] = df.loc[df["r2"].idxmax(), "horizon"]
    except Exception:
        pass
    try:
        out["best_horizon_by_mae"] = df.loc[df["mae"].idxmin(), "horizon"]
    except Exception:
        pass
    return out

def summarize_forecast_by_product(forecast_csv: str) -> Dict:
    if not _exists(forecast_csv):
        return {"missing": True, "path": forecast_csv}
    df = pd.read_csv(forecast_csv)
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
    return {
        "shape": [int(n_rows), int(n_cols)],
        "numeric_stats": stats,
        "columns": list(df.columns),
    }

# =========================================================
# 3) Prompt / LLM 호출
# =========================================================
SYS_PROMPT = (
    "You are an operations planning analyst AI. "
    "Summarize supply-chain production plan and forecasting quality for a weekly executive report. "
    "Be concise, numeric, and actionable. Use bullet points, Korean language. "
    "Return a JSON with fixed keys, then also a Markdown block."
)

USER_TASK = """다음의 '사전 정량 요약(Facts)'은 CSV에서 직접 계산된 사실입니다.
이 사실을 최우선으로 반영하여 보고서를 작성하세요.
추가로 제공되는 '샘플 미리보기'는 참고용이며, 길이 제한으로 인해 전체가 아닙니다.
제품별 요약(product_summary)의 Top5(backlog/overproduction)를 정확히 반영하세요.

[Facts(JSON)]
{facts_json}

[샘플 미리보기]
{samples}

[TASK]
1) 먼저 아래 스키마(JSON)를 **정확히** 출력
2) 이어서 '---' 이후에 **Markdown 보고서** 작성
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
    import os, time
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
# 4) Verifier Agent
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

# =========================================================
# 5) 메인 엔드포인트
# =========================================================
def build_report_with_llm(
    plan_csv: str = "",
    forecast_csv: str = "",
    metrics_csv: str = "",
    # 신규: 복수 시나리오 입력
    plan_csvs: Optional[List[str]] = None,
    scenario_names: Optional[List[str]] = None,
    model_name: str = "gpt-4o-mini",
    max_head_rows: int = 40,
    max_chars: int = 6000,
    auto_regen_on_fail: bool = True
) -> Dict:
    """
    반환: {"json": <구조화 결과 or None>, "markdown": <문서 or None>, "raw": <LLM원문>, "verify": {...}, "regen": bool}
    """
    cfg = LLMConfig(model=model_name)

    # ----- Plans (단일 또는 다중)
    if plan_csvs and len(plan_csvs) > 0:
        plans = plan_csvs
    elif plan_csv:
        plans = [plan_csv]
    else:
        plans = []

    plans_summary = summarize_plans(plans, names=scenario_names) if plans else {"scenarios": []}
    # summary 수준에서 공통/대표 KPI도 뽑아 LLM에 넘기기 쉽도록(첫 시나리오 기준)
    rep_sum = plans_summary["scenarios"][0]["summary"] if plans_summary["scenarios"] else {}

    # ----- Metrics / Forecast
    metrics_sum = summarize_metrics(metrics_csv) if metrics_csv else {}
    forecast_sum = summarize_forecast_by_product(forecast_csv) if forecast_csv else {}

    product_summary = summarize_by_product(plans[0]) if plans else {}

    facts = {
        "plan_scenarios": plans_summary,     # 다중 시나리오 KPI + Pareto
        "plan_summary_rep": rep_sum,         # 대표(첫 번째) 요약 (단일 입력과 호환)
        "metrics_summary": metrics_sum,
        "forecast_summary": forecast_sum,
        "product_summary": product_summary,
    }

    # ----- 샘플 미리보기
    samples = []
    if plans:
        # 시나리오별 샘플 헤드
        for nm, p in zip(scenario_names or [f"scenario_{i+1}" for i in range(len(plans))], plans):
            samples.append(f"[PRODUCTION_PLAN: {nm}]\n{_read_clip_csv(p, max_rows=max_head_rows, max_chars=max_chars)}")
    if forecast_csv:
        samples.append(f"[FORECAST_BY_PRODUCT]\n{_read_clip_csv(forecast_csv, max_rows=max_head_rows, max_chars=max_chars)}")
    if metrics_csv:
        samples.append(f"[FORECAST_METRICS]\n{_read_clip_csv(metrics_csv, max_rows=max_head_rows, max_chars=max_chars)}")
    if product_summary and not product_summary.get("missing") and not product_summary.get("schema_error"):
        try:
            df_preview = pd.DataFrame(product_summary.get("table_head", []))
            if not df_preview.empty:
                txt = df_preview.to_csv(index=False)
                if len(txt) > max_chars:
                    txt = txt[:max_chars] + f"\n...[truncated to {max_chars} chars]"
                samples.append(f"[PRODUCT_SUMMARY_BY_ITEM]\n{txt}")
        except Exception:
            pass

    sys = SystemMessage(content=SYS_PROMPT)
    user = HumanMessage(content=USER_TASK.format(
        facts_json=json.dumps(facts, ensure_ascii=False, indent=2),
        samples="\n\n".join(samples) if samples else "[NO SAMPLES]"
    ))

    raw = _call_llm([sys, user], cfg)
    js, md = _split_json_markdown(raw)

    # ----- Verifier Agent
    verification = {"ok": True, "report": "OK"}
    regen = False
    if js is not None:
        verification = verify_report(js, facts, cfg)
        if auto_regen_on_fail and not verification["ok"]:
            # 재생성: 문제 리스트를 추가 요구로 전달
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
            # 2차 결과로 교체
            raw, js, md = raw2, js2, md2
            regen = True
            # 최종 1회 재검증(보고만)
            verification = verify_report(js if js else {}, facts, cfg)

    return {"json": js, "markdown": md, "raw": raw, "verify": verification, "regen": regen}

from pathlib import Path

def _ensure_parent_dir(path: str):
    p = Path(path)
    if p.parent:  # 빈 문자열 대비
        p.parent.mkdir(parents=True, exist_ok=True)

# =========================================================
# 6) CLI
# =========================================================
def main():
    p = argparse.ArgumentParser(description="Weekly report generator (LLM-augmented, with Verifier & Scenarios)")
    # 단일 입력(하위호환)
    p.add_argument("--plan", help="production_plan.csv 경로 (단일)")
    # 복수 시나리오 입력
    p.add_argument("--plans", help="쉼표(,)로 구분된 production_plan.csv 경로 목록")
    p.add_argument("--scenario_names", help="쉼표(,)로 구분된 시나리오 이름 목록 (plans와 동일 길이)")
    p.add_argument("--forecast", help="forecast_by_product.csv 경로")
    p.add_argument("--metrics", help="forecast_metrics.csv 경로")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--out_md", default="weekly_report.md")
    p.add_argument("--out_json", default="weekly_report.json")
    p.add_argument("--out_verify", default="weekly_report.verify.txt")
    p.add_argument("--no_regen", action="store_true", help="검증 실패 시 재생성 비활성화")
    args = p.parse_args()

    plans = []
    names = None
    if args.plans:
        plans = [s.strip() for s in args.plans.split(",") if s.strip()]
    if args.scenario_names:
        names = [s.strip() for s in args.scenario_names.split(",") if s.strip()]

    out = build_report_with_llm(
        plan_csv=args.plan or "",
        plan_csvs=plans or None,
        scenario_names=names,
        forecast_csv=args.forecast or "",
        metrics_csv=args.metrics or "",
        model_name=args.model,
        auto_regen_on_fail=not args.no_regen
    )

    # 저장
    if out.get("markdown"):
        _ensure_parent_dir(args.out_md)
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write(out["markdown"])

    if out.get("json") is not None:
        _ensure_parent_dir(args.out_json)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out["json"], f, ensure_ascii=False, indent=2)

    if out.get("verify"):
        _ensure_parent_dir(args.out_verify)
        with open(args.out_verify, "w", encoding="utf-8") as f:
            v = out["verify"]
            f.write(("OK" if v.get("ok") else "NG") + "\n\n")
            f.write(v.get("report", ""))

    print(f"[OK] Saved:\n- {args.out_md}\n- {args.out_json}\n- {args.out_verify}\n(re-generated: {out.get('regen')})")

if __name__ == "__main__":
    main()
