#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluator.py

- 통합 감사(Audit): 규칙 기반 검증 + (옵션) LLM 기반 자기평가/상호검증
- 정책 학습(Policy): 성과 기반 파라미터 자동 업데이트, 저장/로딩
- MC 시나리오 검증(ShortageRate 기반, planner_opt.py와 1:1 정합)

[이번 수정 핵심]
1) ✅ Product 컬럼/이름 정합성 강화:
   - load_mc_scenarios / simulate_fixed_plan / mc_validate_plan 모두 product_col 인자 지원
   - main.py에서 prod_col 바꿔도 evaluator가 깨지지 않게 방어

2) ✅ MC 시뮬레이션 shortage 정의를 planner_opt와 "완전히 동일하게" 맞춤:
   - simulate_fixed_plan에서 shortage = backlog 증가분(신규 미충족)으로 계산/저장
   - 기존 코드의 shortage = demand - fulfilled (일별 미충족량) 방식은
     backlog 증가분과 값이 같아 보이지만, 경계/재계산/집계에서 오차가 날 수 있어
     아예 backlog 기반으로 동일하게 구현

3) ✅ CVaR mean plan에서 schema/규칙 검증 완화는 유지하되,
   - shortage 컬럼이 있으면 CVaR mean plan으로 간주 (기존 유지)
   - INV_BACKLOG_BOTH_POS는 CVaR mean plan이면 WARN 처리

4) ✅ (forecast.py 변경 반영) evaluator도 MC 입력을 유연하게:
   - 기존: --mc_scenarios_csv (long CSV)만 지원
   - 추가: --mc_npz 지원 (forecast.py가 저장한 npz; mc_mode=product 권장)
   - 추가: --mc_csv alias 지원
   - csv가 있으면 csv 우선, 없으면 npz 사용
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple, Any
import json
import os
import argparse
import numpy as np
import pandas as pd


# =========================================================
# Utils
# =========================================================

def _normalize_product(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    s = s.replace("\ufeff", "").replace("\u200b", "").replace("\xa0", " ")
    s = s.strip()
    # numeric-like "123.0" -> "123"
    s = pd.Series([s]).astype(str).str.replace(r"\.0$", "", regex=True).iloc[0]
    return s


def _quantile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, q))


def _cvar(x: np.ndarray, alpha: float) -> float:
    """CVaR_alpha = mean of tail beyond VaR_alpha (upper tail, larger is worse)."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    a = float(alpha)
    a = min(max(a, 0.0), 1.0)
    var = np.quantile(x, a)
    tail = x[x >= var]
    if tail.size == 0:
        return float(var)
    return float(np.mean(tail))


def _ensure_cols(df: pd.DataFrame, cols: List[str], fill: float = 0.0) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = fill
    return df


# =========================================================
# 규칙 기반 Verifier
# =========================================================

def _is_cvar_mean_plan(plan_df: pd.DataFrame) -> bool:
    """
    CVaR mean plan은 시나리오 평균 때문에
    (end_inventory > 0) and (backlog > 0)가 같은 row에서 발생할 수 있음.
    shortage 컬럼이 있으면 CVaR plan(또는 MC 요약 plan)로 간주.
    """
    return ("shortage" in plan_df.columns)


def verify_plan(plan_df: pd.DataFrame, daily_capacity: float, product_col: str = "Product_Number") -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []

    # 1) CAPA 위반 (ERROR)
    if "day_idx" in plan_df.columns and "produce" in plan_df.columns:
        by_day = plan_df.groupby("day_idx")["produce"].sum()
        viol = by_day[by_day > daily_capacity + 1e-9]
        if not viol.empty:
            issues.append({
                "type": "CAPA_EXCEEDED",
                "severity": "ERROR",
                "days": list(map(int, viol.index.tolist())),
                "values": [float(v) for v in viol.round(2).tolist()],
            })
    else:
        issues.append({
            "type": "SCHEMA_MISSING",
            "severity": "ERROR",
            "reason": "plan_df must contain day_idx and produce",
        })

    # 2) 음수값 (ERROR)
    for c in ["demand", "produce", "backlog", "end_inventory", "shortage"]:
        if c in plan_df.columns:
            neg = plan_df[plan_df[c] < -1e-9]
            if len(neg):
                issues.append({
                    "type": "NEGATIVE_VALUES",
                    "severity": "ERROR",
                    "column": c,
                    "count": int(len(neg)),
                })

    # 3) 재고/백로그 동시 양수 (CVaR 평균 plan에서는 WARN)
    if {"backlog", "end_inventory"} <= set(plan_df.columns):
        both_pos = ((plan_df["backlog"] > 0) & (plan_df["end_inventory"] > 0)).mean()
        if both_pos > 0.01:
            is_cvar = _is_cvar_mean_plan(plan_df)
            issues.append({
                "type": "INV_BACKLOG_BOTH_POS",
                "severity": "WARN" if is_cvar else "ERROR",
                "rate": float(both_pos),
                "note": "CVaR mean plan에서는 시나리오 평균 때문에 동시양수가 자연스럽게 발생할 수 있음."
                        if is_cvar else "단일 시나리오/결정론 plan에서 동시양수는 데이터/로직 점검 필요."
            })

    has_error = any(i.get("severity") == "ERROR" for i in issues)
    return {"ok": (not has_error), "issues": issues}


def suggest_fixes(verify_result: Dict[str, Any]) -> List[str]:
    fixes: List[str] = []
    for i in verify_result.get("issues", []):
        t = i.get("type")
        if t == "CAPA_EXCEEDED":
            fixes.append("일일 CAPA 위반: daily_capacity를 늘리거나, min_lot/weight/smooth 파라미터를 조정해 생산량이 분산되도록 하세요.")
        elif t == "NEGATIVE_VALUES":
            fixes.append(f"음수값 존재: 컬럼({i.get('column')}) 전처리/clip 및 scale 설정을 점검하세요.")
        elif t == "INV_BACKLOG_BOTH_POS":
            fixes.append("재고/백로그 동시양수: CVaR mean plan이면 정상일 수 있으나, 단일 plan이면 inventory/backlog 업데이트 식 점검이 필요합니다.")
        elif t == "SCHEMA_MISSING":
            fixes.append("스키마 누락: plan_df에 day_idx, produce 컬럼이 있는지 확인하세요.")
    return fixes


# =========================================================
# (옵션) LLM 기반 critique / crosscheck (자리만 유지)
# =========================================================

def llm_critique(plan_df: pd.DataFrame, metrics_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"enabled": False, "note": "LLM critique disabled in this environment."}


def llm_crosscheck(plan_df: pd.DataFrame, metrics_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"enabled": False, "note": "LLM crosscheck disabled in this environment."}


# =========================================================
# Policy 저장/로딩 + 업데이트 (간단 유지)
# =========================================================

def load_policy(path: str) -> Dict[str, Any]:
    if not path or (not os.path.exists(path)):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_policy(path: str, policy: Dict[str, Any]) -> None:
    if not path:
        return
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(policy, f, ensure_ascii=False, indent=2)


def _estimate_utilization_from_plan(plan_df: pd.DataFrame, daily_capacity: float) -> Dict[str, float]:
    if "day_idx" not in plan_df.columns or "produce" not in plan_df.columns:
        return {"Utilization_mean": float("nan"), "Utilization_total": float("nan")}
    by_day = plan_df.groupby("day_idx")["produce"].sum().sort_index()
    n_days = max(int(by_day.shape[0]), 1)
    util_mean = float(by_day.mean() / (daily_capacity + 1e-9))
    util_total = float(by_day.sum() / ((daily_capacity + 1e-9) * n_days))
    return {"Utilization_mean": util_mean, "Utilization_total": util_total}


def update_policy_from_outcomes(policy: Dict[str, Any], outcomes: Dict[str, Any]) -> Dict[str, Any]:
    policy = dict(policy or {})
    hist = policy.get("history", [])
    hist.append(outcomes)
    policy["history"] = hist[-50:]  # keep last 50
    return policy


# =========================================================
# MC Scenario loader
# =========================================================

def load_mc_scenarios(path: str, product_col: str = "Product_Number") -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"scenario_id", "day_idx", product_col, "demand"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"MC scenarios missing cols: {missing}")

    df["scenario_id"] = pd.to_numeric(df["scenario_id"], errors="coerce").fillna(0).astype(int)
    df["day_idx"] = pd.to_numeric(df["day_idx"], errors="coerce").fillna(0).astype(int)
    df["demand"] = pd.to_numeric(df["demand"], errors="coerce").fillna(0.0)
    df[product_col] = df[product_col].map(_normalize_product)
    return df


def load_mc_scenarios_npz(path: str, product_col: str = "Product_Number") -> pd.DataFrame:
    """Load MC scenarios from .npz produced by forecast.py (mc_mode=product recommended).
    Expected keys (flexible):
      - scenarios: (S,P,D)
      - products:  (P,)
    Returns long-form DataFrame: scenario_id, day_idx, product_col, demand
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    z = np.load(path, allow_pickle=True)

    scenarios = None
    for key in ["scenarios", "scenario", "X", "arr_0"]:
        if key in z:
            scenarios = z[key]
            break
    products = None
    for key in ["products", "product_list", "prod", "arr_1"]:
        if key in z:
            products = z[key]
            break

    if scenarios is None:
        raise KeyError(f"mc_npz missing scenarios. keys={list(z.keys())}")
    scenarios = np.asarray(scenarios, dtype=float)

    if products is None:
        # Some npz may not store products (e.g., mc_mode=raw). Not supported here.
        raise KeyError(
            "mc_npz missing products list. Please generate MC with mc_mode=product (so products are saved) "
            f"or provide --mc_scenarios_csv. keys={list(z.keys())}"
        )

    products = pd.Series(products).astype(str).str.replace(r"\.0$", "", regex=True).map(_normalize_product).tolist()

    if scenarios.ndim != 3:
        raise ValueError(f"mc_npz scenarios must be 3D (S,P,D). got shape={scenarios.shape}")
    S, P, D = scenarios.shape
    if len(products) != P:
        raise ValueError(f"mc_npz products length mismatch: len(products)={len(products)} vs P={P}")

    # Vectorized long conversion
    scenario_id = np.repeat(np.arange(S, dtype=int), P * D)
    day_idx = np.tile(np.repeat(np.arange(D, dtype=int), P), S)
    prod_arr = np.tile(np.tile(np.array(products, dtype=object), D), S)
    demand = scenarios.reshape(-1)

    out = pd.DataFrame({
        "scenario_id": scenario_id,
        "day_idx": day_idx,
        product_col: prod_arr,
        "demand": demand,
    })
    return out


def load_mc_scenarios_any(
    mc_npz: Optional[str] = None,
    mc_csv: Optional[str] = None,
    product_col: str = "Product_Number",
) -> pd.DataFrame:
    """Load MC scenarios from either npz or csv (csv preferred if both given)."""
    if mc_csv:
        return load_mc_scenarios(mc_csv, product_col=product_col)
    if mc_npz:
        return load_mc_scenarios_npz(mc_npz, product_col=product_col)
    raise ValueError("Either mc_npz or mc_csv must be provided.")


# =========================================================
# MC simulation + validation (planner_opt와 정합)
# =========================================================

def _build_product_day_grid(products: List[str], days: List[int], product_col: str = "Product_Number") -> pd.DataFrame:
    rows = []
    for p in products:
        for d in days:
            rows.append({product_col: p, "day_idx": int(d)})
    return pd.DataFrame(rows)


def simulate_fixed_plan(
    plan_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    initial_inventory: float = 0.0,
    product_col: str = "Product_Number",
) -> pd.DataFrame:
    """
    고정 생산계획(plan_df)을 특정 시나리오 수요(demand_df) 하에서 시뮬레이션.

    ✅ planner_opt/evaluator 정합: shortage = backlog 증가분 (신규 미충족)
      - backlog_t = max(-stock_t, 0)
      - shortage_t = max(backlog_t - backlog_{t-1}, 0), shortage_0 = backlog_0

    plan/demand에 일부 (product, day)가 빠져도 (products×days) 그리드로 0 채움.
    """
    required_plan = {product_col, "day_idx", "produce"}
    required_dem  = {product_col, "day_idx", "demand"}
    if not (required_plan <= set(plan_df.columns)):
        raise ValueError(f"plan_df missing cols: {required_plan - set(plan_df.columns)}")
    if not (required_dem <= set(demand_df.columns)):
        raise ValueError(f"demand_df missing cols: {required_dem - set(demand_df.columns)}")

    plan = plan_df.copy()
    plan[product_col] = plan[product_col].map(_normalize_product)
    plan["day_idx"] = pd.to_numeric(plan["day_idx"], errors="coerce").fillna(0).astype(int)
    plan["produce"] = pd.to_numeric(plan["produce"], errors="coerce").fillna(0.0)

    dem = demand_df.copy()
    dem[product_col] = dem[product_col].map(_normalize_product)
    dem["day_idx"] = pd.to_numeric(dem["day_idx"], errors="coerce").fillna(0).astype(int)
    dem["demand"] = pd.to_numeric(dem["demand"], errors="coerce").fillna(0.0)

    prod_agg = plan.groupby([product_col, "day_idx"], as_index=False)["produce"].sum()
    dem_agg  = dem.groupby([product_col, "day_idx"], as_index=False)["demand"].sum().rename(columns={"demand": "demand_s"})

    products = sorted(set(prod_agg[product_col]).union(set(dem_agg[product_col])))
    if not products:
        return pd.DataFrame(columns=[
            product_col, "day_idx", "produce", "demand", "fulfilled", "shortage", "end_inventory", "backlog"
        ])

    days = sorted(set(prod_agg["day_idx"]).union(set(dem_agg["day_idx"])))
    if not days:
        days = [0]

    grid = _build_product_day_grid(products, days, product_col=product_col)

    sim = (
        grid.merge(prod_agg, on=[product_col, "day_idx"], how="left")
            .merge(dem_agg,  on=[product_col, "day_idx"], how="left")
    )
    sim["produce"] = sim["produce"].fillna(0.0)
    sim["demand_s"] = sim["demand_s"].fillna(0.0)

    # simulate per product
    sim = sim.sort_values([product_col, "day_idx"]).reset_index(drop=True)

    out_rows = []
    for p, g in sim.groupby(product_col, sort=False):
        stock = float(initial_inventory)
        prev_backlog = 0.0
        for r in g.itertuples(index=False):
            d = int(r.day_idx)
            prod = float(r.produce)
            demd = float(r.demand_s)

            stock = stock + prod - demd
            inv = max(stock, 0.0)
            backlog = max(-stock, 0.0)

            # ✅ shortage = backlog increase
            if d == g["day_idx"].min():
                shortage = backlog
            else:
                shortage = max(backlog - prev_backlog, 0.0)

            prev_backlog = backlog

            fulfilled = demd - shortage  # 참고용(정합 상 shortage 우선)
            if fulfilled < 0:
                fulfilled = 0.0

            out_rows.append({
                product_col: p,
                "day_idx": d,
                "produce": prod,
                "demand": demd,
                "fulfilled": fulfilled,
                "shortage": shortage,
                "end_inventory": inv,
                "backlog": backlog,
            })

    return pd.DataFrame(out_rows)


def summarize_mc_results(
    per_scenario: pd.DataFrame,
    alpha: float = 0.9,
    loss_col: str = "loss",
) -> Dict[str, Any]:
    loss = pd.to_numeric(per_scenario[loss_col], errors="coerce").dropna().to_numpy(dtype=float)
    sr = pd.to_numeric(per_scenario["ShortageRate"], errors="coerce").dropna().to_numpy(dtype=float)

    return {
        "S": int(len(per_scenario)),
        "loss_mean": float(np.mean(loss)) if loss.size else float("nan"),
        "loss_var": _quantile(loss, alpha),
        "loss_cvar": _cvar(loss, alpha),
        "shortage_rate_mean": float(np.mean(sr)) if sr.size else float("nan"),
        "shortage_rate_var": _quantile(sr, alpha),
        "shortage_rate_cvar": _cvar(sr, alpha),
    }


def mc_validate_plan(
    plan_df: pd.DataFrame,
    scenarios_df: pd.DataFrame,
    initial_inventory: float,
    alpha: float,
    daily_capacity: float,
    w_b: float = 1.0,
    w_i: float = 0.2,
    fail_threshold: float = 0.03,
    product_col: str = "Product_Number",
) -> Dict[str, Any]:
    """
    scenarios_df: long-form (scenario_id, day_idx, product_col, demand)
    return:
      - per_scenario metrics + summary
    """
    req = {"scenario_id", "day_idx", product_col, "demand"}
    missing = req - set(scenarios_df.columns)
    if missing:
        raise ValueError(f"scenarios_df missing cols: {missing}")

    scenarios_df = scenarios_df.copy()
    scenarios_df[product_col] = scenarios_df[product_col].map(_normalize_product)
    scenarios_df["scenario_id"] = pd.to_numeric(scenarios_df["scenario_id"], errors="coerce").fillna(0).astype(int)
    scenarios_df["day_idx"] = pd.to_numeric(scenarios_df["day_idx"], errors="coerce").fillna(0).astype(int)
    scenarios_df["demand"] = pd.to_numeric(scenarios_df["demand"], errors="coerce").fillna(0.0)

    # plan preprocess
    plan = plan_df.copy()
    plan = _ensure_cols(plan, [product_col, "day_idx", "produce"], fill=0.0)
    plan[product_col] = plan[product_col].map(_normalize_product)
    plan["day_idx"] = pd.to_numeric(plan["day_idx"], errors="coerce").fillna(0).astype(int)
    plan["produce"] = pd.to_numeric(plan["produce"], errors="coerce").fillna(0.0)

    # CAPA check
    verify = verify_plan(plan, daily_capacity=daily_capacity, product_col=product_col)

    per_rows = []
    for sid, dem_s in scenarios_df.groupby("scenario_id", sort=True):
        sim = simulate_fixed_plan(
            plan_df=plan,
            demand_df=dem_s[[product_col, "day_idx", "demand"]],
            initial_inventory=float(initial_inventory),
            product_col=product_col,
        )

        total_demand = float(sim["demand"].sum())
        total_short = float(sim["shortage"].sum())
        total_inv = float(sim["end_inventory"].sum())

        shortage_rate = float(total_short / (total_demand + 1e-9))
        inv_rate = float(total_inv / ((daily_capacity + 1e-9) * max(sim["day_idx"].nunique(), 1)))

        loss = float(w_b * shortage_rate + w_i * inv_rate)
        per_rows.append({
            "scenario_id": int(sid),
            "TotalDemand": total_demand,
            "TotalShortage": total_short,
            "ShortageRate": shortage_rate,
            "InventoryRate": inv_rate,
            "loss": loss,
            "fail": bool(shortage_rate > float(fail_threshold)),
        })

    per_df = pd.DataFrame(per_rows).sort_values("scenario_id").reset_index(drop=True)
    summary = summarize_mc_results(per_df, alpha=float(alpha), loss_col="loss")

    return {
        "verify": verify,
        "per_scenario": per_df.to_dict(orient="records"),
        "summary": summary,
        "config": {
            "alpha": float(alpha),
            "w_b": float(w_b),
            "w_i": float(w_i),
            "fail_threshold": float(fail_threshold),
            "initial_inventory": float(initial_inventory),
        }
    }


# =========================================================
# Audit + Learn
# =========================================================

def audit_and_learn(
    plan_df: pd.DataFrame,
    daily_capacity: float,
    metrics_summary: Optional[Dict[str, Any]] = None,
    policy_path: Optional[str] = None,
    llm_enabled: bool = False,
    product_col: str = "Product_Number",
) -> Dict[str, Any]:

    verify = verify_plan(plan_df, daily_capacity=daily_capacity, product_col=product_col)
    fixes = suggest_fixes(verify)

    policy = {}
    if policy_path:
        policy = load_policy(policy_path)

    util = _estimate_utilization_from_plan(plan_df, daily_capacity=daily_capacity)

    result: Dict[str, Any] = {
        "verify": verify,
        "fixes": fixes,
        "utilization": util,
        "metrics_summary": metrics_summary,
    }

    if llm_enabled:
        result["llm_critique"] = llm_critique(plan_df, metrics_summary)
        result["llm_crosscheck"] = llm_crosscheck(plan_df, metrics_summary)

    # policy update
    outcomes = {
        "verify_ok": bool(verify.get("ok", False)),
        "utilization": util,
        "metrics_summary": metrics_summary,
    }
    policy = update_policy_from_outcomes(policy, outcomes)

    if policy_path:
        save_policy(policy_path, policy)
        result["policy_saved"] = policy_path

    return result


# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Planner Evaluator (verify → fixes → policy update)")
    ap.add_argument("--plan_csv", required=True, help="production_plan.csv 경로")
    ap.add_argument("--daily_capacity", type=float, required=True, help="일일 CAPA")
    ap.add_argument("--product_col", type=str, default="Product_Number", help="제품 컬럼명 (default: Product_Number)")
    ap.add_argument("--metrics_json", type=str, default=None, help="planning_metrics.json 경로(옵션)")
    ap.add_argument("--policy_path", type=str, default=None, help="정책 저장/로드 파일 경로(옵션)")
    ap.add_argument("--llm_enabled", action="store_true", help="LLM 기반 critique/crosscheck 사용")
    ap.add_argument("--out_json", type=str, default=None, help="감사 결과 저장 경로(기본: plan_csv 옆 governance_audit.json)")

    # MC options
    ap.add_argument("--mc_scenarios_csv", type=str, default=None,
                    help="MC 시나리오 수요 CSV (scenario_id, day_idx, Product_Number, demand)")
    ap.add_argument("--mc_npz", type=str, default=None,
                    help="MC 시나리오 npz (forecast.py에서 생성된 npz; mc_mode=product 권장)")
    ap.add_argument("--mc_csv", type=str, default=None,
                    help="(alias) MC 시나리오 수요 CSV. --mc_scenarios_csv와 동일")
    ap.add_argument("--initial_inventory", type=float, default=0.0, help="MC 시뮬레이션 초기 재고")
    ap.add_argument("--mc_out_json", type=str, default=None, help="MC 검증 결과 JSON 저장 경로")
    ap.add_argument("--mc_alpha", type=float, default=0.9, help="MC 요약 분위수 (0.9=VaR)")
    ap.add_argument("--mc_wb", type=float, default=1.0, help="MC Loss 가중치: ShortageRate")
    ap.add_argument("--mc_wi", type=float, default=0.2, help="MC Loss 가중치: InventoryRate")
    ap.add_argument("--mc_fail_threshold", type=float, default=0.03, help="Fail 기준: ShortageRate > threshold")

    args = ap.parse_args()

    # plan 로드 + 숫자 보정
    plan_df = pd.read_csv(args.plan_csv)
    plan_df = _ensure_cols(plan_df, [args.product_col, "day_idx", "produce"], fill=0.0)

    plan_df[args.product_col] = plan_df[args.product_col].map(_normalize_product)
    plan_df["day_idx"] = pd.to_numeric(plan_df["day_idx"], errors="coerce").fillna(0).astype(int)
    plan_df["produce"] = pd.to_numeric(plan_df["produce"], errors="coerce").fillna(0.0)

    for c in ["demand", "backlog", "end_inventory", "shortage"]:
        if c in plan_df.columns:
            plan_df[c] = pd.to_numeric(plan_df[c], errors="coerce").fillna(0.0)

    # MC 검증(옵션)
    mc_validation = None
    mc_csv_path = args.mc_scenarios_csv or args.mc_csv
    if mc_csv_path or args.mc_npz:
        scenarios_df = load_mc_scenarios_any(mc_npz=args.mc_npz, mc_csv=mc_csv_path, product_col=args.product_col)
        mc_validation = mc_validate_plan(
            plan_df=plan_df,
            scenarios_df=scenarios_df,
            initial_inventory=float(args.initial_inventory),
            alpha=float(args.mc_alpha),
            daily_capacity=float(args.daily_capacity),
            w_b=float(args.mc_wb),
            w_i=float(args.mc_wi),
            fail_threshold=float(args.mc_fail_threshold),
            product_col=args.product_col,
        )

        mc_out_path = args.mc_out_json
        if not mc_out_path:
            base_dir = os.path.dirname(args.out_json) if args.out_json else os.path.dirname(args.plan_csv)
            mc_out_path = os.path.join(base_dir, "mc_validation.json")

        d = os.path.dirname(mc_out_path)
        if d:
            os.makedirs(d, exist_ok=True)

        with open(mc_out_path, "w", encoding="utf-8") as f:
            json.dump(mc_validation, f, ensure_ascii=False, indent=2)
        print(f"[OK] Saved MC validation → {mc_out_path}")

    # metrics_json 로드(옵션)
    metrics_summary = None
    if args.metrics_json and os.path.exists(args.metrics_json):
        with open(args.metrics_json, "r", encoding="utf-8") as f:
            metrics_summary = json.load(f)

    # 감사 & 정책 업데이트
    result = audit_and_learn(
        plan_df=plan_df,
        daily_capacity=float(args.daily_capacity),
        metrics_summary=metrics_summary,
        policy_path=args.policy_path,
        llm_enabled=args.llm_enabled,
        product_col=args.product_col,
    )

    if mc_validation is not None:
        result["mc_validation"] = mc_validation.get("summary", mc_validation)

    out_path = args.out_json or os.path.join(os.path.dirname(args.plan_csv), "governance_audit.json")
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved evaluator output → {out_path}")

    if not result.get("verify", {}).get("ok", True):
        print("[WARN] Verification issues detected. See 'issues' in the JSON.")