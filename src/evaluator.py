#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluator.py

- 통합 감사(Audit): 규칙 기반 검증 + (옵션) LLM 기반 자기평가/상호검증
- 정책 학습(Policy): 성과 기반 파라미터 자동 업데이트, 저장/로딩
- MC 시나리오 검증(ShortageRate 기반, planner_opt.py와 1:1 정합):
    * 입력: mc_scenarios_csv (scenario_id, day_idx, Product_Number, demand)
    * 고정 생산계획(plan_df)을 시나리오 수요에 대해 시뮬레이션하여
      "미충족(Shortage)" 기반 리스크 요약 저장

[정합성 규칙: planner_opt와 동일]
- TotalShortage = sum(shortage)  (누적 미충족)
- ShortageRate  = TotalShortage / TotalDemand
- AvgInventory  = sum(end_inventory)/(P*D)   (product-day 평균 재고)
- InventoryRate = AvgInventory / daily_capacity
- Loss          = wb * ShortageRate + wi * InventoryRate

[요약 지표]
- Loss: mean / VaR(alpha) / CVaR(alpha) / worst
- ShortageRate/InventoryRate 등: mean / VaR(alpha) / worst  (CVaR는 Loss만 제공)

[FailRate]
- ShortageRate > fail_threshold 인 시나리오 비율

[강건성 개선]
- product/day 커버리지 불일치 자동 보정(outer join + 0 fill)
- (products × days) 그리드를 만들어 simulate 시 누락 일자 방지
- Utilization이 metrics_json에 없으면 plan에서 직접 추정 가능

[중요 수정]
- CVaR plan 출력은 (시나리오 평균) end_inventory/backlog가 동시에 양수로 보일 수 있음
  (시나리오 A는 inv>0, 시나리오 B는 backlog>0 → 평균은 둘 다 >0).
  따라서 INV_BACKLOG_BOTH_POS는 CVaR plan에서는 ERROR가 아니라 WARN로 처리.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Any as AnyType

import argparse
import os
import json

import numpy as np
import pandas as pd

_EPS = 1e-9


# =========================================================
# Utils
# =========================================================
def _normalize_product(x: AnyType) -> str:
    return str(x).replace(".0", "")

def _quantile(arr: np.ndarray, q: float) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.quantile(arr, q))

def _cvar(arr: np.ndarray, alpha: float) -> float:
    """CVaR_alpha = E[X | X >= VaR_alpha]"""
    if arr.size == 0:
        return 0.0
    thr = np.quantile(arr, alpha)
    tail = arr[arr >= thr]
    return float(tail.mean()) if tail.size else float(thr)

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
    CVaR 모드 출력(plan_df)의 전형적 특징:
    - shortage 컬럼이 존재 (planner_opt CVaR 출력이 shortage 평균을 넣음)
    - 또는 backlog/end_inventory가 '평균값'일 가능성이 높아 둘 다 양수 문제 발생 가능
    """
    if "shortage" in plan_df.columns:
        return True
    # 매우 보수적으로: backlog/end_inventory 둘 다 있고, 값이 연속형(평균치)처럼 보이면 True로 볼 수도 있지만
    # 여기서는 shortage 존재만으로 충분히 처리 (안전하게).
    return False

def verify_plan(plan_df: pd.DataFrame, daily_capacity: float) -> Dict[str, Any]:
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

    # ok 판정: ERROR만 있으면 false, WARN만 있으면 true
    has_error = any(i.get("severity") == "ERROR" for i in issues)
    return {"ok": (not has_error), "issues": issues}


def suggest_fixes(plan_df: pd.DataFrame, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
    suggestions: List[Dict[str, Any]] = []
    for issue in issues:
        t = issue.get("type")
        sev = issue.get("severity", "ERROR")
        if t == "CAPA_EXCEEDED":
            suggestions.append({
                "target": "lambda_smooth",
                "action": "increase",
                "reason": "일별 생산 변동성 완화로 CAPA 피크를 줄입니다.",
                "severity": sev,
            })
        elif t == "INV_BACKLOG_BOTH_POS":
            # CVaR mean plan이면 경고성 안내만
            suggestions.append({
                "target": "verifier_rule",
                "action": "treat_as_warn_for_cvar_mean_plan",
                "reason": "CVaR 평균 출력에서는 시나리오 평균으로 인해 inv/backlog 동시 양수가 자연스러울 수 있습니다.",
                "severity": sev,
            })
        elif t == "NEGATIVE_VALUES":
            suggestions.append({
                "target": "data_cleaning",
                "action": "sanity_check",
                "reason": f"{issue.get('column')} 컬럼 음수값 발견. 입력/전처리 확인 필요.",
                "severity": sev,
            })
        elif t == "SCHEMA_MISSING":
            suggestions.append({
                "target": "schema",
                "action": "fix",
                "reason": issue.get("reason", "입력 스키마 확인 필요"),
                "severity": sev,
            })
    return {"suggestions": suggestions}


# =========================================================
# LLM 기반 Evaluator (옵션)
# =========================================================
def llm_critique(metrics_summary: Dict[str, Any], plan_df_sample: pd.DataFrame, enabled: bool = False) -> str:
    if not enabled:
        return ""
    return "(LLM critique placeholder)"

def llm_crosscheck(planner_note: str, reporter_note: str, enabled: bool = False) -> str:
    if not enabled:
        return ""
    return "(LLM cross-check placeholder)"


# =========================================================
# Policy (경험 학습/저장)
# =========================================================
_DEFAULT_POLICY = {
    "lambda_smooth": 1.0,
    "WEIGHT_MAP": {"0": 5.0, "1": 2.0, "2": 0.5, "3": 1.0},
    "daily_capacity": None,
}

def load_policy(path: str) -> Dict[str, Any]:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(json.dumps(_DEFAULT_POLICY))

def save_policy(path: str, policy: Dict[str, Any]) -> None:
    if not path:
        return
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(policy, f, ensure_ascii=False, indent=2)

def _estimate_utilization_from_plan(plan_df: pd.DataFrame, daily_capacity: float) -> float:
    if daily_capacity <= 0:
        return 1.0
    if "day_idx" not in plan_df.columns or "produce" not in plan_df.columns:
        return 1.0
    by_day = plan_df.groupby("day_idx")["produce"].sum()
    if len(by_day) == 0:
        return 1.0
    return float((by_day / float(daily_capacity)).clip(0, 1.5).mean())

def update_policy_from_outcomes(policy: Dict[str, Any], metrics: Dict[str, Any], plan_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    pm = metrics.get("planning_metrics", {}) or {}
    shortage = float(pm.get("ShortageRate", pm.get("BacklogRate", 0.0)))

    util = pm.get("Utilization", None)
    if util is None and plan_df is not None:
        cap = float(policy.get("daily_capacity") or 0.0)
        if cap <= 0:
            cap = float(metrics.get("daily_capacity", 0.0) or 0.0)
        if cap > 0:
            util = _estimate_utilization_from_plan(plan_df, cap)
        else:
            util = 1.0
    util = float(util) if util is not None else 1.0

    if shortage > 0.03:
        policy["lambda_smooth"] = max(0.2, float(policy.get("lambda_smooth", 1.0)) - 0.2)
        wm = policy.get("WEIGHT_MAP", {}) or {}
        wm["0"] = float(wm.get("0", 5.0)) + 0.5
        policy["WEIGHT_MAP"] = wm

    if util < 0.7:
        policy["lambda_smooth"] = min(3.0, float(policy.get("lambda_smooth", 1.0)) + 0.2)

    return policy


# =========================================================
# MC 시나리오 로드/시뮬레이션/검증
# =========================================================
def load_mc_scenarios(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"scenario_id", "day_idx", "Product_Number", "demand"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"MC scenarios missing cols: {missing}")

    df["scenario_id"] = pd.to_numeric(df["scenario_id"], errors="coerce").fillna(0).astype(int)
    df["day_idx"] = pd.to_numeric(df["day_idx"], errors="coerce").fillna(0).astype(int)
    df["demand"] = pd.to_numeric(df["demand"], errors="coerce").fillna(0.0)
    df["Product_Number"] = df["Product_Number"].map(_normalize_product)
    return df

def _build_product_day_grid(products: List[str], days: List[int]) -> pd.DataFrame:
    return pd.MultiIndex.from_product(
        [products, days],
        names=["Product_Number", "day_idx"],
    ).to_frame(index=False)

def simulate_fixed_plan(
    plan_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    initial_inventory: float = 0.0
) -> pd.DataFrame:
    """
    고정 생산계획(plan_df)을 특정 시나리오 수요(demand_df) 하에서 시뮬레이션.

    - stock = inv - backlog (net stock)
    - available = stock + produce
    - fulfilled = min(max(available,0), demand)
    - shortage = demand - fulfilled
    - stock = available - demand (carry)

    ✅ plan/demand에 일부 (product, day)가 빠져도 (products×days) 그리드로 0 채움.
    """
    required_plan = {"Product_Number", "day_idx", "produce"}
    required_dem  = {"Product_Number", "day_idx", "demand"}
    if not (required_plan <= set(plan_df.columns)):
        raise ValueError(f"plan_df missing cols: {required_plan - set(plan_df.columns)}")
    if not (required_dem <= set(demand_df.columns)):
        raise ValueError(f"demand_df missing cols: {required_dem - set(demand_df.columns)}")

    plan = plan_df.copy()
    plan["Product_Number"] = plan["Product_Number"].map(_normalize_product)
    plan["day_idx"] = pd.to_numeric(plan["day_idx"], errors="coerce").fillna(0).astype(int)
    plan["produce"] = pd.to_numeric(plan["produce"], errors="coerce").fillna(0.0)

    dem = demand_df.copy()
    dem["Product_Number"] = dem["Product_Number"].map(_normalize_product)
    dem["day_idx"] = pd.to_numeric(dem["day_idx"], errors="coerce").fillna(0).astype(int)
    dem["demand"] = pd.to_numeric(dem["demand"], errors="coerce").fillna(0.0)

    prod_agg = plan.groupby(["Product_Number", "day_idx"], as_index=False)["produce"].sum()
    dem_agg  = dem.groupby(["Product_Number", "day_idx"], as_index=False)["demand"].sum().rename(columns={"demand": "demand_s"})

    products = sorted(set(prod_agg["Product_Number"]).union(set(dem_agg["Product_Number"])))
    if not products:
        return pd.DataFrame(columns=[
            "Product_Number","day_idx","produce","demand","fulfilled","shortage","end_inventory","backlog"
        ])

    days = sorted(set(prod_agg["day_idx"]).union(set(dem_agg["day_idx"])))
    if not days:
        days = [0]

    grid = _build_product_day_grid(products, days)

    sim = (
        grid.merge(prod_agg, on=["Product_Number", "day_idx"], how="left")
            .merge(dem_agg,  on=["Product_Number", "day_idx"], how="left")
    )
    sim["produce"] = sim["produce"].fillna(0.0)
    sim["demand_s"] = sim["demand_s"].fillna(0.0)
    sim = sim.sort_values(["Product_Number","day_idx"]).reset_index(drop=True)

    out_rows: List[Dict[str, Any]] = []
    for p, g in sim.groupby("Product_Number", sort=False):
        stock = float(initial_inventory)
        for _, r in g.iterrows():
            produce = float(r["produce"])
            demand  = float(r["demand_s"])

            available = stock + produce
            fulfilled = min(max(available, 0.0), demand)
            shortage = max(demand - fulfilled, 0.0)

            stock = available - demand
            inv = max(stock, 0.0)
            backlog = max(-stock, 0.0)

            out_rows.append({
                "Product_Number": p,
                "day_idx": int(r["day_idx"]),
                "produce": produce,
                "demand": demand,
                "fulfilled": float(fulfilled),
                "shortage": float(shortage),
                "end_inventory": float(inv),
                "backlog": float(backlog),
            })

    return pd.DataFrame(out_rows)


def summarize_mc_results(per_scenario_metrics: pd.DataFrame, alpha: float, fail_threshold: float) -> Dict[str, Any]:
    """
    summary 스키마 고정:
    - metric: mean / VaR / worst (+ Loss는 CVaR)
    """
    out: Dict[str, Any] = {}
    if per_scenario_metrics.empty:
        return {"alpha": float(alpha), "S": 0, "FailRate": 0.0, "FailThreshold": float(fail_threshold)}

    def pack(arr: np.ndarray, with_cvar: bool = False) -> Dict[str, float]:
        d = {
            "mean": float(arr.mean()) if arr.size else 0.0,
            "VaR": _quantile(arr, alpha),
            "worst": float(arr.max()) if arr.size else 0.0,
        }
        if with_cvar:
            d["CVaR"] = _cvar(arr, alpha)
        return d

    for col in ["ShortageRate", "InventoryRate", "Loss", "TotalShortage", "AvgInventory", "TotalDemand"]:
        if col not in per_scenario_metrics.columns:
            continue
        arr = per_scenario_metrics[col].astype(float).to_numpy()
        out[col] = pack(arr, with_cvar=(col == "Loss"))

    out["FailRate"] = float((per_scenario_metrics["ShortageRate"] > float(fail_threshold)).mean())
    out["FailThreshold"] = float(fail_threshold)
    out["alpha"] = float(alpha)
    out["S"] = int(per_scenario_metrics["scenario_id"].nunique())

    # 디버깅용(평균 P/D)
    if "P" in per_scenario_metrics.columns and len(per_scenario_metrics["P"]) > 0:
        out["P_mean"] = float(per_scenario_metrics["P"].mean())
    if "D" in per_scenario_metrics.columns and len(per_scenario_metrics["D"]) > 0:
        out["D_mean"] = float(per_scenario_metrics["D"].mean())

    return out


def mc_validate_plan(
    plan_df: pd.DataFrame,
    scenarios_df: pd.DataFrame,
    initial_inventory: float,
    alpha: float,
    daily_capacity: float,
    w_b: float,
    w_i: float,
    fail_threshold: float,
) -> Dict[str, Any]:
    """
    planner_opt CVaR과 1:1 정합 (rate 기반):
    - AvgInventory = sum(end_inventory)/(P*D)
    """
    plan = plan_df.copy()
    plan["Product_Number"] = plan["Product_Number"].map(_normalize_product)
    plan["day_idx"] = pd.to_numeric(plan["day_idx"], errors="coerce").fillna(0).astype(int)
    plan["produce"] = pd.to_numeric(plan["produce"], errors="coerce").fillna(0.0)

    sc = scenarios_df.copy()
    sc["scenario_id"] = pd.to_numeric(sc["scenario_id"], errors="coerce").fillna(0).astype(int)
    sc["Product_Number"] = sc["Product_Number"].map(_normalize_product)
    sc["day_idx"] = pd.to_numeric(sc["day_idx"], errors="coerce").fillna(0).astype(int)
    sc["demand"] = pd.to_numeric(sc["demand"], errors="coerce").fillna(0.0)

    cap_norm = float(max(daily_capacity, _EPS))

    per_rows: List[Dict[str, Any]] = []
    for sid, g in sc.groupby("scenario_id", sort=True):
        sim = simulate_fixed_plan(
            plan_df=plan,
            demand_df=g[["Product_Number", "day_idx", "demand"]],
            initial_inventory=float(initial_inventory),
        )

        total_dem = float(sim["demand"].sum())
        total_short = float(sim["shortage"].sum())
        short_rate = float(total_short / (total_dem + _EPS))

        P = int(sim["Product_Number"].nunique()) if not sim.empty else 1
        D = int(sim["day_idx"].nunique()) if not sim.empty else 1

        avg_inv = float(sim["end_inventory"].sum() / float(max(P * D, 1)))
        inv_rate = float(avg_inv / cap_norm)

        loss = float(w_b * short_rate + w_i * inv_rate)

        per_rows.append({
            "scenario_id": int(sid),
            "ShortageRate": short_rate,
            "InventoryRate": inv_rate,
            "Loss": loss,
            "TotalShortage": total_short,
            "AvgInventory": avg_inv,
            "TotalDemand": total_dem,
            "P": P,
            "D": D,
        })

    per_df = pd.DataFrame(per_rows)
    summary = summarize_mc_results(per_df, alpha=float(alpha), fail_threshold=float(fail_threshold))
    return {"summary": summary, "per_scenario": per_df.to_dict(orient="records")}


# =========================================================
# 통합 오케스트레이션
# =========================================================
def audit_and_learn(
    plan_df: pd.DataFrame,
    daily_capacity: float,
    metrics_summary: Optional[Dict[str, Any]] = None,
    policy_path: Optional[str] = None,
    llm_enabled: bool = False,
    planner_note: str = "",
    reporter_note: str = "",
) -> Dict[str, Any]:
    ver = verify_plan(plan_df, daily_capacity)
    fixes = suggest_fixes(plan_df, ver.get("issues", []))

    critique_text = llm_critique(metrics_summary or {}, plan_df.head(200), enabled=llm_enabled)
    crosscheck_text = llm_crosscheck(planner_note, reporter_note, enabled=llm_enabled)

    policy = load_policy(policy_path) if policy_path else json.loads(json.dumps(_DEFAULT_POLICY))
    policy = update_policy_from_outcomes(policy, metrics_summary or {}, plan_df=plan_df)
    if policy_path:
        save_policy(policy_path, policy)

    pm = (metrics_summary or {}).get("planning_metrics", {}) or {}
    shortage_rate = float(pm.get("ShortageRate", pm.get("BacklogRate", 0.0)))
    need_replan = (not ver["ok"]) or (shortage_rate > 0.05)

    return {
        "verify": ver,
        "fixes": fixes,
        "critique": critique_text,
        "crosscheck": crosscheck_text,
        "policy": policy,
        "need_replan": need_replan,
    }


# =========================================================
# CLI entrypoint
# =========================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Planner Evaluator (verify → fixes → policy update)")
    ap.add_argument("--plan_csv", required=True, help="production_plan.csv 경로")
    ap.add_argument("--daily_capacity", type=float, required=True, help="일일 CAPA")
    ap.add_argument("--metrics_json", type=str, default=None, help="planning_metrics.json 경로(옵션)")
    ap.add_argument("--policy_path", type=str, default=None, help="정책 저장/로드 파일 경로(옵션)")
    ap.add_argument("--llm_enabled", action="store_true", help="LLM 기반 critique/crosscheck 사용")
    ap.add_argument("--out_json", type=str, default=None, help="감사 결과 저장 경로(기본: plan_csv 옆 governance_audit.json)")

    # MC options
    ap.add_argument("--mc_scenarios_csv", type=str, default=None,
                    help="MC 시나리오 수요 CSV (scenario_id, day_idx, Product_Number, demand)")
    ap.add_argument("--initial_inventory", type=float, default=0.0, help="MC 시뮬레이션 초기 재고")
    ap.add_argument("--mc_out_json", type=str, default=None, help="MC 검증 결과 JSON 저장 경로")
    ap.add_argument("--mc_alpha", type=float, default=0.9, help="MC 요약 분위수 (0.9=VaR)")
    ap.add_argument("--mc_wb", type=float, default=1.0, help="MC Loss 가중치: ShortageRate")
    ap.add_argument("--mc_wi", type=float, default=0.2, help="MC Loss 가중치: InventoryRate")
    ap.add_argument("--mc_fail_threshold", type=float, default=0.03, help="Fail 기준: ShortageRate > threshold")

    args = ap.parse_args()

    # plan 로드 + 숫자 보정
    plan_df = pd.read_csv(args.plan_csv)
    plan_df = _ensure_cols(plan_df, ["Product_Number", "day_idx", "produce"], fill=0.0)

    plan_df["Product_Number"] = plan_df["Product_Number"].map(_normalize_product)
    plan_df["day_idx"] = pd.to_numeric(plan_df["day_idx"], errors="coerce").fillna(0).astype(int)
    plan_df["produce"] = pd.to_numeric(plan_df["produce"], errors="coerce").fillna(0.0)

    for c in ["demand", "backlog", "end_inventory", "shortage"]:
        if c in plan_df.columns:
            plan_df[c] = pd.to_numeric(plan_df[c], errors="coerce").fillna(0.0)

    # MC 검증(옵션)
    mc_validation = None
    if args.mc_scenarios_csv:
        scenarios_df = load_mc_scenarios(args.mc_scenarios_csv)
        mc_validation = mc_validate_plan(
            plan_df=plan_df,
            scenarios_df=scenarios_df,
            initial_inventory=float(args.initial_inventory),
            alpha=float(args.mc_alpha),
            daily_capacity=float(args.daily_capacity),
            w_b=float(args.mc_wb),
            w_i=float(args.mc_wi),
            fail_threshold=float(args.mc_fail_threshold),
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