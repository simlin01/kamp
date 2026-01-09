# src/evaluator.py
# -*- coding: utf-8 -*-
"""
evaluator.py
- 통합 감사(Audit): 규칙 기반 검증 + (옵션) LLM 기반 자기평가/상호검증
- 정책 학습(Policy): 성과 기반 파라미터 자동 업데이트, 저장/로딩
- (NEW) MC 시나리오 검증:
    * 입력: mc_scenarios_csv (scenario_id, day_idx, Product_Number, demand)
    * 고정 생산계획(plan_df)을 시나리오 수요에 대해 시뮬레이션하여 backlog 리스크 요약 저장

CLI 예시:
python -m src.evaluator \
  --plan_csv ./outputs/outputs/production_plan.csv \
  --daily_capacity 5000 \
  --metrics_json ./outputs/outputs/planning_metrics.json \
  --policy_path ./outputs/policy.json \
  --out_json ./outputs/governance_audit.json \
  --mc_scenarios_csv ./outputs/outputs/pred_final_mc.csv \
  --initial_inventory 0.0 \
  --mc_out_json ./outputs/outputs/mc_validation.json
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import os, json
import numpy as np
import pandas as pd

_EPS = 1e-9

# =========================================================
# 규칙 기반 Verifier
# =========================================================
def verify_plan(plan_df: pd.DataFrame, daily_capacity: float) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []

    # 1) CAPA 위반
    if "day_idx" in plan_df.columns and "produce" in plan_df.columns:
        by_day = plan_df.groupby("day_idx")["produce"].sum()
        viol = by_day[by_day > daily_capacity + 1e-9]
        if not viol.empty:
            issues.append({
                "type": "CAPA_EXCEEDED",
                "days": list(map(int, viol.index.tolist())),
                "values": [float(v) for v in viol.round(2).tolist()]
            })
    else:
        issues.append({
            "type": "SCHEMA_MISSING",
            "reason": "plan_df must contain day_idx and produce"
        })

    # 2) 음수값
    for c in ["demand", "produce", "backlog", "end_inventory"]:
        if c in plan_df.columns:
            neg = plan_df[plan_df[c] < -1e-9]
            if len(neg):
                issues.append({
                    "type": "NEGATIVE_VALUES",
                    "column": c,
                    "count": int(len(neg))
                })

    # 3) 재고/백로그 동시 양수 비율
    if {"backlog", "end_inventory"} <= set(plan_df.columns):
        both_pos = ((plan_df["backlog"] > 0) & (plan_df["end_inventory"] > 0)).mean()
        if both_pos > 0.01:
            issues.append({
                "type": "INV_BACKLOG_BOTH_POS",
                "rate": float(both_pos)
            })

    return {"ok": len(issues) == 0, "issues": issues}


def suggest_fixes(plan_df: pd.DataFrame, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
    suggestions: List[Dict[str, Any]] = []
    for issue in issues:
        t = issue.get("type")
        if t == "CAPA_EXCEEDED":
            suggestions.append({
                "target": "lambda_smooth",
                "action": "increase",
                "reason": "일별 생산 변동성 완화로 CAPA 피크를 줄입니다."
            })
        elif t == "INV_BACKLOG_BOTH_POS":
            suggestions.append({
                "target": "WEIGHT_MAP.low_priority",
                "action": "decrease",
                "reason": "저우선 클러스터 생산 비중을 낮춰 백로그 집중 해소."
            })
        elif t == "NEGATIVE_VALUES":
            suggestions.append({
                "target": "data_cleaning",
                "action": "sanity_check",
                "reason": f"{issue.get('column')} 컬럼 음수값 발견. 입력/전처리 확인 필요."
            })
        elif t == "SCHEMA_MISSING":
            suggestions.append({
                "target": "schema",
                "action": "fix",
                "reason": issue.get("reason", "입력 스키마 확인 필요")
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
    "daily_capacity": None  # None이면 config의 값을 따름
}

def load_policy(path: str) -> Dict[str, Any]:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(json.dumps(_DEFAULT_POLICY))

def save_policy(path: str, policy: Dict[str, Any]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(policy, f, ensure_ascii=False, indent=2)

def update_policy_from_outcomes(policy: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    pm = metrics.get("planning_metrics", {}) or {}
    backlog = float(pm.get("BacklogRate", 0.0))
    util = float(pm.get("Utilization", 1.0))

    # 백로그가 높으면: 핵심 가중치↑, 스무딩↓(탄력↑)
    if backlog > 0.03:
        policy["lambda_smooth"] = max(0.2, float(policy.get("lambda_smooth", 1.0)) - 0.2)
        wm = policy.get("WEIGHT_MAP", {}) or {}
        wm["0"] = float(wm.get("0", 5.0)) + 0.5
        policy["WEIGHT_MAP"] = wm

    # 활용률 낮으면: 변동 억제(스무딩↑)로 현실화 유도
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
    df["Product_Number"] = df["Product_Number"].astype(str).str.replace(r"\.0$", "", regex=True)
    return df

def simulate_fixed_plan(
    plan_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    initial_inventory: float = 0.0
) -> pd.DataFrame:
    """
    고정 생산계획(plan_df)을 특정 수요(demand_df) 하에서 시뮬레이션.
    plan_df: day_idx, Product_Number, produce
    demand_df: day_idx, Product_Number, demand
    """
    required_plan = {"Product_Number", "day_idx", "produce"}
    required_dem = {"Product_Number", "day_idx", "demand"}
    if not (required_plan <= set(plan_df.columns)):
        raise ValueError(f"plan_df missing cols: {required_plan - set(plan_df.columns)}")
    if not (required_dem <= set(demand_df.columns)):
        raise ValueError(f"demand_df missing cols: {required_dem - set(demand_df.columns)}")

    prod = plan_df.groupby(["Product_Number", "day_idx"], as_index=False)["produce"].sum()
    dem = demand_df.rename(columns={"demand": "demand_s"}).copy()

    sim = prod.merge(dem, on=["Product_Number", "day_idx"], how="left")
    sim["demand_s"] = sim["demand_s"].fillna(0.0)

    sim = sim.sort_values(["Product_Number", "day_idx"]).reset_index(drop=True)

    out_rows = []
    for p, g in sim.groupby("Product_Number"):
        stock = float(initial_inventory)   # stock = inv - backlog
        for _, r in g.iterrows():
            stock = stock + float(r["produce"]) - float(r["demand_s"])
            inv = max(stock, 0.0)
            backlog = max(-stock, 0.0)
            out_rows.append({
                "Product_Number": p,
                "day_idx": int(r["day_idx"]),
                "produce": float(r["produce"]),
                "demand": float(r["demand_s"]),
                "end_inventory": float(inv),
                "backlog": float(backlog),
            })
    return pd.DataFrame(out_rows)

def summarize_mc_results(per_scenario_metrics: pd.DataFrame, alpha: float = 0.9) -> dict:
    """
    per_scenario_metrics: columns include scenario_id, BacklogRate, TotalBacklog
    """
    def q(arr: np.ndarray) -> float:
        return float(np.quantile(arr, alpha)) if arr.size else 0.0

    out: Dict[str, Any] = {}
    for col in ["BacklogRate", "TotalBacklog"]:
        if col in per_scenario_metrics.columns:
            arr = per_scenario_metrics[col].astype(float).to_numpy()
            out[col] = {
                "mean": float(arr.mean()) if arr.size else 0.0,
                "p90": q(arr),
                "max": float(arr.max()) if arr.size else 0.0,
            }

    if "BacklogRate" in per_scenario_metrics.columns:
        out["FailRate"] = float((per_scenario_metrics["BacklogRate"] > 0.0).mean())

    return out

def mc_validate_plan(
    plan_df: pd.DataFrame,
    scenarios_df: pd.DataFrame,
    initial_inventory: float = 0.0,
    alpha: float = 0.9,
) -> Dict[str, Any]:
    """
    고정 계획(plan_df)을 MC 수요 시나리오(scenarios_df)에 대해 평가.
    반환:
      {
        "summary": {...},
        "per_scenario": [...]
      }
    """
    # sanitize
    plan = plan_df.copy()
    plan["Product_Number"] = plan["Product_Number"].astype(str).str.replace(r"\.0$", "", regex=True)
    plan["day_idx"] = pd.to_numeric(plan["day_idx"], errors="coerce").fillna(0).astype(int)
    plan["produce"] = pd.to_numeric(plan["produce"], errors="coerce").fillna(0.0)

    sc = scenarios_df.copy()
    sc["scenario_id"] = pd.to_numeric(sc["scenario_id"], errors="coerce").fillna(0).astype(int)
    sc["Product_Number"] = sc["Product_Number"].astype(str).str.replace(r"\.0$", "", regex=True)
    sc["day_idx"] = pd.to_numeric(sc["day_idx"], errors="coerce").fillna(0).astype(int)
    sc["demand"] = pd.to_numeric(sc["demand"], errors="coerce").fillna(0.0)

    per_rows = []
    for sid, g in sc.groupby("scenario_id"):
        sim = simulate_fixed_plan(
            plan_df=plan,
            demand_df=g[["Product_Number", "day_idx", "demand"]],
            initial_inventory=initial_inventory,
        )
        total_dem = float(sim["demand"].sum())
        total_back = float(sim["backlog"].sum())
        back_rate = float(total_back / (total_dem + _EPS))
        per_rows.append({
            "scenario_id": int(sid),
            "BacklogRate": back_rate,
            "TotalBacklog": total_back,
        })

    per_df = pd.DataFrame(per_rows)
    summary = summarize_mc_results(per_df, alpha=alpha) if not per_df.empty else {"FailRate": 0.0}

    return {
        "summary": summary,
        "per_scenario": per_df.to_dict(orient="records"),
    }

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
    reporter_note: str = ""
) -> Dict[str, Any]:
    """
    1) 규칙 검증 → 2) 수정 제안 → 3) (옵션) LLM 자기평가/상호검증 → 4) 정책 업데이트/저장
    반환:
      {
        "verify":..., "fixes":..., "critique":..., "crosscheck":...,
        "policy":..., "need_replan": bool
      }
    """
    ver = verify_plan(plan_df, daily_capacity)
    fixes = suggest_fixes(plan_df, ver.get("issues", []))

    critique_text = llm_critique(metrics_summary or {}, plan_df.head(200), enabled=llm_enabled)
    crosscheck_text = llm_crosscheck(planner_note, reporter_note, enabled=llm_enabled)

    policy = load_policy(policy_path) if policy_path else json.loads(json.dumps(_DEFAULT_POLICY))
    policy = update_policy_from_outcomes(policy, metrics_summary or {})
    if policy_path:
        save_policy(policy_path, policy)

    backlog_rate = float((metrics_summary or {}).get("planning_metrics", {}).get("BacklogRate", 0.0))
    need_replan = (not ver["ok"]) or (backlog_rate > 0.05)

    return {
        "verify": ver,
        "fixes": fixes,
        "critique": critique_text,
        "crosscheck": crosscheck_text,
        "policy": policy,
        "need_replan": need_replan
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
    ap.add_argument("--mc_alpha", type=float, default=0.9, help="MC 요약 분위수 (0.9=p90)")

    args = ap.parse_args()

    # plan 로드
    plan_df = pd.read_csv(args.plan_csv)
    # 숫자 보정
    for c in ["demand", "produce", "backlog", "end_inventory"]:
        if c in plan_df.columns:
            plan_df[c] = pd.to_numeric(plan_df[c], errors="coerce").fillna(0.0)
    if "Product_Number" in plan_df.columns:
        plan_df["Product_Number"] = plan_df["Product_Number"].astype(str).str.replace(r"\.0$", "", regex=True)
    if "day_idx" in plan_df.columns:
        plan_df["day_idx"] = pd.to_numeric(plan_df["day_idx"], errors="coerce").fillna(0).astype(int)

    # MC 검증(옵션)
    mc_validation = None
    if args.mc_scenarios_csv:
        scenarios_df = load_mc_scenarios(args.mc_scenarios_csv)
        mc_validation = mc_validate_plan(
            plan_df=plan_df,
            scenarios_df=scenarios_df,
            initial_inventory=args.initial_inventory,
            alpha=float(args.mc_alpha),
        )

        mc_out_path = args.mc_out_json
        if not mc_out_path:
            base_dir = os.path.dirname(args.out_json) if args.out_json else os.path.dirname(args.plan_csv)
            mc_out_path = os.path.join(base_dir, "mc_validation.json")

        os.makedirs(os.path.dirname(mc_out_path), exist_ok=True)
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
        daily_capacity=args.daily_capacity,
        metrics_summary=metrics_summary,
        policy_path=args.policy_path,
        llm_enabled=args.llm_enabled,
    )

    # mc 요약을 governance audit에도 포함
    if mc_validation is not None:
        result["mc_validation"] = mc_validation.get("summary", mc_validation)

    # 결과 저장
    out_path = args.out_json or os.path.join(os.path.dirname(args.plan_csv), "governance_audit.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved evaluator output → {out_path}")

    if not result.get("verify", {}).get("ok", True):
        print("[WARN] Verification issues detected. See 'issues' in the JSON.")