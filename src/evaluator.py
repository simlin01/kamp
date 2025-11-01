# src/evaluator.py
# -*- coding: utf-8 -*-
"""
evaluator.py
- 통합 감사(Audit): 규칙 기반 검증 + LLM 기반 자기평가/상호검증
- 정책 학습(Policy): 성과 기반 파라미터 자동 업데이트, 저장/로딩

python -m src.evaluator \
  --plan_csv ./outputs/production_plan.csv \
  --daily_capacity 5000 \
  --metrics_json ./outputs/planning_metrics.json \
  --policy_path ./outputs/policy.json \
  --out_json ./outputs/governance_audit.json

"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import os, json
import pandas as pd

_EPS = 1e-9

# ========== 규칙 기반 Verifier ==========
def verify_plan(plan_df: pd.DataFrame, daily_capacity: float) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []

    # 1) CAPA 위반
    by_day = plan_df.groupby("day_idx")["produce"].sum()
    viol = by_day[by_day > daily_capacity + 1e-9]
    if not viol.empty:
        issues.append({
            "type": "CAPA_EXCEEDED",
            "days": list(map(int, viol.index.tolist())),
            "values": [float(v) for v in viol.round(2).tolist()]
        })

    # 2) 음수값
    for c in ["demand", "produce", "backlog", "end_inventory"]:
        neg = plan_df[plan_df[c] < -1e-9]
        if len(neg):
            issues.append({
                "type": "NEGATIVE_VALUES",
                "column": c,
                "count": int(len(neg))
            })

    # 3) 재고/백로그 동시 양수 비율
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
        t = issue["type"]
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
    return {"suggestions": suggestions}

# ========== LLM 기반 Evaluator (옵션) ==========
def llm_critique(metrics_summary: Dict[str, Any], plan_df_sample: pd.DataFrame, enabled: bool = False) -> str:
    if not enabled:
        return ""
    # 외부 LLM 호출을 여기에 붙이거나, report_llm의 헬퍼를 재사용
    # return call_llm(prompt)
    return "(LLM critique placeholder)"

def llm_crosscheck(planner_note: str, reporter_note: str, enabled: bool = False) -> str:
    if not enabled:
        return ""
    # return call_llm(...)
    return "(LLM cross-check placeholder)"

# ========== Policy (경험 학습/저장) ==========
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(policy, f, ensure_ascii=False, indent=2)

def update_policy_from_outcomes(policy: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    pm = metrics.get("planning_metrics", {}) or {}
    backlog = float(pm.get("BacklogRate", 0.0))
    util = float(pm.get("Utilization", 1.0))

    # 간단 규칙 예시: 백로그가 높으면 핵심 가중치↑, 스무딩↓(탄력↑)
    if backlog > 0.03:
        policy["lambda_smooth"] = max(0.2, float(policy.get("lambda_smooth", 1.0)) - 0.2)
        wm = policy.get("WEIGHT_MAP", {})
        wm["0"] = float(wm.get("0", 5.0)) + 0.5
        policy["WEIGHT_MAP"] = wm

    # 활용률 낮으면 변동 억제(스무딩↑)로 현실화 유도
    if util < 0.7:
        policy["lambda_smooth"] = min(3.0, float(policy.get("lambda_smooth", 1.0)) + 0.2)

    return policy

# ========== 통합 오케스트레이션 ==========
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
    반환: {"verify":..., "fixes":..., "critique":..., "crosscheck":..., "policy":..., "need_replan": bool}
    """
    ver = verify_plan(plan_df, daily_capacity)
    fixes = suggest_fixes(plan_df, ver["issues"])

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

if __name__ == "__main__":
    import argparse, os, json

    ap = argparse.ArgumentParser(description="Planner Evaluator (verify → fixes → policy update)")
    ap.add_argument("--plan_csv", required=True, help="production_plan.csv 경로")
    ap.add_argument("--daily_capacity", type=float, required=True, help="일일 CAPA")
    ap.add_argument("--metrics_json", type=str, default=None, help="planning_metrics.json 경로(옵션)")
    ap.add_argument("--policy_path", type=str, default=None, help="정책 저장/로드 파일 경로(옵션)")
    ap.add_argument("--llm_enabled", action="store_true", help="LLM 기반 critique/crosscheck 사용")
    ap.add_argument("--out_json", type=str, default=None, help="감사 결과 저장 경로(기본: plan_csv 옆 audit.json)")
    args = ap.parse_args()

    # 입력 불러오기
    plan_df = pd.read_csv(args.plan_csv)
    # 숫자 보정 (문자열로 저장된 .2f 대응)
    for c in ["demand", "produce", "backlog", "end_inventory"]:
        if c in plan_df.columns:
            plan_df[c] = pd.to_numeric(plan_df[c], errors="coerce").fillna(0.0)

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

    # 출력 저장
    out_path = args.out_json or os.path.join(os.path.dirname(args.plan_csv), "governance_audit.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved evaluator output → {out_path}")
    if not result.get("verify", {}).get("ok", True):
        print("[WARN] Verification issues detected. See 'issues' in the JSON.")
