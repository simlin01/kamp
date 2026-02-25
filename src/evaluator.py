#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluator.py

- 통합 감사(Audit): 규칙 기반 검증 + (옵션) LLM 기반 자기평가/상호검증
- 정책 학습(Policy): 성과 기반 파라미터 자동 업데이트, 저장/로딩
- MC 시나리오 검증(ShortageRate 기반, planner_opt.py와 1:1 정합)

[기존 유지]
- product_col 유연성
- shortage 정의 = backlog 증가분(신규 미충족)  ✅ planner_opt와 동일
- CVaR mean plan에서 INV_BACKLOG_BOTH_POS 완화(WARN)

[이번 추가(핵심)]
✅ mc_validate_plan()에 weight_map + cluster_info를 "옵션"으로 받아서,
   - 기존 지표(무가중): ShortageRate / loss 그대로 유지
   - 추가 지표(가중): WeightedShortageRate / WeightedInventoryRate / weighted_loss 를 함께 저장
   - 요약(summary)에도 weighted_* 통계(mean/var/cvar) 추가
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


def _load_json_map_or_none(s: Optional[str]) -> Optional[dict]:
    if s is None:
        return None
    s2 = str(s).strip()
    if not s2:
        return None
    if s2.startswith("@"):
        with open(s2[1:], "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(s2)


def _as_int_key_float_map(d: Optional[dict]) -> Optional[Dict[int, float]]:
    if d is None:
        return None
    out: Dict[int, float] = {}
    for k, v in d.items():
        out[int(k)] = float(v)
    return out


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
# Cluster info loader (for weighted MC metrics)
# =========================================================

def load_cluster_info_from_feat(feat_csv: str, product_col: str = "Product_Number") -> Dict[str, int]:
    """
    feat.csv 에서 (Product_Number -> Cluster) 매핑 로드.
    - 없으면 빈 dict 반환(=모두 Cluster=1 가정은 weight 쪽에서 처리)
    """
    if (not feat_csv) or (not os.path.exists(feat_csv)):
        return {}
    df = pd.read_csv(feat_csv)
    if product_col not in df.columns:
        # normalize 시도
        cols = {c: c.strip() for c in df.columns}
        if product_col not in cols:
            return {}
    if "Cluster" not in df.columns:
        return {}
    m = df[[product_col, "Cluster"]].drop_duplicates()
    m[product_col] = m[product_col].map(_normalize_product)
    m["Cluster"] = pd.to_numeric(m["Cluster"], errors="coerce").fillna(1).astype(int)
    return m.set_index(product_col)["Cluster"].to_dict()


def _build_product_weights(
    products: List[str],
    *,
    cluster_info: Optional[Dict[str, int]],
    weight_map: Optional[Dict[int, float]],
    default_cluster: int = 1,
    default_weight: float = 1.0,
) -> np.ndarray:
    """
    products 순서에 맞춘 weight 벡터 (float).
    - cluster_info가 없거나 제품이 없으면 default_cluster 적용
    - weight_map이 없거나 cluster 키가 없으면 default_weight 적용
    """
    cluster_info = cluster_info or {}
    weight_map = weight_map or {}
    w = np.empty(len(products), dtype=float)
    for i, p in enumerate(products):
        cid = int(cluster_info.get(p, default_cluster))
        w[i] = float(weight_map.get(cid, default_weight))
    return w


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

    sim = sim.sort_values([product_col, "day_idx"]).reset_index(drop=True)

    out_rows = []
    for p, g in sim.groupby(product_col, sort=False):
        stock = float(initial_inventory)
        prev_backlog = 0.0
        min_day = int(g["day_idx"].min())
        for r in g.itertuples(index=False):
            d = int(r.day_idx)
            prod = float(r.produce)
            demd = float(r.demand_s)

            stock = stock + prod - demd
            inv = max(stock, 0.0)
            backlog = max(-stock, 0.0)

            # ✅ shortage = backlog increase
            if d == min_day:
                shortage = backlog
            else:
                shortage = max(backlog - prev_backlog, 0.0)

            prev_backlog = backlog

            fulfilled = demd - shortage  # 참고용
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
    w_loss_col: Optional[str] = None,
) -> Dict[str, Any]:
    loss = pd.to_numeric(per_scenario[loss_col], errors="coerce").dropna().to_numpy(dtype=float)
    sr = pd.to_numeric(per_scenario["ShortageRate"], errors="coerce").dropna().to_numpy(dtype=float)

    out = {
        "S": int(len(per_scenario)),
        "loss_mean": float(np.mean(loss)) if loss.size else float("nan"),
        "loss_var": _quantile(loss, alpha),
        "loss_cvar": _cvar(loss, alpha),
        "shortage_rate_mean": float(np.mean(sr)) if sr.size else float("nan"),
        "shortage_rate_var": _quantile(sr, alpha),
        "shortage_rate_cvar": _cvar(sr, alpha),
    }

    # ✅ weighted summary (optional)
    if w_loss_col and (w_loss_col in per_scenario.columns):
        wloss = pd.to_numeric(per_scenario[w_loss_col], errors="coerce").dropna().to_numpy(dtype=float)
        wsr = pd.to_numeric(per_scenario.get("WeightedShortageRate", np.nan), errors="coerce").dropna().to_numpy(dtype=float)
        out.update({
            "weighted_loss_mean": float(np.mean(wloss)) if wloss.size else float("nan"),
            "weighted_loss_var": _quantile(wloss, alpha),
            "weighted_loss_cvar": _cvar(wloss, alpha),
            "weighted_shortage_rate_mean": float(np.mean(wsr)) if wsr.size else float("nan"),
            "weighted_shortage_rate_var": _quantile(wsr, alpha),
            "weighted_shortage_rate_cvar": _cvar(wsr, alpha),
        })

    return out


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
    # ✅ NEW (optional)
    cluster_info: Optional[Dict[str, int]] = None,   # product -> cluster
    weight_map: Optional[Dict[int, float]] = None,   # cluster -> weight
    weight_scale: int = 100,                         # planner_opt의 W_SCALE과 같은 의미(정수화 분모 복원용)
) -> Dict[str, Any]:
    """
    scenarios_df: long-form (scenario_id, day_idx, product_col, demand)
    return:
      - per_scenario metrics + summary
      - 기존(무가중) 지표는 유지
      - (옵션) weighted 지표는 같이 추가
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

    plan = plan_df.copy()
    plan = _ensure_cols(plan, [product_col, "day_idx", "produce"], fill=0.0)
    plan[product_col] = plan[product_col].map(_normalize_product)
    plan["day_idx"] = pd.to_numeric(plan["day_idx"], errors="coerce").fillna(0).astype(int)
    plan["produce"] = pd.to_numeric(plan["produce"], errors="coerce").fillna(0.0)

    verify = verify_plan(plan, daily_capacity=daily_capacity, product_col=product_col)

    # ✅ weighted 활성 여부
    use_weighted = (cluster_info is not None) and (weight_map is not None) and (len(weight_map) > 0)

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

        row = {
            "scenario_id": int(sid),
            "TotalDemand": total_demand,
            "TotalShortage": total_short,
            "ShortageRate": shortage_rate,
            "InventoryRate": inv_rate,
            "loss": loss,
            "fail": bool(shortage_rate > float(fail_threshold)),
        }

        # ✅ weighted metrics (planner_opt 철학과 정렬)
        # - WeightedShortageRate = sum(w_i * shortage_i) / total_demand
        # - WeightedInventoryRate = sum(w_i * inv_i) / (P*D*day_cap)
        # - weighted_loss = w_b*WeightedShortageRate + w_i*WeightedInventoryRate
        if use_weighted and (not sim.empty):
            prods = sim[product_col].astype(str).map(_normalize_product).unique().tolist()
            w_vec = _build_product_weights(
                prods,
                cluster_info=cluster_info,
                weight_map=weight_map,
                default_cluster=1,
                default_weight=1.0,
            )
            w_int = np.rint(w_vec * float(weight_scale)).astype(int)  # int weights (like planner)

            w_map_prod = {p: int(wi) for p, wi in zip(prods, w_int)}
            sim2 = sim.copy()
            sim2["_w"] = sim2[product_col].map(w_map_prod).fillna(int(weight_scale)).astype(int)

            total_short_w_int = float((sim2["_w"] * sim2["shortage"]).sum())
            total_inv_w_int = float((sim2["_w"] * sim2["end_inventory"]).sum())

            # 분모에 weight_scale을 곱해 스케일 복원(= planner_opt와 동일한 정수화 철학)
            w_short_rate = float(total_short_w_int / ((total_demand + 1e-9) * float(weight_scale)))
            denom_inv = float(max(sim2[product_col].nunique(), 1) * max(sim2["day_idx"].nunique(), 1) * (daily_capacity + 1e-9))
            w_inv_rate = float(total_inv_w_int / (denom_inv * float(weight_scale)))

            w_loss = float(w_b * w_short_rate + w_i * w_inv_rate)

            row.update({
                "WeightedShortageRate": w_short_rate,
                "WeightedInventoryRate": w_inv_rate,
                "weighted_loss": w_loss,
                "weight_scale": int(weight_scale),
                "weighted_enabled": True,
            })
        else:
            row.update({
                "weighted_enabled": False,
            })

        per_rows.append(row)

    per_df = pd.DataFrame(per_rows).sort_values("scenario_id").reset_index(drop=True)

    summary = summarize_mc_results(
        per_df,
        alpha=float(alpha),
        loss_col="loss",
        w_loss_col="weighted_loss" if ("weighted_loss" in per_df.columns) else None,
    )

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
            "weighted_enabled": bool(use_weighted),
            "weight_scale": int(weight_scale),
            "weight_map": weight_map if use_weighted else None,
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

    # ✅ NEW: weighted options (optional)
    ap.add_argument("--feat_csv", type=str, default=None, help="feat.csv 경로(가중 지표 계산용 Cluster 로드)")
    ap.add_argument("--weight_map", type=str, default=None,
                    help="cluster->weight JSON string or @file. 예: '{\"0\":5,\"1\":2,\"2\":0.5,\"3\":1}'")
    ap.add_argument("--weight_scale", type=int, default=100, help="weight 정수화 스케일 (planner_opt와 동일 의미)")

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

    # metrics_json 로드(옵션)
    metrics_summary = None
    if args.metrics_json and os.path.exists(args.metrics_json):
        with open(args.metrics_json, "r", encoding="utf-8") as f:
            metrics_summary = json.load(f)

    # weighted inputs (optional)
    cluster_info = None
    weight_map = None
    if args.feat_csv and os.path.exists(args.feat_csv) and args.weight_map:
        cluster_info = load_cluster_info_from_feat(args.feat_csv, product_col=args.product_col)
        weight_map = _as_int_key_float_map(_load_json_map_or_none(args.weight_map))

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
            cluster_info=cluster_info,
            weight_map=weight_map,
            weight_scale=int(args.weight_scale),
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