#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
planner_opt.py — CP-SAT 생산계획 (CVaR objective)

[핵심 수정]
1) ✅ detect_horizons()를 한국어 컬럼에서도 "항상" 올바른 순서로 정렬
   - _sort_horizons_kor()를 실제로 적용 (기존: 정의만 있고 미사용)
   - "T일 예상/예정 수주량", "T+3일 ..." 같은 케이스에서 day_idx 축 꼬임 방지
2) ✅ preprocess_forecast()를 prod_col 인자로 받아 main/forecast와 컬럼명 통일
   - (기존: Product_Number 고정)
3) ✅ load_cluster_info()도 prod_col 인자화 + 결측 클러스터 안전 처리
4) ✅ detect_horizons 후보 수집 로직을 정규식 기반으로 단순/강건화
   - "T", "T+3" 같은 축약 컬럼
   - "T일 예상 수주량", "T+3일 예정 수주량" 같은 한국어 풀네임 컬럼
   - 기타 "T+3 ..." 부분 문자열 포함 컬럼

※ 최적화 모델(CVaR 정의/로직)은 사용자가 주신 버전을 유지(정합성 목적).
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple, Any
import json
import os
import re
import unicodedata
import argparse
import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

# =========================================================
# (A) Utils & Preprocess
# =========================================================

WEIRD_SPACES = ["\ufeff", "\u200b", "\u200c", "\u200d", "\xa0"]

def _normalize_col(c: str) -> str:
    c2 = unicodedata.normalize("NFKC", str(c))
    for w in WEIRD_SPACES:
        c2 = c2.replace(w, "")
    c2 = re.sub(r"\s+", " ", c2).strip()
    return c2


def _sort_horizons_kor(hs: List[str]) -> List[str]:
    """
    한국어 horizon 컬럼까지 안정 정렬:
    - "T일 ..." -> 0
    - "T+1일 ..." -> 1
    - 축약형 "T" -> 0, "T+3" -> 3
    """
    def key(h: str) -> int:
        s = _normalize_col(h)

        # 축약형
        if s == "T":
            return 0
        m0 = re.fullmatch(r"T\+(\d+)", s)
        if m0:
            return int(m0.group(1))

        # 한국어 풀네임
        if s.startswith("T일"):
            return 0
        m = re.search(r"T\+(\d+)", s)
        if m:
            return int(m.group(1))

        # 혹시 "T+3일"처럼 붙어있어도 잡힘
        m2 = re.search(r"T\+(\d+)\s*일", s)
        if m2:
            return int(m2.group(1))

        return 10_000

    # 중복 제거 + 안정 정렬
    hs_u = list(dict.fromkeys(hs))
    return sorted(hs_u, key=key)


def preprocess_forecast(df: pd.DataFrame, prod_col: str = "Product_Number", dt_col: str = "DateTime") -> pd.DataFrame:
    """
    - 컬럼 normalize
    - DateTime이 있으면 제품별 최신 스냅샷으로 축약
    - DateTime이 없으면 그대로(이미 product-level이면 OK)
    """
    df = df.copy()
    df.columns = [_normalize_col(c) for c in df.columns]

    if prod_col not in df.columns:
        raise KeyError(f"'{prod_col}' 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}")

    if dt_col not in df.columns:
        return df

    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    valid = df.dropna(subset=[dt_col])
    if valid.empty:
        return df.drop(columns=[dt_col], errors="ignore")

    latest_dt = (
        valid.groupby(prod_col, as_index=False)[dt_col].max()
             .rename(columns={dt_col: "_LatestDT"})
    )
    merged = df.merge(latest_dt, on=prod_col, how="inner")
    picked = merged[merged[dt_col] == merged["_LatestDT"]].copy()

    num_cols = picked.select_dtypes(include="number").columns.tolist()
    non_num_cols = [c for c in picked.columns if c not in num_cols]

    # num mean / others first
    agg = {**{c: "first" for c in non_num_cols}, **{c: "mean" for c in num_cols}}
    snapped = picked.groupby(prod_col, as_index=False).agg(agg)
    return snapped.drop(columns=["_LatestDT", dt_col], errors="ignore")


def load_cluster_info(feat_file: str, prod_col: str = "Product_Number") -> Dict[str, int]:
    """
    feat.csv에서 제품별 Cluster 매핑 로드.
    - Cluster 없으면 전체 1로 처리(안전)
    """
    df = pd.read_csv(feat_file)
    df.columns = [_normalize_col(c) for c in df.columns]

    if prod_col not in df.columns:
        raise ValueError(f"feat.csv에는 '{prod_col}' 컬럼이 필요합니다.")

    if "Cluster" not in df.columns:
        print("[WARN] feat.csv에 'Cluster' 컬럼이 없습니다. 모든 제품을 Cluster=1로 처리합니다.")
        return {str(p): 1 for p in df[prod_col].astype(str).str.replace(r"\.0$", "", regex=True).unique().tolist()}

    m = df[[prod_col, "Cluster"]].drop_duplicates()
    m[prod_col] = m[prod_col].astype(str).str.replace(r"\.0$", "", regex=True)
    m["Cluster"] = pd.to_numeric(m["Cluster"], errors="coerce").fillna(1).astype(int)
    return m.set_index(prod_col)["Cluster"].to_dict()

# =========================================================
# (A-1) Warm-start Hint loader
# =========================================================

def load_hint_plan_csv(path: str, scale: int = 10, prod_col: str = "Product_Number") -> Dict[Tuple[str, int], int]:
    """
    hint plan CSV -> {(Product_Number, day_idx): produce_int_scaled}
    """
    df = pd.read_csv(path)
    need = {prod_col, "day_idx", "produce"}
    if not need.issubset(df.columns):
        raise ValueError(f"hint_plan_csv에는 {sorted(need)} 컬럼이 필요합니다. 현재={list(df.columns)}")

    df[prod_col] = df[prod_col].astype(str).str.replace(r"\.0$", "", regex=True)
    df["day_idx"] = pd.to_numeric(df["day_idx"], errors="coerce").fillna(0).astype(int)
    df["produce"] = pd.to_numeric(df["produce"], errors="coerce").fillna(0.0)

    m: Dict[Tuple[str, int], int] = {}
    for r in df.itertuples(index=False):
        p = str(getattr(r, prod_col))
        d = int(r.day_idx)
        v = int(round(float(r.produce) * scale))
        if v < 0:
            v = 0
        m[(p, d)] = v
    return m

# =========================================================
# (A-2) MC loader (raw scenarios)
# =========================================================

def load_mc_scenarios(
    mc_npz: Optional[str],
    mc_csv: Optional[str],
    product_col: str = "Product_Number",
) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    """CVaR용 raw scenarios 반환 (S,P,D), products list"""
    if mc_npz and os.path.exists(mc_npz):
        z = np.load(mc_npz, allow_pickle=True)

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

        if scenarios is None or products is None:
            raise KeyError(f"mc_npz missing scenarios/products. keys={list(z.keys())}")

        scenarios = np.asarray(scenarios, dtype=float)
        products = pd.Series(products).astype(str).str.replace(r"\.0$", "", regex=True).tolist()
        return scenarios, products

    if mc_csv and os.path.exists(mc_csv):
        df = pd.read_csv(mc_csv)

        need = {"scenario_id", "day_idx", product_col, "demand"}
        if not need.issubset(df.columns):
            raise ValueError(f"mc_csv에는 {sorted(need)} 컬럼이 필요합니다. 현재={list(df.columns)}")

        df["scenario_id"] = pd.to_numeric(df["scenario_id"], errors="coerce").fillna(0).astype(int)
        df["day_idx"] = pd.to_numeric(df["day_idx"], errors="coerce").fillna(0).astype(int)
        df[product_col] = df[product_col].astype(str).str.replace(r"\.0$", "", regex=True)
        df["demand"] = pd.to_numeric(df["demand"], errors="coerce").fillna(0.0)

        S = int(df["scenario_id"].max()) + 1
        D = int(df["day_idx"].max()) + 1
        products = sorted(df[product_col].unique().tolist())
        P = len(products)
        pid = {p: i for i, p in enumerate(products)}

        scenarios = np.zeros((S, P, D), dtype=float)
        for r in df.itertuples(index=False):
            s = int(r.scenario_id)
            d = int(r.day_idx)
            p = str(getattr(r, product_col))
            scenarios[s, pid[p], d] = float(r.demand)

        return scenarios, products

    return None, None


def reorder_mc_scenarios_to_forecast(
    scenarios: np.ndarray,
    mc_products: List[str],
    forecast_products: List[str],
) -> np.ndarray:
    mc_index = {p: i for i, p in enumerate(mc_products)}
    S, _, D = scenarios.shape
    out = np.zeros((S, len(forecast_products), D), dtype=float)
    for j, p in enumerate(forecast_products):
        if p in mc_index:
            out[:, j, :] = scenarios[:, mc_index[p], :]
    return out


def detect_horizons(df: pd.DataFrame) -> List[str]:
    """
    horizon 컬럼 후보를 강건하게 수집하고,
    ✅ 반드시 _sort_horizons_kor()로 정렬해서 반환.
    """
    candidates: List[str] = []

    # 1) 축약형: "T", "T+1" ...
    for c in df.columns:
        cc = _normalize_col(c)
        if re.fullmatch(r"T(\+\d+)?", cc):
            candidates.append(c)

    # 2) 한국어 풀네임: "T일 예상/예정 수주량", "T+3일 ..." 등
    rgx_full = re.compile(r"^T(\+\d+)?일\s*(예상|예정)?\s*.*$")  # 예상/예정 없어도 허용
    for c in df.columns:
        cc = _normalize_col(c)
        if rgx_full.match(cc) and ("수주" in cc or "예상" in cc or "예정" in cc):
            candidates.append(c)

    # 3) 마지막 fallback: "T+숫자"가 포함된 컬럼
    for c in df.columns:
        cc = _normalize_col(c)
        if re.search(r"T\+\d+", cc):
            candidates.append(c)

    candidates = _sort_horizons_kor(list(dict.fromkeys(candidates)))

    if not candidates:
        raise ValueError("horizons 자동 감지 실패. --horizons 로 명시해 주세요.")
    return candidates

# =========================================================
# (B) CP-SAT var helpers
# =========================================================

def _make_2d_int(model: cp_model.CpModel, P: int, D: int, lb: int, ub: int, name: str):
    return [[model.NewIntVar(lb, ub, f"{name}_{i}_{d}") for d in range(D)] for i in range(P)]

def _make_3d_int(model: cp_model.CpModel, S: int, P: int, D: int, lb: int, ub: int, name: str):
    return [[[model.NewIntVar(lb, ub, f"{name}_{s}_{i}_{d}") for d in range(D)] for i in range(P)] for s in range(S)]

def _make_2d_bool(model: cp_model.CpModel, P: int, D: int, name: str):
    return [[model.NewBoolVar(f"{name}_{i}_{d}") for d in range(D)] for i in range(P)]

# =========================================================
# (C) Optimization
# =========================================================

_INT64_MAX = 9_000_000_000_000_000_000  # 9e18 safety

def _make_solver(
    solver_seed: int,
    max_time: float,
    workers: int,
    gap: float,
    log_progress: bool,
) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    solver.parameters.relative_gap_limit = float(gap)
    solver.parameters.max_time_in_seconds = float(max_time)
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.log_search_progress = bool(log_progress)
    solver.parameters.random_seed = int(solver_seed)
    return solver

def _safe_int_cap(x: int, cap: int = 5_000_000_000) -> int:
    return int(min(max(int(x), 1), int(cap)))

def optimize_plan(
    forecast_by_product: pd.DataFrame,
    horizons: List[str],
    prod_col: str,
    cluster_info: Dict[str, int],
    daily_capacity: int = 15000,
    lambda_smooth: float = 0.5,
    initial_inventory: float = 0.0,
    int_production: bool = True,  # CLI 호환 유지
    scale: int = 10,
    initial_inventory_map: Optional[Dict[str, float]] = None,
    min_lot_map: Optional[Dict[int, float]] = None,
    safety_stock_map: Optional[Dict[int, float]] = None,
    weight_map: Optional[Dict[int, float]] = None,

    # ---- Warm start ----
    hint_plan: Optional[Dict[Tuple[str, int], int]] = None,

    # ---- CVaR objective ----
    use_cvar_obj: bool = False,
    mc_scenarios: Optional[np.ndarray] = None,     # (S,P,D) in forecast product order
    cvar_alpha: float = 0.9,
    lambda_cvar: float = 0.3,
    lambda_scale: int = 100_000_000,
    mc_wb: float = 1.0,
    mc_wi: float = 0.2,
    loss_scale: int = 100_000,

    # ---- solver controls ----
    solver_seed: int = 42,
    max_time: float = 300.0,
    workers: int = 1,
    gap: float = 0.1,
    log_progress: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    min_lot_map = min_lot_map or {0: 100, 1: 50, 2: 0, 3: 200}
    safety_stock_map = safety_stock_map or {0: 0, 1: 0, 2: 0, 3: 0}
    weight_map = weight_map or {0: 5.0, 1: 2.0, 2: 0.5, 3: 1.0}

    df = preprocess_forecast(forecast_by_product, prod_col=prod_col, dt_col="DateTime")
    df[prod_col] = df[prod_col].astype(str).str.replace(r"\.0$", "", regex=True)

    products = df[prod_col].tolist()
    P, D = len(products), len(horizons)

    demand_f = df[horizons].to_numpy(dtype=float)
    demand_i = np.rint(np.maximum(demand_f, 0.0) * scale).astype(int)  # (P,D)

    day_cap = int(daily_capacity * scale)

    # UB 타이트닝
    PROD_UB = day_cap
    DIFF_UB = day_cap

    model = cp_model.CpModel()
    produce = _make_2d_int(model, P, D, 0, PROD_UB, "produce")
    is_prod = _make_2d_bool(model, P, D, "is_prod")

    # ---- Warm-start (AddHint) ----
    if hint_plan:
        for i, p in enumerate(products):
            for d in range(D):
                key = (p, d)
                if key in hint_plan:
                    v = int(hint_plan[key])
                    if v < 0:
                        v = 0
                    if v > PROD_UB:
                        v = PROD_UB
                    model.AddHint(produce[i][d], v)
                    model.AddHint(is_prod[i][d], 1 if v > 0 else 0)

    # lot/production + CAPA
    for i, p in enumerate(products):
        cid = int(cluster_info.get(p, 1))
        min_lot = int(min_lot_map.get(cid, 0) * scale)
        for d in range(D):
            if min_lot > 0:
                model.Add(produce[i][d] >= min_lot).OnlyEnforceIf(is_prod[i][d])
                model.Add(produce[i][d] <= day_cap).OnlyEnforceIf(is_prod[i][d])
                model.Add(produce[i][d] == 0).OnlyEnforceIf(is_prod[i][d].Not())
            else:
                model.Add(produce[i][d] <= day_cap * is_prod[i][d])

    for d in range(D):
        model.Add(sum(produce[i][d] for i in range(P)) <= day_cap)

    # =====================================================
    # BASIC MODE
    # =====================================================
    if not use_cvar_obj:
        cum_dem = np.cumsum(demand_i, axis=1)  # (P,D)
        STATE_UB = _safe_int_cap(int(cum_dem.max() * 2))

        # stock 기반(음수 허용)
        stock   = _make_2d_int(model, P, D, -STATE_UB, STATE_UB, "stock")
        inv     = _make_2d_int(model, P, D, 0, STATE_UB, "inv")
        backlog = _make_2d_int(model, P, D, 0, STATE_UB, "backlog")
        ZERO = model.NewIntVar(0, 0, "ZERO_BASIC")

        for i, p in enumerate(products):
            cid = int(cluster_info.get(p, 1))
            s_stock = int(safety_stock_map.get(cid, 0) * scale)

            for d in range(D):
                if d == 0:
                    init_inv_i = int(round((initial_inventory_map.get(p, initial_inventory) if initial_inventory_map else initial_inventory) * scale))
                    prev_stock = init_inv_i
                else:
                    prev_stock = stock[i][d-1]

                model.Add(stock[i][d] == prev_stock + produce[i][d] - demand_i[i, d])
                model.AddMaxEquality(inv[i][d], [stock[i][d], ZERO])

                neg_stock = model.NewIntVar(-STATE_UB, STATE_UB, f"neg_stock_{i}_{d}")
                model.Add(neg_stock == -stock[i][d])
                model.AddMaxEquality(backlog[i][d], [neg_stock, ZERO])

                if s_stock > 0:
                    model.Add(inv[i][d] >= s_stock)

        terms = []
        for i, p in enumerate(products):
            cid = int(cluster_info.get(p, 1))
            w = int(round(weight_map.get(cid, 1.0) * 100))
            for d in range(D):
                terms.append(w * backlog[i][d])

        smooth_w = int(round(lambda_smooth * 100))  # 통일: *100
        if smooth_w > 0:
            for i in range(P):
                for d in range(1, D):
                    diff = model.NewIntVar(0, DIFF_UB, f"diff_{i}_{d}")
                    model.Add(diff >= produce[i][d] - produce[i][d-1])
                    model.Add(diff >= produce[i][d-1] - produce[i][d])
                    terms.append(smooth_w * diff)

        model.Minimize(sum(terms))

        solver = _make_solver(solver_seed, max_time, workers, gap, log_progress)
        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(f"OR-Tools: {solver.StatusName(status)} (실행 가능한 계획 실패)")

        rows = []
        for d_in, hcol in enumerate(horizons):
            for i, p in enumerate(products):
                rows.append({
                    "day_idx": int(d_in),
                    "horizon": hcol,
                    "Product_Number": p,
                    "demand": float(demand_i[i, d_in] / scale),
                    "produce": float(solver.Value(produce[i][d_in]) / scale),
                    "end_inventory": float(solver.Value(inv[i][d_in]) / scale),
                    "backlog": float(solver.Value(backlog[i][d_in]) / scale),
                })
        plan_df = pd.DataFrame(rows)
        diag = {
            "mode": "basic",
            "status": solver.StatusName(status),
            "objective": float(solver.ObjectiveValue()),
            "STATE_UB": int(STATE_UB),
            "PROD_UB": int(PROD_UB),
            "smooth_w": int(smooth_w),
        }
        return plan_df, diag

    # =====================================================
    # CVaR MODE (ShortageRate 기반, evaluator와 정합)
    # =====================================================
    if mc_scenarios is None:
        raise ValueError("use_cvar_obj=True 인데 mc_scenarios가 없습니다. (S,P,D) 시나리오 텐서를 넘겨주세요.")

    mc_scenarios = np.asarray(mc_scenarios, dtype=float)
    if mc_scenarios.ndim != 3:
        raise ValueError(f"mc_scenarios는 (S,P,D) 이어야 합니다. got shape={mc_scenarios.shape}")

    S, P2, D2 = mc_scenarios.shape
    if P2 != P:
        raise ValueError(f"mc_scenarios P축({P2}) != forecast P({P}). 제품 reorder가 필요합니다.")
    if D2 < D:
        raise ValueError(f"mc_scenarios D({D2}) < horizons D({D}).")
    if D2 > D:
        mc_scenarios = mc_scenarios[:, :, :D]

    mc_demand_i = np.rint(np.maximum(mc_scenarios, 0.0) * scale).astype(int)  # (S,P,D)

    # UB: 시나리오 누적수요 최댓값 기반(타이트)
    cum_dem_max = np.cumsum(mc_demand_i, axis=2).max(axis=0)  # (P,D)
    STATE_UB = _safe_int_cap(int(cum_dem_max.max() * 2))

    stock_s   = _make_3d_int(model, S, P, D, -STATE_UB, STATE_UB, "stock_s")
    inv_s     = _make_3d_int(model, S, P, D, 0, STATE_UB, "inv_s")
    backlog_s = _make_3d_int(model, S, P, D, 0, STATE_UB, "backlog_s")
    ZERO = model.NewIntVar(0, 0, "ZERO_CVAR")

    for s in range(S):
        for i, p in enumerate(products):
            cid = int(cluster_info.get(p, 1))
            s_stock = int(safety_stock_map.get(cid, 0) * scale)

            for d in range(D):
                if d == 0:
                    init_inv_i = int(round((initial_inventory_map.get(p, initial_inventory) if initial_inventory_map else initial_inventory) * scale))
                    prev_stock = init_inv_i
                else:
                    prev_stock = stock_s[s][i][d-1]

                model.Add(stock_s[s][i][d] == prev_stock + produce[i][d] - mc_demand_i[s, i, d])
                model.AddMaxEquality(inv_s[s][i][d], [stock_s[s][i][d], ZERO])

                neg_stock = model.NewIntVar(-STATE_UB, STATE_UB, f"neg_stock_s{s}_{i}_{d}")
                model.Add(neg_stock == -stock_s[s][i][d])
                model.AddMaxEquality(backlog_s[s][i][d], [neg_stock, ZERO])

                if s_stock > 0:
                    model.Add(inv_s[s][i][d] >= s_stock)

    # ---- Shortage 정의: backlog 증가분(신규 미충족) ----
    SHORT_UB = STATE_UB
    shortage_s = _make_3d_int(model, S, P, D, 0, SHORT_UB, "short_s")

    for s in range(S):
        for i in range(P):
            model.Add(shortage_s[s][i][0] == backlog_s[s][i][0])
            for d in range(1, D):
                inc = model.NewIntVar(-STATE_UB, STATE_UB, f"back_inc_s{s}_{i}_{d}")
                model.Add(inc == backlog_s[s][i][d] - backlog_s[s][i][d-1])
                model.AddMaxEquality(shortage_s[s][i][d], [inc, ZERO])

    # ---- Scenario aggregates ----
    sum_bd = int(STATE_UB * P * D)
    sum_bd = _safe_int_cap(sum_bd, cap=3_000_000_000)

    total_short_s = [model.NewIntVar(0, sum_bd, f"total_short_s{s}") for s in range(S)]
    sum_inv_s     = [model.NewIntVar(0, sum_bd, f"sum_inv_s{s}")     for s in range(S)]

    for s in range(S):
        model.Add(total_short_s[s] == sum(shortage_s[s][i][d] for i in range(P) for d in range(D)))
        model.Add(sum_inv_s[s]     == sum(inv_s[s][i][d]      for i in range(P) for d in range(D)))

    # =====================================================
    # Rate-based loss (evaluator와 1:1 일치)
    # =====================================================
    total_dem_i = mc_demand_i.sum(axis=(1, 2)).astype(int)  # (S,)
    total_dem_i = np.maximum(total_dem_i, 1)

    denom_inv = int(max(P * D * day_cap, 1))

    short_rate_int = [model.NewIntVar(0, int(loss_scale), f"short_rate_int_s{s}") for s in range(S)]
    inv_rate_int   = [model.NewIntVar(0, int(loss_scale), f"inv_rate_int_s{s}")   for s in range(S)]

    for s in range(S):
        model.AddDivisionEquality(short_rate_int[s], total_short_s[s] * int(loss_scale), int(total_dem_i[s]))
        model.AddDivisionEquality(inv_rate_int[s],   sum_inv_s[s]     * int(loss_scale), int(denom_inv))

    WDEN = 1000
    wb_w = max(0, int(round(mc_wb * WDEN)))
    wi_w = max(0, int(round(mc_wi * WDEN)))

    weighted_sum_ub = int((wb_w + wi_w) * loss_scale)
    weighted_sum_ub = max(weighted_sum_ub, 1)

    weighted_sum_s = [model.NewIntVar(0, weighted_sum_ub, f"w_sum_s{s}") for s in range(S)]
    loss_int_s     = [model.NewIntVar(0, int(loss_scale), f"loss_int_s{s}") for s in range(S)]

    for s in range(S):
        model.Add(weighted_sum_s[s] == wb_w * short_rate_int[s] + wi_w * inv_rate_int[s])
        model.AddDivisionEquality(loss_int_s[s], weighted_sum_s[s], WDEN)

    # ---- CVaR linearization on loss_int_s ----
    a = float(cvar_alpha)
    a = min(max(a, 0.0), 0.999999)

    den = int(np.ceil((1.0 - a) * S))
    den = max(den, 1)

    eta = model.NewIntVar(0, int(loss_scale), "eta_var")
    z   = [model.NewIntVar(0, int(loss_scale), f"z_s{s}") for s in range(S)]
    for s in range(S):
        model.Add(z[s] >= loss_int_s[s] - eta)
        model.Add(z[s] >= 0)

    sum_z_ub = int(S * loss_scale)
    sum_z = model.NewIntVar(0, sum_z_ub, "sum_z")
    model.Add(sum_z == sum(z))

    tail_avg = model.NewIntVar(0, int(loss_scale), "tail_avg")
    model.AddDivisionEquality(tail_avg, sum_z, den)

    cvar_avg = model.NewIntVar(0, int(2 * loss_scale), "cvar_avg")
    model.Add(cvar_avg == eta + tail_avg)

    # ---- mean loss ----
    sum_loss = model.NewIntVar(0, int(S * loss_scale), "sum_loss")
    model.Add(sum_loss == sum(loss_int_s))
    mean_loss = model.NewIntVar(0, int(loss_scale), "mean_loss")
    model.AddDivisionEquality(mean_loss, sum_loss, int(S))

    # ---- smooth penalty ----
    base_terms = []
    smooth_w = int(round(lambda_smooth * 100))  # 통일: *100
    if smooth_w > 0:
        for i in range(P):
            for d in range(1, D):
                diff = model.NewIntVar(0, day_cap, f"diff_{i}_{d}")
                model.Add(diff >= produce[i][d] - produce[i][d-1])
                model.Add(diff >= produce[i][d-1] - produce[i][d])
                base_terms.append(smooth_w * diff)

    # ---- objective ----
    lam_int = int(round(float(lambda_cvar) * int(lambda_scale)))
    lam_int = max(lam_int, 0)

    if lam_int > 0 and lam_int * int(2 * loss_scale) > _INT64_MAX:
        raise RuntimeError("objective overflow risk. Try smaller --lambda_scale or --lambda_cvar.")

    model.Minimize(sum(base_terms) + mean_loss + lam_int * cvar_avg)

    solver = _make_solver(solver_seed, max_time, workers, gap, log_progress)
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"OR-Tools: {solver.StatusName(status)} (실행 가능한 계획 실패)")

    demand_mean = mc_scenarios.mean(axis=0)  # (P,D)

    rows = []
    for d_in, hcol in enumerate(horizons):
        for i, p in enumerate(products):
            inv_mean  = float(np.mean([solver.Value(inv_s[s][i][d_in]) for s in range(S)]) / scale)
            back_mean = float(np.mean([solver.Value(backlog_s[s][i][d_in]) for s in range(S)]) / scale)
            sh_mean   = float(np.mean([solver.Value(shortage_s[s][i][d_in]) for s in range(S)]) / scale)
            rows.append({
                "day_idx": int(d_in),
                "horizon": hcol,
                "Product_Number": p,
                "demand": float(demand_mean[i, d_in]),
                "produce": float(solver.Value(produce[i][d_in]) / scale),
                "end_inventory": inv_mean,
                "backlog": back_mean,
                "shortage": sh_mean,
            })

    plan_df = pd.DataFrame(rows)
    diag = {
        "mode": "cvar_shortage_rate",
        "status": solver.StatusName(status),
        "objective": float(solver.ObjectiveValue()),
        "eta": int(solver.Value(eta)),
        "cvar_avg": int(solver.Value(cvar_avg)),
        "mean_loss": int(solver.Value(mean_loss)),
        "den": int(den),
        "S": int(S),
        "lambda_cvar": float(lambda_cvar),
        "lambda_scale": int(lambda_scale),
        "loss_scale": int(loss_scale),
        "solver_seed": int(solver_seed),
        "max_time": float(max_time),
        "gap": float(gap),
        "workers": int(workers),
        "STATE_UB": int(STATE_UB),
        "PROD_UB": int(day_cap),
        "smooth_w": int(smooth_w),
        "wb_w": int(wb_w),
        "wi_w": int(wi_w),
        "note": "rate-based loss: wb*ShortageRate + wi*InventoryRate (matches evaluator)",
    }
    print("[CVaR]", diag)
    return plan_df, diag

# =========================================================
# (D) CLI
# =========================================================

def _load_map_json_or_file(s: Optional[str]) -> Optional[Dict[int, float]]:
    if not s:
        return None
    if s.startswith("@"):
        with open(s[1:], "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = json.loads(s)
    return {int(k): float(v) for k, v in data.items()}

def _run_optimize_with_fallbacks(
    *,
    forecast_df: pd.DataFrame,
    horizons: List[str],
    args: argparse.Namespace,
    cluster_info: Dict[str, int],
    inv_map: Dict[str, float],
    scenarios_re: Optional[np.ndarray],
    hint_plan: Optional[Dict[Tuple[str, int], int]],
    lam: float,
    loss_scale_init: int,
) -> Tuple[pd.DataFrame, Dict[str, Any], int]:
    attempts = []
    attempts.append({"workers": int(args.workers), "max_time": float(args.max_time), "gap": float(args.gap), "loss_scale": int(loss_scale_init)})
    attempts.append({"workers": 1, "max_time": float(args.max_time) * 2.0, "gap": min(0.10, float(args.gap) * 1.5), "loss_scale": int(loss_scale_init)})
    attempts.append({"workers": 1, "max_time": float(args.max_time) * 2.0, "gap": min(0.15, float(args.gap) * 2.0), "loss_scale": max(100, int(loss_scale_init // 2))})

    last_err = None
    for k, cfg in enumerate(attempts, start=1):
        try:
            plan_df, diag = optimize_plan(
                forecast_by_product=forecast_df,
                horizons=horizons,
                prod_col=args.product_col,
                cluster_info=cluster_info,
                daily_capacity=args.daily_capacity,
                lambda_smooth=args.lambda_smooth,
                initial_inventory=args.initial_inventory,
                int_production=args.int_production,
                scale=args.scale,
                min_lot_map=_load_map_json_or_file(args.min_lot_map),
                safety_stock_map=_load_map_json_or_file(args.safety_stock_map),
                weight_map=_load_map_json_or_file(args.weight_map),
                initial_inventory_map=inv_map,
                hint_plan=hint_plan,
                use_cvar_obj=True,
                mc_scenarios=scenarios_re,
                cvar_alpha=args.cvar_alpha,
                lambda_cvar=float(lam),
                lambda_scale=args.lambda_scale,
                mc_wb=args.mc_wb,
                mc_wi=args.mc_wi,
                loss_scale=int(cfg["loss_scale"]),
                solver_seed=args.solver_seed,
                max_time=float(cfg["max_time"]),
                workers=int(cfg["workers"]),
                gap=float(cfg["gap"]),
                log_progress=args.log_progress,
            )
            diag["fallback_attempt"] = k
            diag["workers_used"] = int(cfg["workers"])
            diag["max_time_used"] = float(cfg["max_time"])
            diag["gap_used"] = float(cfg["gap"])
            diag["loss_scale_used"] = int(cfg["loss_scale"])
            return plan_df, diag, int(cfg["loss_scale"])
        except Exception as e:
            last_err = str(e)
            print(f"[FALLBACK-FAIL] attempt={k} workers={cfg['workers']} time={cfg['max_time']} gap={cfg['gap']} loss_scale={cfg['loss_scale']} err={last_err}")
            continue

    raise RuntimeError(last_err or "All fallback attempts failed.")

def main():
    ap = argparse.ArgumentParser(description="CP-SAT 생산계획 (CVaR objective + fallback 포함)")

    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--feat_csv", required=True)
    ap.add_argument("--out_csv", required=True)

    ap.add_argument("--product_col", default="Product_Number")
    ap.add_argument("--horizons", nargs="*", default=None)

    ap.add_argument("--daily_capacity", type=int, default=15000)
    ap.add_argument("--lambda_smooth", type=float, default=0.5)
    ap.add_argument("--initial_inventory", type=float, default=0.0)

    ap.add_argument("--scale", type=int, default=10)

    ap.add_argument("--int_production", action="store_true")
    ap.add_argument("--min_lot_map", type=str, default=None)
    ap.add_argument("--safety_stock_map", type=str, default=None)
    ap.add_argument("--weight_map", type=str, default=None)
    ap.add_argument("--initial_inventory_map", type=str, default=None)

    # Warm-start
    ap.add_argument("--hint_plan_csv", type=str, default=None, help="Warm-start hint plan CSV (Product_Number, day_idx, produce)")

    ap.add_argument("--solver_seed", type=int, default=42)
    ap.add_argument("--max_time", type=float, default=300.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--gap", type=float, default=0.1)
    ap.add_argument("--log_progress", action="store_true")

    ap.add_argument("--use_cvar_obj", action="store_true")
    ap.add_argument("--mc_npz", default=None)
    ap.add_argument("--mc_csv", default=None)

    ap.add_argument("--cvar_alpha", type=float, default=0.9)
    ap.add_argument("--lambda_cvar", type=float, default=0.3)
    ap.add_argument("--lambda_scale", type=int, default=100_000_000)
    ap.add_argument("--mc_wb", type=float, default=1.0)
    ap.add_argument("--mc_wi", type=float, default=0.2)
    ap.add_argument("--loss_scale", type=int, default=100_000)

    args = ap.parse_args()

    pred = pd.read_csv(args.in_csv)
    cluster_info = load_cluster_info(args.feat_csv, prod_col=args.product_col)

    forecast_df = preprocess_forecast(pred, prod_col=args.product_col, dt_col="DateTime")
    horizons = args.horizons or detect_horizons(forecast_df)
    horizons = _sort_horizons_kor(horizons)  # ✅ 최종 안전 정렬

    inv_map_raw = _load_map_json_or_file(args.initial_inventory_map)
    inv_map: Dict[str, float] = inv_map_raw or {}

    hint_plan = None
    if args.hint_plan_csv:
        hint_plan = load_hint_plan_csv(args.hint_plan_csv, scale=args.scale, prod_col=args.product_col)
        print(f"[HINT] loaded: {args.hint_plan_csv} | keys={len(hint_plan)}")

    if not args.use_cvar_obj:
        plan_df, diag = optimize_plan(
            forecast_by_product=forecast_df,
            horizons=horizons,
            prod_col=args.product_col,
            cluster_info=cluster_info,
            daily_capacity=args.daily_capacity,
            lambda_smooth=args.lambda_smooth,
            initial_inventory=args.initial_inventory,
            int_production=args.int_production,
            scale=args.scale,
            min_lot_map=_load_map_json_or_file(args.min_lot_map),
            safety_stock_map=_load_map_json_or_file(args.safety_stock_map),
            weight_map=_load_map_json_or_file(args.weight_map),
            initial_inventory_map=inv_map,
            hint_plan=hint_plan,
            use_cvar_obj=False,
            solver_seed=args.solver_seed,
            max_time=args.max_time,
            workers=args.workers,
            gap=args.gap,
            log_progress=args.log_progress,
        )
        plan_df.to_csv(args.out_csv, index=False, float_format="%.2f")
        print(f"[OK] Saved plan: {args.out_csv} (rows={len(plan_df)})")
        print("[BASIC diag]", diag)
        return

    scenarios, mc_products = load_mc_scenarios(args.mc_npz, args.mc_csv, product_col=args.product_col)
    if scenarios is None or mc_products is None:
        raise ValueError("--use_cvar_obj 사용 시 --mc_npz 또는 --mc_csv 필요")

    forecast_df[args.product_col] = forecast_df[args.product_col].astype(str).str.replace(r"\.0$", "", regex=True)
    forecast_products = forecast_df[args.product_col].tolist()
    scenarios_re = reorder_mc_scenarios_to_forecast(scenarios, mc_products, forecast_products)

    plan_df, diag, used_loss_scale = _run_optimize_with_fallbacks(
        forecast_df=forecast_df,
        horizons=horizons,
        args=args,
        cluster_info=cluster_info,
        inv_map=inv_map,
        scenarios_re=scenarios_re,
        hint_plan=hint_plan,
        lam=float(args.lambda_cvar),
        loss_scale_init=int(args.loss_scale),
    )

    plan_df.to_csv(args.out_csv, index=False, float_format="%.2f")
    print(f"[OK] Saved plan: {args.out_csv} (rows={len(plan_df)})")
    print("[CVaR diag]", diag)

if __name__ == "__main__":
    main()