# src/planner_opt.py
from __future__ import annotations
from typing import List, Dict
import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

# ---- 0) 클러스터 정책 맵 (키 미존재 시 안전한 기본값 사용) ----
MIN_LOT_MAP = {0: 100, 1: 50, 2: 0, 3: 200}      # 최소 로트
# SAFETY_STOCK_MAP = {0: 150, 1: 50, 2: 0, 3: 0}   # 안전재고
SAFETY_STOCK_MAP = {0: 0, 1: 0, 2: 0, 3: 0} 
WEIGHT_MAP = {0: 5.0, 1: 2.0, 2: 0.5, 3: 1.0}    # 백로그 페널티

def load_cluster_info(feat_file: str) -> Dict[str, int]:
    df = pd.read_csv(feat_file)
    if "Product_Number" not in df.columns or "Cluster" not in df.columns:
        raise ValueError("feat.csv에는 'Product_Number', 'Cluster' 컬럼이 필요합니다.")
    m = df[["Product_Number", "Cluster"]].drop_duplicates()
    return m.set_index("Product_Number")["Cluster"].to_dict()

def _make_2d_int(model: cp_model.CpModel, P: int, D: int, lb: int, ub: int, name: str):
    return [[model.NewIntVar(lb, ub, f"{name}_{i}_{d}") for d in range(D)] for i in range(P)]

def _make_2d_bool(model: cp_model.CpModel, P: int, D: int, name: str):
    return [[model.NewBoolVar(f"{name}_{i}_{d}") for d in range(D)] for i in range(P)]

def optimize_plan(
    forecast_by_product: pd.DataFrame,
    horizons: List[str],
    prod_col: str,
    cluster_info: Dict[str, int],
    daily_capacity: int = 10000,
    lambda_smooth: float = 1.0,
    initial_inventory: int = 0,
    int_production: bool = True,   
    scale: int = 10               # 소수 수요를 정수화할 스케일(예: 2자리 → 100)
) -> pd.DataFrame:

    model = cp_model.CpModel()

    # ----- 1) 데이터 준비 (정수 스케일링) -----
    df = forecast_by_product.copy()
    products = df[prod_col].tolist()
    P, D = len(products), len(horizons)

    # demand float → int
    demand_f = df[horizons].to_numpy(dtype=float)
    demand_i = np.rint(demand_f * scale).astype(int) 
    init_inv_i = int(round(initial_inventory * scale))

    # 상계 (생산 총량 여유치)
    BIG = int((daily_capacity * scale) * D * 2)
    day_cap = daily_capacity * scale

    # ----- 2) 변수 -----
    produce = _make_2d_int(model, P, D, 0, BIG, "produce")   # 생산량
    inv     = _make_2d_int(model, P, D, 0, BIG, "inv")       # 종료 재고
    backlog = _make_2d_int(model, P, D, 0, BIG, "backlog")   # 종료 백로그
    is_prod = _make_2d_bool(model, P, D, "is_prod")          # 생산 여부
    # has_backlog = _make_2d_bool(model, P, D, "has_backlog")  # 백로그 존재 여부

    # ----- 3) 제약 -----
    # 3-1) 재고 흐름 + 상호배타 + 안전재고/최소로트(클러스터별)
    for i, p in enumerate(products):
        cid = cluster_info.get(p, 1)  # 없으면 1로 처리
        min_lot = int(MIN_LOT_MAP.get(cid, 0) * scale)
        s_stock = int(SAFETY_STOCK_MAP.get(cid, 0) * scale)

        for d in range(D):
            prev_inv = inv[i][d-1] if d > 0 else init_inv_i
            prev_back = backlog[i][d-1] if d > 0 else 0

            # 재고 흐름: prev_inv + produce + prev_back - demand = inv + backlog
            # --- 재고 흐름 (완화형) ---
            model.Add(prev_inv + produce[i][d] - demand_i[i, d] == inv[i][d] - backlog[i][d])

            # --- 재고/백로그 상호배타 ---
            inv_pos   = model.NewBoolVar(f"inv_pos_{i}_{d}")      # inv >= 1 ?
            back_pos  = model.NewBoolVar(f"back_pos_{i}_{d}")     # backlog >= 1 ?

            # inv >= 1  <=> inv_pos = 1  (반대방향은 relax로 충분)
            model.Add(inv[i][d] >= 1).OnlyEnforceIf(inv_pos)
            model.Add(inv[i][d] <= 0).OnlyEnforceIf(inv_pos.Not())

            # backlog >= 1  <=> back_pos = 1
            model.Add(backlog[i][d] >= 1).OnlyEnforceIf(back_pos)
            model.Add(backlog[i][d] <= 0).OnlyEnforceIf(back_pos.Not())

            # 동시에 양수 금지:  (inv_pos & back_pos) 금지
            model.AddBoolOr([inv_pos.Not(), back_pos.Not()])

            # --- 안전재고 (d>1만 적용) ---
            if s_stock > 0 and d > 0:
                model.Add(inv[i][d] >= s_stock).OnlyEnforceIf(back_pos.Not())

            # --- 최소 로트 ---
            if min_lot > 0:
                model.Add(produce[i][d] >= min_lot).OnlyEnforceIf(is_prod[i][d])
                model.Add(produce[i][d] <= demand_i[i, d] + day_cap).OnlyEnforceIf(is_prod[i][d])
                model.Add(produce[i][d] == 0).OnlyEnforceIf(is_prod[i][d].Not())
            else:
                model.Add(produce[i][d] <= day_cap * is_prod[i][d])


    # 3-2) 일일 총 CAPA (스케일 적용)
    day_cap = daily_capacity * scale
    for d in range(D):
        model.Add(sum(produce[i][d] for i in range(P)) <= day_cap)

    # ----- 4) 목적함수 -----
    terms = []

    # (1) 백로그 최소화 (클러스터별 가중치)  → 정수화 위해 ×100
    for i, p in enumerate(products):
        cid = cluster_info.get(p, 1)
        w = int(round(WEIGHT_MAP.get(cid, 1.0) * 100))
        for d in range(D):
            terms.append(w * backlog[i][d])

    # (2) 생산변동 완화 |produce_d - produce_{d-1}|
    lam = int(round(lambda_smooth * 1))  # 정수 계수
    if lam > 0:
        for i in range(P):
            for d in range(1, D):
                diff = model.NewIntVar(0, BIG, f"diff_{i}_{d}")
                model.Add(diff >= produce[i][d] - produce[i][d-1])
                model.Add(diff >= produce[i][d-1] - produce[i][d])
                terms.append(lam * diff)

    model.Minimize(sum(terms))

    # ----- 5) 풀기 -----
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"OR-Tools: {solver.StatusName(status)} (실행 가능한 계획 실패)")

    # ----- 6) 결과 복원(스케일 되돌리기) -----
    rows = []
    for d, hcol in enumerate(horizons):
        for i, p in enumerate(products):
            rows.append({
                "day_idx": d,
                "horizon": hcol,
                prod_col: p,
                "demand": demand_i[i, d] / scale,
                "produce": solver.Value(produce[i][d]) / scale,
                "end_inventory": solver.Value(inv[i][d]) / scale,
                "backlog": solver.Value(backlog[i][d]) / scale,
            })
    return pd.DataFrame(rows)
