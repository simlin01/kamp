# src/planner_opt.py
'''
목적함수: 누적 미충족 수요(backlog) 최소화 + (선택) 생산변동 완화
제약: 일일 총 CAPA, 재고흐름 균형식, 비음수/정수 제약, 최소 로트/안전재고
'''
from typing import List, Dict
import pandas as pd
from ortools.sat.python import cp_model

def optimize_plan(
    forecast_by_product: pd.DataFrame,
    horizons: List[str],
    prod_col: str,
    daily_capacity: int = 5000,
    min_lot_size: int = 0,
    safety_stock: int = 0,
    int_production: bool = False
) -> pd.DataFrame:
    """
    변수:
      produce[p,d]  >= 0  (정수/실수 선택)
      inv[p,d]      >= 0
      backlog[p,d]  >= 0   (수요 미충족)
    흐름:
      inv[p,d] = inv[p,d-1] + produce[p,d] - demand[p,d] + (backlog[p,d-1] - backlog[p,d])
    단순화를 위해 아래와 같이 재구성:
      inv_prev + produce - demand + backlog_prev = inv + backlog
    용량:
      Σ_p produce[p,d] ≤ daily_capacity
    최소로트:
      produce[p,d] == 0 or ≥ min_lot_size  (indicator로 근사)
    안전재고:
      inv[p,d] ≥ safety_stock  (원하면 강제, 여기선 inv에서 safety_stock을 유지하도록 구성)
    목적:
      Σ_{p,d} backlog[p,d] + λ * Σ_{p,d>0} |produce[p,d] - produce[p,d-1]|
    """

    products = forecast_by_product[prod_col].tolist()
    P = len(products)
    D = len(horizons)

    # demand dict
    demand = { (i,d): float(forecast_by_product.loc[i, horizons[d]]) for i in range(P) for d in range(D) }

    model = cp_model.CpModel()

    # Domains
    BIG = 10**7
    # 변수 정의
    produce = {}
    inv = {}
    backlog = {}
    is_prod = {}  # lot size indicator

    for i in range(P):
        for d in range(D):
            if int_production:
                produce[i,d] = model.NewIntVar(0, BIG, f"produce_{i}_{d}")
            else:
                produce[i,d] = model.NewIntVar(0, BIG, f"produce_{i}_{d}")  # CP-SAT는 정수만 지원

            inv[i,d] = model.NewIntVar(0, BIG, f"inv_{i}_{d}")
            backlog[i,d] = model.NewIntVar(0, BIG, f"backlog_{i}_{d}")
            if min_lot_size > 0:
                is_prod[i,d] = model.NewBoolVar(f"is_prod_{i}_{d}")

    # 초기조건: inv[-1]=0, backlog[-1]=0 가정
    for i in range(P):
        # d=0 흐름식: inv0 + backlog0 = produce0 - demand0 + 0 + 0  → 재배열
        # produce0 - demand0 = inv0 + backlog0
        lhs = produce[i,0] - int(demand[i,0])
        model.Add(lhs == inv[i,0] + backlog[i,0])

        # Lot size
        if min_lot_size > 0:
            model.Add(produce[i,0] == 0).OnlyEnforceIf(is_prod[i,0].Not())
            model.Add(produce[i,0] >= min_lot_size).OnlyEnforceIf(is_prod[i,0])

        # 안전재고
        if safety_stock > 0:
            model.Add(inv[i,0] >= safety_stock)

    for i in range(P):
        for d in range(1, D):
            # 흐름: inv_prev + produce - demand + backlog_prev = inv + backlog
            model.Add(inv[i,d-1] + produce[i,d] - int(demand[i,d]) + backlog[i,d-1] == inv[i,d] + backlog[i,d])

            if min_lot_size > 0:
                model.Add(produce[i,d] == 0).OnlyEnforceIf(is_prod[i,d].Not())
                model.Add(produce[i,d] >= min_lot_size).OnlyEnforceIf(is_prod[i,d])

            if safety_stock > 0:
                model.Add(inv[i,d] >= safety_stock)

    # 일일 CAPA
    for d in range(D):
        model.Add(sum(produce[i,d] for i in range(P)) <= daily_capacity)

    # 목적함수: backlog 최소 + 생산변동 완화(선택 λ=1 가중, 단순화 위해 L1 근사)
    terms = []
    for i in range(P):
        for d in range(D):
            terms.append(backlog[i,d])

    # 생산변동 완화 (|p_d - p_{d-1}| 를 새 변수로 근사)
    lambda_smooth = 1
    for i in range(P):
        for d in range(1, D):
            diff = model.NewIntVar(0, BIG, f"diff_{i}_{d}")
            model.Add(diff >= produce[i,d] - produce[i,d-1])
            model.Add(diff >= produce[i,d-1] - produce[i,d])
            terms.append(lambda_smooth * diff)

    model.Minimize(sum(terms))

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0
    solver.parameters.num_search_workers = 8
    result = solver.Solve(model)
    if result not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible plan found by OR-Tools.")

    # Build plan dataframe
    rows = []
    for d in range(D):
        for i,p in enumerate(products):
            rows.append({
                "day_idx": d,
                "horizon": horizons[d],
                prod_col: p,
                "demand": demand[i,d],
                "produce": solver.Value(produce[i,d]),
                "end_inventory": solver.Value(inv[i,d]),
                "backlog": solver.Value(backlog[i,d]),
            })
    return pd.DataFrame(rows)
