# src/planner_opt.py
from __future__ import annotations
from typing import List, Dict
import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

# ---- 0) 클러스터 정책 맵 (키 미존재 시 안전한 기본값 사용) ----
MIN_LOT_MAP = {0: 100, 1: 50, 2: 0, 3: 200}      # 최소 로트
# MIN_LOT_MAP      = {0:0, 1:0, 2:0, 3:0}

SAFETY_STOCK_MAP = {0: 150, 1: 50, 2: 0, 3: 0}   # 안전재고
# SAFETY_STOCK_MAP = {0: 0, 1: 0, 2: 0, 3: 0} 
WEIGHT_MAP = {0: 5.0, 1: 2.0, 2: 0.5, 3: 1.0}    # 백로그 페널티

def load_cluster_info(feat_file: str) -> Dict[str, int]:
    """feat.csv에서 Product_Number→Cluster 매핑을 읽음."""
    df = pd.read_csv(feat_file)
    if "Product_Number" not in df.columns or "Cluster" not in df.columns:
        raise ValueError("feat.csv에는 'Product_Number', 'Cluster' 컬럼이 필요합니다.")
    m = df[["Product_Number", "Cluster"]].drop_duplicates()
    return m.set_index("Product_Number")["Cluster"].to_dict()

def _make_2d_int(model: cp_model.CpModel, P: int, D: int, lb: int, ub: int, name: str):
    return [[model.NewIntVar(lb, ub, f"{name}_{i}_{d}") for d in range(D)] for i in range(P)]

def _make_2d_bool(model: cp_model.CpModel, P: int, D: int, name: str):
    return [[model.NewBoolVar(f"{name}_{i}_{d}") for d in range(D)] for i in range(P)]

import unicodedata, re

WEIRD_SPACES = ["\ufeff", "\u200b", "\u200c", "\u200d", "\xa0"]

def _normalize_col(c: str) -> str:
    c2 = unicodedata.normalize("NFKC", str(c))
    for w in WEIRD_SPACES:
        c2 = c2.replace(w, "")
    c2 = re.sub(r"\s+", " ", c2).strip()
    return c2

def preprocess_forecast(df: pd.DataFrame) -> pd.DataFrame:
    # 0) 컬럼명 정규화
    df = df.copy()
    df.columns = [_normalize_col(c) for c in df.columns]

    # 1) 안전한 존재성 체크 + 친숙한 별칭 매핑
    if "Product_Number" not in df.columns:
        raise KeyError(f"'Product_Number' 컬럼이 없습니다. 현재 컬럼: {[repr(c) for c in df.columns]}")
    if "DateTime" not in df.columns:
        print("[INFO] DateTime 없음 — 평균/단일 스냅샷으로 간주 (pred_by_product 가능)")
        return df

    # 2) DateTime 파싱
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    valid = df.dropna(subset=["DateTime"])
    if valid.empty:
        print("[WARN] DateTime이 모두 NaT → DateTime 제거 후 그대로 사용")
        return df.drop(columns=["DateTime"], errors="ignore")

    # 3) 제품별 최신 DateTime 계산 후 매칭
    latest_dt_map = (
        valid.groupby("Product_Number", as_index=False)["DateTime"]
             .max()
             .rename(columns={"DateTime": "_LatestDT"})
    )
    merged = df.merge(latest_dt_map, on="Product_Number", how="inner")
    picked = merged[merged["DateTime"] == merged["_LatestDT"]].copy()

    # 4) 동일 제품·동일 최신시점 다건 → 수치 mean, 비수치 first
    num_cols = picked.select_dtypes(include="number").columns.tolist()
    non_num_cols = [c for c in picked.columns if c not in num_cols]
    agg = {**{c: "first" for c in non_num_cols}, **{c: "mean" for c in num_cols}}
    snapped = picked.groupby("Product_Number", as_index=False).agg(agg)

    # 5) 보조 컬럼 제거
    snapped = snapped.drop(columns=["_LatestDT", "DateTime"], errors="ignore")
    print(f"[INFO] 제품별 최신 DateTime 스냅샷 사용 (rows={len(snapped)})")
    return snapped

def optimize_plan(
    forecast_by_product: pd.DataFrame,
    horizons: List[str],
    prod_col: str,
    cluster_info: Dict[str, int],
    daily_capacity: int = 10000,
    lambda_smooth: float = 1.0,
    initial_inventory: int = 0,
    int_production: bool = True,   # CP-SAT은 정수만 지원하지만 인터페이스 유지
    scale: int = 10               # 소수 수요를 정수화할 스케일(예: 2자리 → 100)
) -> pd.DataFrame:

    df = preprocess_forecast(forecast_by_product)
    # 1) 정규화 맵: "정규화된 이름" -> "실제 df 컬럼명"
    norm_map = { _normalize_col(c): c for c in df.columns }

    # 2) prod_col 정규화 후 실제 컬럼명으로 치환
    norm_prod = _normalize_col(prod_col)
    if norm_prod not in norm_map:
        raise KeyError(
            f"'{prod_col}' 컬럼을 찾을 수 없습니다. "
            f"정규화된 컬럼 후보: {list(norm_map.keys())} / 실제 컬럼: {list(df.columns)}"
        )
    prod_col = norm_map[norm_prod]

    # 3) horizons도 동일 처리
    fixed_horizons = []
    for h in horizons:
        nh = _normalize_col(h)
        if nh not in norm_map:
            raise KeyError(
                f"수요 열 '{h}'(정규화 '{nh}')을 찾을 수 없습니다. "
                f"정규화된 컬럼 후보: {list(norm_map.keys())}"
            )
        fixed_horizons.append(norm_map[nh])
    horizons = fixed_horizons
    model = cp_model.CpModel()

    # ----- 1) 데이터 준비 (정수 스케일링) -----
    products = df[prod_col].tolist()
    P, D = len(products), len(horizons)

    # demand float → int
    demand_f = df[horizons].to_numpy(dtype=float)
    demand_i = np.rint(demand_f * scale).astype(int)  # 반올림 정수화
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
        cid = cluster_info.get(p, 1)
        min_lot = int(MIN_LOT_MAP.get(cid, 0) * scale)
        s_stock = int(SAFETY_STOCK_MAP.get(cid, 0) * scale)

        for d in range(D):
            prev_inv  = inv[i][d-1] if d > 0 else init_inv_i
            prev_back = backlog[i][d-1] if d > 0 else 0

            # (1) 재고 흐름식 교정
            model.Add((prev_inv - prev_back) + produce[i][d] - demand_i[i, d] == inv[i][d] - backlog[i][d])

            # (2) 안전재고 (원하면 단순식으로)
            if s_stock > 0:
                model.Add(inv[i][d] >= s_stock)

            # (3) 최소로트/생산여부
            if min_lot > 0:
                model.Add(produce[i][d] >= min_lot).OnlyEnforceIf(is_prod[i][d])
                model.Add(produce[i][d] == 0).OnlyEnforceIf(is_prod[i][d].Not())
                # 상한은 CAPA로 충분하면 아래 한 줄만으로도 됨
                model.Add(produce[i][d] <= day_cap)
            else:
                model.Add(produce[i][d] <= day_cap * is_prod[i][d])

    # (4) 일일 CAPA
    for d in range(D):
        model.Add(sum(produce[i][d] for i in range(P)) <= day_cap)


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
    solver.parameters.relative_gap_limit = 0.02  # 2% 이내면 OK
    solver.parameters.max_time_in_seconds = 300.0

    solver.parameters.log_search_progress = False   # 진행 로그 끔
    solver.parameters.log_to_stdout = True         # 요약 로그도 끔

    # 3) 병렬/로그
    solver.parameters.num_search_workers = 8       # 코어 수에 맞게
    solver.parameters.log_search_progress = True
    solver.parameters.random_seed = 42

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
