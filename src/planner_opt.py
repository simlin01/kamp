#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import List, Dict, Optional
import json
import re
import unicodedata
import argparse
import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

# -----------------------------
# (A) 유틸 & 전처리
# -----------------------------

import re

def _sort_horizons_kor(hs: List[str]) -> List[str]:
    """['T일 예상 수주량', 'T+1일 예상 수주량', ...] 순서로 정렬."""
    def key(h: str) -> int:
        s = str(h)
        if s.startswith("T일"):  # 'T일 예상 수주량'
            return 0
        m = re.search(r"T\+(\d+)", s)  # 'T+3일 예상 수주량' 등
        return int(m.group(1)) if m else 10_000
    return sorted(hs, key=key)

WEIRD_SPACES = ["\ufeff", "\u200b", "\u200c", "\u200d", "\xa0"]

def _normalize_col(c: str) -> str:
    c2 = unicodedata.normalize("NFKC", str(c))
    for w in WEIRD_SPACES:
        c2 = c2.replace(w, "")
    c2 = re.sub(r"\s+", " ", c2).strip()
    return c2

def preprocess_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """pred.csv가 시간별 행일 수도, 제품별 스냅샷일 수도 있으므로 안전 스냅샷 처리."""
    df = df.copy()
    df.columns = [_normalize_col(c) for c in df.columns]

    if "Product_Number" not in df.columns:
        raise KeyError(f"'Product_Number' 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}")

    # DateTime이 없으면 그대로 사용(이미 제품별 1행 스냅샷 가정)
    if "DateTime" not in df.columns:
        return df

    # DateTime 파싱 및 최신 스냅샷 선택
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    valid = df.dropna(subset=["DateTime"])
    if valid.empty:
        return df.drop(columns=["DateTime"], errors="ignore")

    latest_dt = (
        valid.groupby("Product_Number", as_index=False)["DateTime"].max()
              .rename(columns={"DateTime": "_LatestDT"})
    )
    merged = df.merge(latest_dt, on="Product_Number", how="inner")
    picked = merged[merged["DateTime"] == merged["_LatestDT"]].copy()

    num_cols = picked.select_dtypes(include="number").columns.tolist()
    non_num_cols = [c for c in picked.columns if c not in num_cols]
    agg = {**{c: "first" for c in non_num_cols}, **{c: "mean" for c in num_cols}}
    snapped = picked.groupby("Product_Number", as_index=False).agg(agg)
    return snapped.drop(columns=["_LatestDT", "DateTime"], errors="ignore")

def load_cluster_info(feat_file: str) -> Dict[str, int]:
    df = pd.read_csv(feat_file)
    cols = [_normalize_col(c) for c in df.columns]
    df.columns = cols
    if "Product_Number" not in df.columns or "Cluster" not in df.columns:
        raise ValueError("feat.csv에는 'Product_Number', 'Cluster' 컬럼이 필요합니다.")
    m = df[["Product_Number", "Cluster"]].drop_duplicates()
    return m.set_index("Product_Number")["Cluster"].to_dict()

def detect_horizons(df: pd.DataFrame) -> List[str]:
    """T, T+1, ... 혹은 'T일 예상 수주량' 같은 열명을 자동 감지."""
    candidates = []
    for c in df.columns:
        cc = _normalize_col(c)
        if re.fullmatch(r"T(\+\d+)?", cc):
            candidates.append(c)
        elif ("T" in cc and "예상" in cc) or ("T" in cc and "수주" in cc):
            candidates.append(c)
    # 정렬: T, T+1, T+2 ...
    def _key(x: str) -> int:
        s = _normalize_col(x)
        if s == "T": return 0
        m = re.match(r"T\+(\d+)", s)
        if m: return int(m.group(1))
        # 한국어 케이스: "T+1일 예상 수주량" 등
        m2 = re.search(r"T\+(\d+)", s)
        return int(m2.group(1)) if m2 else 10_000
    candidates = sorted(set(candidates), key=_key)
    if not candidates:
        raise ValueError("horizons 자동 감지 실패. --horizons 로 명시해 주세요.")
    return candidates

# -----------------------------
# (B) CP-SAT 변수 헬퍼
# -----------------------------
def _make_2d_int(model: cp_model.CpModel, P: int, D: int, lb: int, ub: int, name: str):
    return [[model.NewIntVar(lb, ub, f"{name}_{i}_{d}") for d in range(D)] for i in range(P)]

def _make_2d_bool(model: cp_model.CpModel, P: int, D: int, name: str):
    return [[model.NewBoolVar(f"{name}_{i}_{d}") for d in range(D)] for i in range(P)]

# -----------------------------
# (C) 최적화 코어
# -----------------------------
def optimize_plan(
    forecast_by_product: pd.DataFrame,
    horizons: List[str],
    prod_col: str,
    cluster_info: Dict[str, int],
    daily_capacity: int = 10000,
    lambda_smooth: float = 1.0,
    initial_inventory: float = 0.0,
    int_production: bool = True,   # CP-SAT은 정수, 인터페이스 일치 목적
    scale: int = 10,
    # 정책 주입(옵션)
    min_lot_map: Optional[Dict[int, float]] = None,
    safety_stock_map: Optional[Dict[int, float]] = None,
    weight_map: Optional[Dict[int, float]] = None,
) -> pd.DataFrame:
    
    def _diag_horizons(df: pd.DataFrame, horizons: list, prod_col: str, tag: str="[DIAG]"):
        # 1) 컬럼 정규화 흔적 확인
        print(f"{tag} columns(sample 10):", list(df.columns)[:10])

        # 2) 누락/추가된 horizon 탐지
        miss = [h for h in horizons if h not in df.columns]
        extra = [c for c in df.columns if c not in ([prod_col] + horizons)]
        print(f"{tag} missing_horizons:", miss)
        if miss:
            # 어떤 유니코드/공백 문제인지 가까운 후보를 보여줌
            import difflib
            for h in miss:
                near = difflib.get_close_matches(h, [str(c) for c in df.columns], n=3, cutoff=0.6)
                print(f"{tag}  -> near matches for '{h}': {near}")

        # 3) 데이터 타입/NaN 비율
        sub = df[[prod_col] + [h for h in horizons if h in df.columns]].copy()
        num = sub.select_dtypes(include="number").columns.tolist()
        print(f"{tag} numeric_cols in horizons:", [c for c in horizons if c in num])
        print(f"{tag} NaN ratio per horizon:", sub[horizons].isna().mean().round(4).to_dict())

        # 4) 첫 몇 행 프린트
        print(f"{tag} head:")
        print(sub.head(3))
    
    """클러스터 정책을 인자로 주입 가능한 CP-SAT 생산계획 최적화."""
    min_lot_map = min_lot_map or {0: 100, 1: 50, 2: 0, 3: 200}
    safety_stock_map = safety_stock_map or {0: 0, 1: 0, 2: 0, 3: 0}
    weight_map = weight_map or {0: 5.0, 1: 2.0, 2: 0.5, 3: 1.0}

    df = preprocess_forecast(forecast_by_product)
    model = cp_model.CpModel()

    # ----- 데이터 준비 -----
    products = df[prod_col].tolist()
    P, D = len(products), len(horizons)

    demand_f = df[horizons].to_numpy(dtype=float)
    demand_i = np.rint(demand_f * scale).astype(int)
    init_inv_i = int(round(initial_inventory * scale))

    BIG = int((daily_capacity * scale) * D * 2)
    day_cap = daily_capacity * scale

    # ----- 변수 -----
    produce = _make_2d_int(model, P, D, 0, BIG, "produce")   # 생산량
    inv     = _make_2d_int(model, P, D, 0, BIG, "inv")       # 종료 재고
    backlog = _make_2d_int(model, P, D, 0, BIG, "backlog")   # 종료 백로그
    is_prod = _make_2d_bool(model, P, D, "is_prod")          # 생산 여부

    # ----- 제약 -----
    for i, p in enumerate(products):
        cid = int(cluster_info.get(p, 1))  # 키 없을 때 기본 클러스터 1
        min_lot = int(min_lot_map.get(cid, 0) * scale)
        s_stock = int(safety_stock_map.get(cid, 0) * scale)

        for d in range(D):
            prev_inv = inv[i][d-1] if d > 0 else init_inv_i
            # 재고 흐름 (완화형): prev_inv + produce - demand == inv - backlog
            model.Add(prev_inv + produce[i][d] - demand_i[i, d] == inv[i][d] - backlog[i][d])

            # 재고/백로그 동시양수 방지 (논리)
            inv_pos  = model.NewBoolVar(f"inv_pos_{i}_{d}")
            back_pos = model.NewBoolVar(f"back_pos_{i}_{d}")
            model.Add(inv[i][d] >= 1).OnlyEnforceIf(inv_pos)
            model.Add(inv[i][d] <= 0).OnlyEnforceIf(inv_pos.Not())
            model.Add(backlog[i][d] >= 1).OnlyEnforceIf(back_pos)
            model.Add(backlog[i][d] <= 0).OnlyEnforceIf(back_pos.Not())
            model.AddBoolOr([inv_pos.Not(), back_pos.Not()])

            # 안전재고 (d>0일 때만 강제)
            if s_stock > 0 and d > 0:
                model.Add(inv[i][d] >= s_stock).OnlyEnforceIf(back_pos.Not())

            # 최소 로트 / 생산여부
            if min_lot > 0:
                model.Add(produce[i][d] >= min_lot).OnlyEnforceIf(is_prod[i][d])
                model.Add(produce[i][d] <= demand_i[i, d] + day_cap).OnlyEnforceIf(is_prod[i][d])
                model.Add(produce[i][d] == 0).OnlyEnforceIf(is_prod[i][d].Not())
            else:
                model.Add(produce[i][d] <= day_cap * is_prod[i][d])

    # 일일 CAPA
    for d in range(D):
        model.Add(sum(produce[i][d] for i in range(P)) <= day_cap)

    # ----- 목적함수 -----
    terms = []
    # (1) 백로그 최소화 (클러스터 가중치 반영)
    for i, p in enumerate(products):
        cid = int(cluster_info.get(p, 1))
        w = int(round(weight_map.get(cid, 1.0) * 100))
        for d in range(D):
            terms.append(w * backlog[i][d])

    # (2) 생산변동 완화 |produce_d - produce_{d-1}|
    lam = int(round(lambda_smooth * 1))
    if lam > 0:
        for i in range(P):
            for d in range(1, D):
                diff = model.NewIntVar(0, BIG, f"diff_{i}_{d}")
                model.Add(diff >= produce[i][d] - produce[i][d-1])
                model.Add(diff >= produce[i][d-1] - produce[i][d])
                terms.append(lam * diff)

    model.Minimize(sum(terms))

    # ----- 풀기 -----
    solver = cp_model.CpSolver()
    solver.parameters.relative_gap_limit = 0.02
    solver.parameters.max_time_in_seconds = 300.0
    solver.parameters.num_search_workers = 8
    solver.parameters.log_search_progress = True
    solver.parameters.random_seed = 42

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"OR-Tools: {solver.StatusName(status)} (실행 가능한 계획 실패)")

    # ----- 결과 복원 -----
    disp_horizons = _sort_horizons_kor(horizons)
    idx_of = {h: i for i, h in enumerate(horizons)}  # solver 내부 인덱스 매핑

    rows = []
    for d_out, hcol in enumerate(disp_horizons):
        d_in = idx_of[hcol]  # 내부 인덱스
        for i, p in enumerate(products):
            demand_val = demand_i[i, d_in] / scale
            produce_val = solver.Value(produce[i][d_in]) / scale
            inv_val = solver.Value(inv[i][d_in]) / scale
            backlog_val = solver.Value(backlog[i][d_in]) / scale

            rows.append({
                "day_idx": d_out,            # 0..4
                "horizon": hcol,             # 'T일 예상 수주량' ~ 'T+4일 예상 수주량'
                prod_col: p,
                # 소수점 둘째자리 고정(.2f)
                "demand":        f"{demand_val:.2f}",
                "produce":       f"{produce_val:.2f}",
                "end_inventory": f"{inv_val:.2f}",
                "backlog":       f"{backlog_val:.2f}",
            })

    return pd.DataFrame(rows)

# -----------------------------
# (D) CLI (plan_from_csv 통합)
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="CP-SAT 생산계획 (단일 파일: 코어+CLI)")
    ap.add_argument("--in_csv", required=True, help="예측 수요 CSV (pred.csv)")
    ap.add_argument("--feat_csv", required=True, help="클러스터 매핑 CSV (feat.csv)")
    ap.add_argument("--out_csv", required=True, help="출력 파일 (production_plan.csv)")
    ap.add_argument("--product_col", default="Product_Number")
    ap.add_argument("--horizons", nargs="*", default=None, help='예: T "T+1" "T+2" ... (미지정 시 자동 감지)')
    ap.add_argument("--daily_capacity", type=int, default=10000)
    ap.add_argument("--lambda_smooth", type=float, default=1.0)
    ap.add_argument("--initial_inventory", type=float, default=0.0)
    ap.add_argument("--scale", type=int, default=10)
    ap.add_argument("--int_production", action="store_true")
    # 정책 맵 주입(JSON 문자열 또는 파일 경로)
    ap.add_argument("--min_lot_map", type=str, default=None, help='JSON 또는 @file.json')
    ap.add_argument("--safety_stock_map", type=str, default=None)
    ap.add_argument("--weight_map", type=str, default=None)
    args = ap.parse_args()

    def _load_map(s: Optional[str]) -> Optional[Dict[int, float]]:
        if not s: return None
        if s.startswith("@"):
            with open(s[1:], "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(s)
        # 키를 int로 정규화
        return {int(k): float(v) for k, v in data.items()}

    pred = pd.read_csv(args.in_csv)
    feat = args.feat_csv
    cluster_info = load_cluster_info(feat)

    forecast_df = preprocess_forecast(pred)
    horizons = args.horizons or detect_horizons(forecast_df)

    plan_df = optimize_plan(
        forecast_by_product=forecast_df,
        horizons=horizons,
        prod_col=args.product_col,
        cluster_info=cluster_info,
        daily_capacity=args.daily_capacity,
        lambda_smooth=args.lambda_smooth,
        initial_inventory=args.initial_inventory,
        int_production=args.int_production,
        scale=args.scale,
        min_lot_map=_load_map(args.min_lot_map),
        safety_stock_map=_load_map(args.safety_stock_map),
        weight_map=_load_map(args.weight_map),
    )
    plan_df.to_csv(args.out_csv, index=False)
    print(f"[OK] Saved plan: {args.out_csv} (rows={len(plan_df)})")

if __name__ == "__main__":
    main()
