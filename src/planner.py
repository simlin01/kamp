# src/planner.py
import pandas as pd
import numpy as np

def plan_production(
    forecast_by_product: pd.DataFrame,
    horizons: list,
    prod_col: str,
    daily_capacity: int = 5000,
    min_lot_size: int = 0,
    safety_stock: int = 0,
) -> pd.DataFrame:
    """
    아주 단순한 휴리스틱:
    - 각 일(horizon)마다 제품별 수요 합계를 계산
    - 일일 생산가능량(daily_capacity) 한도 내에서 수요 비율대로 배분
    - min_lot_size가 있으면 해당 단위로 올림
    - safety_stock은 여기선 별도 재고 테이블이 없으므로 배분 총량에서 단순 차감 처리(선택)
    반환: 제품별 각 일자 생산계획 테이블 (forecast_by_product와 같은 스키마)
    """
    df = forecast_by_product.copy()
    df_out = df[[prod_col]].copy()

    for h in horizons:
        demand = df[h].clip(lower=0).astype(float)
        total = demand.sum()

        if total <= 0:
            alloc = np.zeros_like(demand)
        else:
            # 일일 총 생산량 = daily_capacity (safety_stock을 총량에서 차감하고 싶다면 max(0, daily_capacity - safety_stock)로도 가능)
            day_cap = max(0, daily_capacity)
            alloc = day_cap * (demand / total)

        # 최소 로트 반영
        if min_lot_size and min_lot_size > 0:
            alloc = np.ceil(alloc / min_lot_size) * min_lot_size

        df_out[h] = alloc

    return df_out