import os
import pandas as pd
import numpy as np
from typing import List, Tuple

def load_data(path: str, encoding: str = "utf-8") -> pd.DataFrame:
    return pd.read_csv(path, encoding=encoding)

def detect_columns(df: pd.DataFrame) -> Tuple[str, List[str], List[str], List[str]]:
    """
    Returns:
      prod_col        : 제품 컬럼명
      target_cols     : 예측 대상 컬럼명 리스트 (T일 예정 수주량, T+1일 ... 순서 정렬)
      last_year_cols  : 전년도 예정 수주량 컬럼명 리스트
      numeric_covars  : 추가 수치형 피처 컬럼명 리스트
    """
    # 제품컬럼 후보
    prod_col = None
    for cand in ["Product_Number", "product", "SKU", "품번"]:
        if cand in df.columns:
            prod_col = cand
            break
    if prod_col is None:
        prod_col = df.columns[0]  # fallback

    # 타깃 컬럼
    target_cols = [c for c in df.columns if "예정 수주량" in c and ("T일" in c or "T+" in c)]
    def _h_idx(c: str) -> int:
        if "T일" in c:
            return 0
        return int(c.split("T+")[1].split("일")[0])
    target_cols = sorted(target_cols, key=_h_idx)

    # 전년도 컬럼
    last_year_cols = [c for c in df.columns if "작년" in c and "예정 수주량" in c]

    # 기타 수치형 피처
    exclude = set([prod_col] + target_cols + last_year_cols)
    numeric_covars = []
    for c in df.columns:
        if c in exclude:
            continue
        if np.issubdtype(df[c].dtype, np.number):
            numeric_covars.append(c)

    return prod_col, target_cols, last_year_cols, numeric_covars

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
