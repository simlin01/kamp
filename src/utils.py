# utils.py
# -*- coding: utf-8 -*-
"""
utils.py — 공통 유틸(로드/저장 + 컬럼 자동탐지)

[핵심 수정]
1) "예정 수주량"만 보던 로직을 "예상/예정" 모두 지원하도록 확장
2) horizon 탐지를 정규식 기반으로 강건화:
   - "T일 예상 수주량", "T+3일 예정 수주량" 등 다양한 공백/표기 대응
   - 정렬 안정화(0,1,2,3,4...)
3) last_year(작년) 컬럼도 "예상/예정" 모두 지원
4) numeric_covars 추출 시, datetime/object에서 숫자로 읽힌 케이스/누락 케이스를 안전 처리
"""

from __future__ import annotations

import os
import re
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

# -------------------------
# IO
# -------------------------
def load_data(path: str, encoding: str = "utf-8") -> pd.DataFrame:
    return pd.read_csv(path, encoding=encoding)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_csv(df: pd.DataFrame, path: str, encoding: str = "utf-8-sig") -> None:
    df.to_csv(path, index=False, encoding=encoding)


# -------------------------
# Column detection
# -------------------------
_TARGET_KEYS = ("예정 수주량", "예상 수주량")

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def _extract_horizon_index(col: str) -> Optional[int]:
    """
    col에서 horizon index를 추출:
      - "T일 ..." -> 0
      - "T+3일 ..." -> 3
      - 축약형 "T" -> 0
      - 축약형 "T+3" -> 3
    실패 시 None.
    """
    c = _norm(col)

    # 축약형
    if re.fullmatch(r"T", c):
        return 0
    m = re.fullmatch(r"T\+(\d+)", c)
    if m:
        return int(m.group(1))

    # 한국어 형태
    if "T일" in c:
        return 0
    m2 = re.search(r"T\+(\d+)\s*일", c)
    if m2:
        return int(m2.group(1))

    # 마지막 fallback: "T+3" 포함
    m3 = re.search(r"T\+(\d+)", c)
    if m3:
        return int(m3.group(1))

    return None


def _is_target_col(col: str) -> bool:
    c = _norm(col)
    if not any(k in c for k in _TARGET_KEYS):
        return False
    # T일 또는 T+ 가 있어야 horizon이라고 봄
    return ("T일" in c) or ("T+" in c)


def _is_last_year_col(col: str) -> bool:
    c = _norm(col)
    return ("작년" in c) and any(k in c for k in _TARGET_KEYS)


def detect_columns(df: pd.DataFrame) -> Tuple[str, List[str], List[str], List[str]]:
    """
    Returns:
      prod_col        : 제품 컬럼명
      target_cols     : 예측 대상 컬럼명 리스트 (T일, T+1, ... 순서 정렬)
      last_year_cols  : 전년도 수주량 컬럼명 리스트
      numeric_covars  : 추가 수치형 피처 컬럼명 리스트
    """

    # 1) 제품컬럼 후보
    prod_col = None
    for cand in ["Product_Number", "product", "SKU", "품번"]:
        if cand in df.columns:
            prod_col = cand
            break
    if prod_col is None:
        prod_col = df.columns[0]  # fallback

    # 2) 타깃 컬럼 탐지 + horizon 정렬
    cand_targets = [c for c in df.columns if _is_target_col(c)]
    with_h = []
    for c in cand_targets:
        h = _extract_horizon_index(c)
        if h is not None:
            with_h.append((h, c))
    # horizon이 없는 타깃이 있으면 뒤로 보냄(가능하면 사용자가 컬럼명 정리하도록 유도)
    no_h = [c for c in cand_targets if _extract_horizon_index(c) is None]

    with_h_sorted = [c for h, c in sorted(with_h, key=lambda x: x[0])]
    target_cols = with_h_sorted + no_h

    # 3) 전년도 컬럼
    last_year_cols = [c for c in df.columns if _is_last_year_col(c)]

    # 4) 기타 수치형 피처
    exclude = set([prod_col] + target_cols + last_year_cols)
    numeric_covars: List[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        # 숫자형만 포함 (bool 포함될 수 있어 제외하고 싶으면 아래에서 제외 가능)
        try:
            if np.issubdtype(df[c].dtype, np.number):
                numeric_covars.append(c)
        except TypeError:
            # dtype 판별 불가한 경우(복합/확장 dtype) -> 안전하게 스킵
            continue

    return prod_col, target_cols, last_year_cols, numeric_covars