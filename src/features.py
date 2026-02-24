#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
features.py — 누출 없이 cross-horizon 기반 파생변수 생성 + 제품 패턴 클러스터링

[핵심 수정]
1) "예정 수주량" / "예상 수주량" 컬럼명 불일치 해결:
   - T, T+1..T+4 컬럼을 자동 탐지해서 런타임에 target map 구성
   - 작년 컬럼도 "작년 T일 예정/예상 수주량" 모두 지원
2) cross-horizon/클러스터링이 컬럼명 때문에 스킵되지 않도록 안전화
3) fillna(0)로 모든 컬럼을 덮어쓰지 않도록(데이터 왜곡 방지) 최소 범위로 처리
   - 파생변수/비율계열만 안전 처리
4) (추가) 키 컬럼(Product_Number/DateTime) normalize + fallback
   - Product_Number가 다른 이름으로 들어와도 최대한 잡아줌
"""

from __future__ import annotations
import argparse
import sys
import warnings
import re
from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# =========================
# 유틸 함수
# =========================

def _safe_parse_datetime(series: pd.Series) -> pd.Series:
    s_raw = series.astype(str).str.strip()
    parsed = pd.to_datetime(s_raw, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    mask = parsed.isna() & s_raw.notna()
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(s_raw.loc[mask], errors="coerce")
    mask = parsed.isna() & s_raw.notna()
    if mask.any():
        s_norm = (
            s_raw.loc[mask]
            .str.replace(".", "-", regex=False)
            .str.replace("/", "-", regex=False)
        )
        parsed.loc[mask] = pd.to_datetime(s_norm, errors="coerce")
    return parsed


def _drop_full_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    before = len(df)
    df2 = df.drop_duplicates(keep="first").copy()
    removed = before - len(df2)
    if removed:
        print(f"완전 중복 행 제거: {removed}행")
    return df2, removed


def _dedup_by_key_mean(df: pd.DataFrame, prod_col: str, dt_col: str) -> Tuple[pd.DataFrame, int]:
    before = len(df)
    if not {prod_col, dt_col}.issubset(df.columns):
        print("키 중복 병합 생략: 필요한 컬럼이 없습니다.")
        return df, 0

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in [prod_col, dt_col]]
    non_num_cols = [c for c in df.columns if c not in num_cols + [prod_col, dt_col]]

    agg_dict = {**{c: "mean" for c in num_cols}, **{c: "first" for c in non_num_cols}}
    df2 = (
        df.groupby([prod_col, dt_col], as_index=False)
          .agg(agg_dict)
          .sort_values([prod_col, dt_col])
          .reset_index(drop=True)
    )
    removed = before - len(df2)
    if removed:
        print(f"({prod_col}, {dt_col}) 기준 병합: {removed}행 축소")
    return df2, removed


def _stabilize_ratio(s: pd.Series, clip_min: float = 0.0, clip_max: float = 5.0, fill_when_nan: float = 0.0) -> pd.Series:
    s = s.replace([np.inf, -np.inf], np.nan).fillna(fill_when_nan)
    if clip_min is not None or clip_max is not None:
        s = np.clip(
            s,
            clip_min if clip_min is not None else s.min(),
            clip_max if clip_max is not None else s.max(),
        )
    return s


# =========================
# 키 컬럼 탐지/정규화
# =========================

DEFAULT_COLS = {
    "prod": "Product_Number",
    "dt": "DateTime",
}

def _normalize_key_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    """
    Product_Number/DateTime이 다른 이름으로 들어올 수 있어서 최대한 맞춰준다.
    - prod 후보: Product_Number, product, SKU, 품번
    - dt 후보: DateTime, datetime, Date, 날짜
    """
    df = df.copy()

    prod_col = None
    for cand in ["Product_Number", "product", "SKU", "품번"]:
        if cand in df.columns:
            prod_col = cand
            break
    if prod_col is None:
        prod_col = DEFAULT_COLS["prod"]
        if prod_col not in df.columns:
            # 진짜 없으면 첫 컬럼 fallback (최후)
            prod_col = df.columns[0]

    dt_col = None
    for cand in ["DateTime", "datetime", "Date", "날짜"]:
        if cand in df.columns:
            dt_col = cand
            break
    if dt_col is None:
        dt_col = DEFAULT_COLS["dt"]

    # 표준명으로 rename(필요한 경우만)
    rename_map = {}
    if prod_col != DEFAULT_COLS["prod"]:
        rename_map[prod_col] = DEFAULT_COLS["prod"]
    if dt_col in df.columns and dt_col != DEFAULT_COLS["dt"]:
        rename_map[dt_col] = DEFAULT_COLS["dt"]

    if rename_map:
        df = df.rename(columns=rename_map)

    return df, DEFAULT_COLS["prod"], DEFAULT_COLS["dt"]


# =========================
# 타깃 컬럼 자동 탐지
# =========================

def _norm_col(s: str) -> str:
    """컬럼명 normalize(공백/특수문자 완화)"""
    return re.sub(r"\s+", " ", str(s)).strip()


def detect_target_cols(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    데이터프레임에서 다음 컬럼을 자동 탐지하여 반환:
    - demand_T, demand_Tp1..Tp4
    - yoy_T (작년 T일 ...)
    '예정/예상' 모두 지원.
    """
    cols_norm = [_norm_col(c) for c in df.columns]
    col_map = { _norm_col(c): c for c in df.columns }  # normalized -> original

    def _find_fullmatch(patterns: List[str]) -> Optional[str]:
        for pat in patterns:
            rgx = re.compile(pat)
            for c in cols_norm:
                if rgx.fullmatch(c):
                    return col_map[c]
        return None

    out: Dict[str, Optional[str]] = {
        "demand_T": None,
        "demand_Tp1": None,
        "demand_Tp2": None,
        "demand_Tp3": None,
        "demand_Tp4": None,
        "yoy_T": None,
    }

    out["demand_T"] = _find_fullmatch([
        r"T일\s*(예정|예상)\s*수주량",
        r"T\s*일\s*(예정|예상)\s*수주량",
    ])
    out["demand_Tp1"] = _find_fullmatch([r"T\+1일\s*(예정|예상)\s*수주량"])
    out["demand_Tp2"] = _find_fullmatch([r"T\+2일\s*(예정|예상)\s*수주량"])
    out["demand_Tp3"] = _find_fullmatch([r"T\+3일\s*(예정|예상)\s*수주량"])
    out["demand_Tp4"] = _find_fullmatch([r"T\+4일\s*(예정|예상)\s*수주량"])

    out["yoy_T"] = _find_fullmatch([
        r"작년\s*T일\s*(예정|예상)\s*수주량",
        r"작년\s*T\s*일\s*(예정|예상)\s*수주량",
    ])

    # fallback: 부분 문자열 기반(T일만이라도 꼭 잡기)
    if out["demand_T"] is None:
        for c0 in df.columns:
            c = _norm_col(c0)
            if ("T일" in c or "T 일" in c) and ("수주량" in c) and (("예상" in c) or ("예정" in c)):
                out["demand_T"] = c0
                break

    return out


# =========================
# 파생변수 생성
# =========================

def add_cross_horizon_features(df: pd.DataFrame, cols: Dict[str, Optional[str]]) -> pd.DataFrame:
    """현재 데이터 구조(T, T+1, ..., T+4)를 활용한 cross-horizon 파생변수"""
    T  = cols.get("demand_T")
    T1 = cols.get("demand_Tp1")
    T2 = cols.get("demand_Tp2")
    T3 = cols.get("demand_Tp3")
    T4 = cols.get("demand_Tp4")

    if not T or T not in df.columns:
        print("cross-horizon 파생 생략: T일 (예정/예상) 수주량 컬럼을 찾지 못했습니다.")
        return df

    # 1) Diff & Ratio (T 기준 변화)
    for k, col in enumerate([T1, T2, T3, T4], start=1):
        if col and col in df.columns:
            base = pd.to_numeric(df[T], errors="coerce").astype(float)
            fut  = pd.to_numeric(df[col], errors="coerce").astype(float)

            df[f"lag_diff_T+{k}"] = (fut - base).fillna(0.0)
            ratio = np.where(base != 0, fut / base, np.nan)
            df[f"lag_ratio_T+{k}"] = _stabilize_ratio(pd.Series(ratio), 0.0, 5.0, 0.0).astype(float)

    # 2) 전체 미래 수주량 요약
    future_cols = [c for c in [T1, T2, T3, T4] if (c and c in df.columns)]
    if future_cols:
        fut_df = df[future_cols].apply(pd.to_numeric, errors="coerce")
        df["cumsum_lag"] = fut_df.sum(axis=1).fillna(0.0).astype(float)
        df["mean_future"] = fut_df.mean(axis=1).fillna(0.0).astype(float)
        df["std_future"] = fut_df.std(axis=1).fillna(0.0).astype(float)
        df["instability_coef"] = np.where(df["mean_future"] != 0, df["std_future"] / df["mean_future"], 0.0).astype(float)
    else:
        print("미래 시점 열이 부족하여 요약형 파생변수 생략")

    # 3) 전체 추세 (가능하면 T→T+4, 아니면 T→T+2)
    base = pd.to_numeric(df[T], errors="coerce").astype(float)
    if T4 and T4 in df.columns:
        t4 = pd.to_numeric(df[T4], errors="coerce").astype(float)
        delta = t4 - base
        df["trend_sign"] = np.sign(delta).astype("Int64")
        df["growth_index_T4"] = _stabilize_ratio(pd.Series(np.where(base != 0, t4 / base, np.nan)), 0.0, 10.0, 0.0).astype(float)
    elif T2 and T2 in df.columns:
        t2 = pd.to_numeric(df[T2], errors="coerce").astype(float)
        delta = t2 - base
        df["trend_sign"] = np.sign(delta).astype("Int64")
        df["growth_index_T4"] = _stabilize_ratio(pd.Series(np.where(base != 0, t2 / base, np.nan)), 0.0, 10.0, 0.0).astype(float)
        print("T+4 부재로 trend_sign/growth_index_T4를 2-step 기준으로 계산")

    # 4) 작년 대비(있을 경우)
    yoy_col = cols.get("yoy_T")
    if yoy_col and yoy_col in df.columns:
        yoy = pd.to_numeric(df[yoy_col], errors="coerce").astype(float)
        df["yoy_T"] = _stabilize_ratio(pd.Series(np.where(yoy != 0, base / yoy, np.nan)), 0.0, 10.0, 0.0).astype(float)

    # ✅ 파생변수에서만 inf/nan 정리
    gen_prefix = ("lag_diff_", "lag_ratio_", "cumsum_lag", "mean_future", "std_future",
                  "instability_coef", "trend_sign", "growth_index_", "yoy_")
    gen_cols = [c for c in df.columns if c.startswith(gen_prefix)]
    if gen_cols:
        df[gen_cols] = df[gen_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df


def add_time_features(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    """Datetime 기반 시간 파생 — 기존 DOW 문자열 제거 후 숫자형 요일 재계산"""
    if dt_col not in df.columns:
        print("시간 파생 생략: DateTime 컬럼 없음")
        return df
    if not np.issubdtype(df[dt_col].dtype, np.datetime64):
        df[dt_col] = _safe_parse_datetime(df[dt_col])

    if "DOW" in df.columns:
        df.drop(columns=["DOW"], inplace=True)
        print("기존 'DOW' 컬럼 삭제 (Datetime 기준으로 새로 계산)")

    df["dow"] = df[dt_col].dt.weekday
    df["month"] = df[dt_col].dt.month
    df["hour"] = df[dt_col].dt.hour
    df["minute"] = df[dt_col].dt.minute

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)

    return df


# =========================
# 제품 클러스터링 (K=4)
# =========================

def cluster_products(df: pd.DataFrame, demand_col: str, prod_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not {demand_col, prod_col}.issubset(df.columns):
        print("클러스터링 생략: 필요한 컬럼이 없습니다.")
        return df, pd.DataFrame()

    x = pd.to_numeric(df[demand_col], errors="coerce").fillna(0.0)

    feats = df.assign(__demand__=x).groupby(prod_col)["__demand__"].agg(
        Mean_Demand="mean",
        Std_Demand="std",
        Zero_Ratio=lambda v: (v == 0).mean(),
        CV_Ratio=lambda v: (v.std() / v.mean()) if v.mean() != 0 else 0.0,
    ).fillna(0.0)

    feats.replace([np.inf, -np.inf], 0.0, inplace=True)

    X = StandardScaler().fit_transform(feats)
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    feats["_label"] = labels
    order = feats.groupby("_label")["Mean_Demand"].mean().sort_values().index.tolist()
    relabel_map = {old: new for new, old in enumerate(order)}
    feats["Cluster"] = [relabel_map[l] for l in labels]

    df_out = df.merge(feats[["Cluster"]], left_on=prod_col, right_index=True, how="left")

    feats.rename(columns={"Cluster": "Cluster(0=희소,1=간헐,2=다수,3=중요)"}, inplace=True)

    print("제품 클러스터 분포:")
    print(feats["Cluster(0=희소,1=간헐,2=다수,3=중요)"].value_counts().sort_index().to_string())

    return df_out, feats


# =========================
# 메인 파이프라인
# =========================

def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # 0) 키컬럼 normalize
    df, prod_col, dt_col = _normalize_key_columns(df)

    # 1) 타깃 컬럼 자동 탐지 (예정/예상)
    tgt = detect_target_cols(df)
    if tgt.get("demand_T"):
        print(f"[targets] T={tgt.get('demand_T')}, "
              f"T+1={tgt.get('demand_Tp1')}, T+2={tgt.get('demand_Tp2')}, "
              f"T+3={tgt.get('demand_Tp3')}, T+4={tgt.get('demand_Tp4')}")

    # 2) DateTime 변환
    if dt_col in df.columns:
        df[dt_col] = _safe_parse_datetime(df[dt_col])
        print(f"DateTime 변환 완료 | 결측: {df[dt_col].isna().sum()}")
    else:
        print("DateTime 컬럼 없음 — 시간 파생은 건너뜀")

    # 3) 완전 중복 제거 및 키 병합
    df, _ = _drop_full_duplicates(df)
    if prod_col in df.columns and dt_col in df.columns:
        df, _ = _dedup_by_key_mean(df, prod_col, dt_col)

    # 4) Humidity 이상치 처리 (clip 방식)
    if "Humidity" in df.columns:
        before_outliers = int((df["Humidity"] > 100).sum() + (df["Humidity"] < 0).sum())
        if before_outliers > 0:
            print(f"🌡️ Humidity 이상치 {before_outliers}건 → 0~100으로 clip 처리")
        df["Humidity"] = pd.to_numeric(df["Humidity"], errors="coerce").clip(lower=0, upper=100)

    # 5) Cross-horizon 파생
    df = add_cross_horizon_features(df, tgt)

    # 6) 시간 파생
    df = add_time_features(df, dt_col)

    # 7) 제품 클러스터링 (T일 수주량 기준)
    clus_summary = pd.DataFrame()
    demand_col = tgt.get("demand_T")
    if demand_col and demand_col in df.columns:
        df, clus_summary = cluster_products(df, demand_col, prod_col)
    else:
        print("클러스터링 생략: T일 (예정/예상) 수주량 컬럼을 찾지 못했습니다.")

    # 8) 정렬
    sort_cols = [c for c in [prod_col, dt_col] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    return df, clus_summary


# =========================
# CLI
# =========================

def _read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, encoding="utf-8-sig")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Preprocess & Feature Engineering")
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", dest="out", required=True)
    args = p.parse_args(argv)

    df = _read_csv(args.inp)
    print(f"입력: {args.inp} | shape={df.shape}")

    out_df, clus_summary = build_features(df)

    out_df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"저장: {args.out} | shape={out_df.shape}")

    if not clus_summary.empty:
        clus_path = args.out.replace(".csv", "_cluster_summary.csv")
        clus_summary.to_csv(clus_path, encoding="utf-8-sig")
        print(f"클러스터 요약 저장: {clus_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())