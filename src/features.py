#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
features.py — 누출 없이 cross-horizon 기반 파생변수 생성 + 제품 패턴 클러스터링

기능 개요
- 안전한 DateTime 변환(열 증가 없이, 결측 최소화)
- 완전 중복행 제거
- (Product_Number, DateTime) 키 중복행 평균 집계(수치형 mean, 범주형 first)
- cross-horizon 파생: lag_diff_T+1~T+4, lag_ratio_T+1~T+4, cumsum_lag, trend_sign,
  mean_future, std_future, instability_coef, growth_index_T4
- 시간 파생: Datetime 기반 요일·월·시·분 및 주기적 인코딩(sin/cos)
- 제품 패턴 클러스터링(K=4, 저/중/고/중요 수요로 재라벨)

CLI 사용
python ./src/features.py --in ./data/data.csv --out ./data/feat.csv
"""

from __future__ import annotations
import argparse
import sys
import warnings
from typing import List, Optional, Tuple
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
        print(f"🔁 완전 중복 행 제거: {removed}행")
    return df2, removed


def _dedup_by_key_mean(df: pd.DataFrame, prod_col: str, dt_col: str) -> Tuple[pd.DataFrame, int]:
    before = len(df)
    if not {prod_col, dt_col}.issubset(df.columns):
        print("⚠️ 키 중복 병합 생략: 필요한 컬럼이 없습니다.")
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
        print(f"🧮 ({prod_col}, {dt_col}) 기준 병합: {removed}행 축소")
    return df2, removed


def _stabilize_ratio(s: pd.Series, clip_min: float = 0.0, clip_max: float = 5.0, fill_when_nan: float = 0.0) -> pd.Series:
    s = s.replace([np.inf, -np.inf], np.nan).fillna(fill_when_nan)
    if clip_min is not None or clip_max is not None:
        s = np.clip(s, clip_min if clip_min is not None else s.min(),
                    clip_max if clip_max is not None else s.max())
    return s

# =========================
# 파생변수 생성
# =========================

COLS = {
    "prod": "Product_Number",
    "dt": "DateTime",
    "demand_T": "T일 예정 수주량",
    "demand_Tp1": "T+1일 예정 수주량",
    "demand_Tp2": "T+2일 예정 수주량",
    "demand_Tp3": "T+3일 예정 수주량",
    "demand_Tp4": "T+4일 예정 수주량",
}

def add_cross_horizon_features(df: pd.DataFrame) -> pd.DataFrame:
    """현재 데이터 구조(T, T+1, ..., T+4)를 활용한 cross-horizon 파생변수"""
    T  = COLS["demand_T"]
    T1 = COLS["demand_Tp1"]
    T2 = COLS["demand_Tp2"]
    T3 = COLS["demand_Tp3"]
    T4 = COLS["demand_Tp4"]

    # ① Diff & Ratio (T 기준 변화)
    for k, col in enumerate([T1, T2, T3, T4], start=1):
        if col in df.columns and T in df.columns:
            df[f"lag_diff_T+{k}"] = (df[col] - df[T]).fillna(0.0)
            ratio = np.where(df[T] != 0, df[col] / df[T], np.nan)
            df[f"lag_ratio_T+{k}"] = _stabilize_ratio(pd.Series(ratio), 0.0, 5.0, 0.0)

    # ② 전체 미래 수주량 요약
    future_cols = [c for c in [T1, T2, T3, T4] if c in df.columns]
    if future_cols:
        df["cumsum_lag"] = df[future_cols].sum(axis=1)
        df["mean_future"] = df[future_cols].mean(axis=1)
        df["std_future"] = df[future_cols].std(axis=1)
        df["instability_coef"] = np.where(df["mean_future"] != 0,
                                          df["std_future"] / df["mean_future"], 0)
    else:
        print("⚠️ 미래 시점 열이 부족하여 요약형 파생변수 생략")

    # ③ 전체 추세 (T→T+4 기준)
    if T in df.columns and T4 in df.columns:
        delta = (df[T4] - df[T]).astype(float)
        df["trend_sign"] = np.sign(delta).astype("Int64")
        df["growth_index_T4"] = np.where(df[T] != 0, df[T4] / df[T], np.nan)
    elif T in df.columns and T2 in df.columns:
        delta = (df[T2] - df[T]).astype(float)
        df["trend_sign"] = np.sign(delta).astype("Int64")
        df["growth_index_T4"] = np.where(df[T] != 0, df[T2] / df[T], np.nan)
        print("ℹ️ T+4 부재로 trend_sign/growth_index_T4를 2-step 기준으로 계산")

    # ④ 작년 대비(있을 경우)
    if "작년 T일 예정 수주량" in df.columns:
        df["yoy_T"] = np.where(df["작년 T일 예정 수주량"] != 0,
                               df[T] / df["작년 T일 예정 수주량"], np.nan)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Datetime 기반 시간 파생 — 기존 DOW 문자열 제거 후 숫자형 요일 재계산"""
    dt_col = COLS["dt"]
    if dt_col not in df.columns:
        print("⚠️ 시간 파생 생략: DateTime 컬럼 없음")
        return df
    if not np.issubdtype(df[dt_col].dtype, np.datetime64):
        df[dt_col] = _safe_parse_datetime(df[dt_col])

    # ⚙️ 기존 문자열형 DOW 삭제
    if "DOW" in df.columns:
        df.drop(columns=["DOW"], inplace=True)
        print("ℹ️ 기존 'DOW' 컬럼 삭제 (Datetime 기준으로 새로 계산)")

    # ✅ Datetime으로부터 시간 관련 변수 생성
    df["dow"] = df[dt_col].dt.weekday       # 요일 (0=월, 6=일)
    df["month"] = df[dt_col].dt.month
    df["hour"] = df[dt_col].dt.hour
    df["minute"] = df[dt_col].dt.minute

    # ✅ 시간 주기성 반영
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
        print("⚠️ 클러스터링 생략: 필요한 컬럼이 없습니다.")
        return df, pd.DataFrame()

    feats = df.groupby(prod_col)[demand_col].agg(
        Mean_Demand="mean",
        Std_Demand="std",
        Zero_Ratio=lambda x: (x == 0).mean(),
        CV_Ratio=lambda x: (x.std() / x.mean()) if x.mean() != 0 else 0.0,
    ).fillna(0)
    feats.replace([np.inf, -np.inf], 0, inplace=True)

    X = StandardScaler().fit_transform(feats)
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    feats["_label"] = labels
    order = feats.groupby("_label")["Mean_Demand"].mean().sort_values().index.tolist()
    relabel_map = {old: new for new, old in enumerate(order)}
    feats["Cluster"] = [relabel_map[l] for l in labels]

    df_out = df.merge(feats[["Cluster"]], left_on=prod_col, right_index=True, how="left")
    feats.rename(columns={"Cluster": "Cluster(0=희소,1=간헐,2=다수,3=중요)"}, inplace=True)

    print("📦 제품 클러스터 분포:")
    print(feats["Cluster(0=희소,1=간헐,2=다수,3=중요)"].value_counts().sort_index().to_string())

    return df_out, feats

# =========================
# 메인 파이프라인
# =========================

def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    prod_col, dt_col = COLS["prod"], COLS["dt"]

    # ✅ 1️⃣ DateTime 변환
    if dt_col in df.columns:
        df[dt_col] = _safe_parse_datetime(df[dt_col])
        print(f"✅ DateTime 변환 완료 | 결측: {df[dt_col].isna().sum()}")
    else:
        print("⚠️ DateTime 컬럼 없음 — 시간 파생은 건너뜀")

    # ✅ 2️⃣ 완전 중복 제거 및 키 병합
    df, _ = _drop_full_duplicates(df)
    if prod_col in df.columns and dt_col in df.columns:
        df, _ = _dedup_by_key_mean(df, prod_col, dt_col)

    # ✅ 3️⃣ Humidity 이상치 처리 (clip 방식)
    if "Humidity" in df.columns:
        before_outliers = (df["Humidity"] > 100).sum() + (df["Humidity"] < 0).sum()
        if before_outliers > 0:
            print(f"🌡️ Humidity 이상치 {before_outliers}건 → 0~100으로 clip 처리")
        df["Humidity"] = df["Humidity"].clip(lower=0, upper=100)

    # ✅ 4️⃣ Cross-horizon 파생
    df = add_cross_horizon_features(df)

    # ✅ 5️⃣ 시간 파생
    df = add_time_features(df)

    # ✅ 6️⃣ 제품 클러스터링
    demand_col = COLS["demand_T"] if COLS["demand_T"] in df.columns else None
    clus_summary = pd.DataFrame()
    if demand_col:
        df, clus_summary = cluster_products(df, demand_col, prod_col)

    # ✅ 7️⃣ 정렬
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
    print(f"📥 입력: {args.inp} | shape={df.shape}")

    out_df, clus_summary = build_features(df)

    out_df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"💾 저장: {args.out} | shape={out_df.shape}")

    if not clus_summary.empty:
        clus_path = args.out.replace(".csv", "_cluster_summary.csv")
        clus_summary.to_csv(clus_path, encoding="utf-8-sig")
        print(f"💾 클러스터 요약 저장: {clus_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())