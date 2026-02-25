#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
forecast.py — Leakage-safe Multi-Horizon Forecast (feat.csv → predict)

[수정 핵심(현 파이프라인 기준 권장)]
- ✅ y(타깃)는 기본적으로 "예정 수주량"만 사용 (planner 입력/수요 예측의 정석)
- ✅ "예상 수주량"은 기본적으로 feature로 활용(=기존 baseline forecast 보정) 가능
  - 단, 원하면 옵션으로 feature 제외/타깃 전환 가능
- ✅ '작년/전년/지난해'는 과거 참조이므로
  - 타깃에서 무조건 제외 (누수/혼입 방지)
  - feature로는 적극 허용
- ✅ horizon 선택 가능(기본 T+1~T+4). planner와 MC day_idx 정의가 깔끔해짐.
- ✅ MC 시나리오: 잔차 벡터 resample로 horizon 상관 보존 + mean-centering
- ✅ planner 입력(pred_*_by_product.csv)과 MC d_hat은 "제품별 최신 스냅샷" 기준으로 통일

[추가 안정화(매우 중요)]
- ✅ find_target_cols: 구버전 호출(키워드 리스트 전달)과 신버전 호출(target_kind str) 모두 허용
- ✅ build_xy: 구버전 호출(build_xy(df, prod_col, target_cols, log_target))이 깨지지 않도록
          시그니처를 (log_target이 4번째 포지션)으로 호환
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Optional, Union
import argparse
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import TweedieRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    precision_recall_fscore_support,
)

# ---- Optional deps
try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    import optuna
except Exception:
    optuna = None


DEFAULT_PROD_COL = "Product_Number"
DEFAULT_DT_COL   = "DateTime"

PAST_MARKERS = ["작년", "전년", "지난해"]

KIND_TO_KEYWORDS = {
    "planned":  ["예정 수주량"],   # y 기본(권장)
    "expected": ["예상 수주량"],   # baseline forecast
    "both":     ["예상 수주량", "예정 수주량"],
}


# =========================================================
# Params save/load
# =========================================================
def load_best_params(path: str | Path) -> Dict | None:
    try:
        p = Path(path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                params = json.load(f)
            print(f"Loaded best params from: {p}")
            return params
    except Exception as e:
        print(f"Failed to load best params ({path}): {e}")
    return None


def save_best_params(path: str | Path, params: Dict) -> None:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=2)
        print(f"Saved best params to: {p}")
    except Exception as e:
        print(f"Failed to save best params ({path}): {e}")


# =========================================================
# Target / Horizon utils
# =========================================================
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def _is_past_ref(col: str) -> bool:
    cc = _norm(col)
    return any(m in cc for m in PAST_MARKERS)


def _extract_horizon_index(col: str) -> Optional[int]:
    """
    - "T일 ..." -> 0
    - "T+3일 ..." -> 3
    - 축약형 "T" -> 0
    - 축약형 "T+3" -> 3
    """
    c = _norm(col)

    # 축약형
    if re.fullmatch(r"T", c):
        return 0
    m = re.fullmatch(r"T\+(\d+)", c)
    if m:
        return int(m.group(1))

    # 한국어/기타
    if "T일" in c:
        return 0
    m2 = re.search(r"T\+(\d+)\s*일", c)
    if m2:
        return int(m2.group(1))
    m3 = re.search(r"T\+(\d+)", c)
    if m3:
        return int(m3.group(1))

    return None


def _target_priority(col: str) -> int:
    """
    같은 horizon 내에서 정렬 우선순위:
      0: '예정'
      1: '예상'
      9: 기타
    """
    cc = _norm(col)
    if "예정" in cc:
        return 0
    if "예상" in cc:
        return 1
    return 9


def _parse_horizons(s: str) -> List[int]:
    """
    예: "1,2,3,4" -> [1,2,3,4]
    예: "0,1,2"   -> [0,1,2]
    """
    parts = [p.strip() for p in str(s).split(",") if p.strip() != ""]
    hs = []
    for p in parts:
        hs.append(int(p))
    hs = sorted(list(dict.fromkeys(hs)))
    return hs


def _find_target_cols_by_keywords(
    df: pd.DataFrame,
    keywords: List[str],
    horizons: Optional[List[int]] = None,
) -> List[str]:
    """
    구버전 호환용: keywords(예: ["예상 수주량","예정 수주량"])를 직접 받아 타깃 탐지.
    - 과거 참조('작년/전년/지난해')는 무조건 제외
    - horizon 패턴 포함
    - horizons가 주어지면 그 h만 남김
    """
    cands: List[str] = []
    for c in df.columns:
        cc = _norm(c)

        if _is_past_ref(cc):
            continue

        has_kw = any(k in cc for k in keywords)
        has_h = ("T일" in cc) or ("T+" in cc) or bool(re.search(r"\bT(\+\d+)?\b", cc))
        if not (has_kw and has_h):
            continue

        h = _extract_horizon_index(cc)
        if horizons is not None and h is not None:
            if h not in set(horizons):
                continue

        cands.append(c)

    tagged = []
    for c in cands:
        h = _extract_horizon_index(c)
        if h is None:
            tagged.append((10_000, 9, c))
        else:
            tagged.append((h, _target_priority(c), c))

    tagged.sort(key=lambda x: (x[0], x[1], str(x[2])))
    out = [c for _, _, c in tagged]
    out = list(dict.fromkeys(out))
    return out


def find_target_cols(
    df: pd.DataFrame,
    target_kind: Union[str, List[str], Tuple[str, ...]],
    horizons: Optional[List[int]] = None,
) -> List[str]:
    """
    ✅ 타깃 컬럼 탐지 (호환 포함)

    [신버전 사용]
      find_target_cols(df, target_kind="planned"|"expected"|"both", horizons=[1,2,3,4])

    [구버전 호환]
      find_target_cols(df, ["예상 수주량","예정 수주량"], horizons=[...])

    공통 정책:
      - '작년/전년/지난해' 포함 컬럼은 타깃에서 무조건 제외 (누수/혼입 방지)
      - (h, priority, name) 정렬
    """
    # ---- 구버전: 키워드 리스트를 넘긴 경우 ----
    if isinstance(target_kind, (list, tuple)):
        keywords = [str(x) for x in target_kind]
        return _find_target_cols_by_keywords(df, keywords=keywords, horizons=horizons)

    # ---- 신버전: kind 문자열 ----
    if target_kind not in KIND_TO_KEYWORDS:
        raise ValueError(f"invalid target_kind={target_kind}. choose from {list(KIND_TO_KEYWORDS.keys())}")
    keywords = KIND_TO_KEYWORDS[target_kind]
    return _find_target_cols_by_keywords(df, keywords=keywords, horizons=horizons)


# =========================================================
# Feature selection (leakage control)
# =========================================================
def select_feature_columns(
    df: pd.DataFrame,
    prod_col: str,
    target_cols: List[str],
    target_kind: str,
    allow_expected_as_feature: bool,
) -> Tuple[List[str], List[str]]:
    """
    기본 정책(권장):
    - y가 planned(예정)일 때: '예상 수주량'(non-past)은 feature로 허용 가능(보정 모델)
    - y가 expected 또는 both일 때: 예/예정(non-past) 모두 feature에서 제외(보수적으로 누수 방지)
    - 작년/전년/지난해는 과거로 간주 → feature 허용
    """
    numeric_all = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    excluded: List[str] = []

    def _is_nonpast_expected(col: str) -> bool:
        cc = _norm(col)
        return ("예상 수주량" in cc) and (not _is_past_ref(cc))

    def _is_nonpast_planned(col: str) -> bool:
        cc = _norm(col)
        return ("예정 수주량" in cc) and (not _is_past_ref(cc))

    for c in list(numeric_all):
        if c in target_cols or c == prod_col:
            excluded.append(c)
            continue

        if target_kind == "planned":
            if _is_nonpast_planned(c):
                excluded.append(c)
                continue
            if _is_nonpast_expected(c) and (not allow_expected_as_feature):
                excluded.append(c)
                continue
        else:
            if _is_nonpast_expected(c) or _is_nonpast_planned(c):
                excluded.append(c)
                continue

    num_cols = [c for c in numeric_all if c not in set(excluded + [prod_col])]
    return num_cols, excluded


def build_xy(
    df: pd.DataFrame,
    prod_col: str,
    target_cols: List[str],
    log_target: bool = False,
    target_kind: str = "planned",
    allow_expected_as_feature: bool = True,
):
    """
    ✅ backward-compatible signature

    - 구버전(많이 쓰던 형태):
        X, y, num_cols, cat_cols, excluded = build_xy(df, prod_col, target_cols, log_target)

    - 신버전:
        build_xy(df, prod_col, target_cols, log_target=False, target_kind="planned", allow_expected_as_feature=True)
    """
    if prod_col not in df.columns:
        raise ValueError(f"'{prod_col}' 컬럼이 없습니다.")

    y = df[target_cols].astype(float).clip(lower=0)
    if log_target:
        y = np.log1p(y)

    num_cols, excluded = select_feature_columns(
        df=df,
        prod_col=prod_col,
        target_cols=target_cols,
        target_kind=target_kind,
        allow_expected_as_feature=allow_expected_as_feature,
    )
    cat_cols = [prod_col]
    if len(num_cols) == 0:
        raise RuntimeError("사용 가능한 수치형 피처가 없습니다. features.py를 확인하세요.")

    X = df[num_cols + cat_cols].copy()
    print(f"Features used: {len(num_cols)} numeric + {len(cat_cols)} categorical")
    if excluded:
        print(f"Excluded (leakage/targets): {len(excluded)} cols")
    return X, y, num_cols, cat_cols, excluded


# =========================================================
# Metrics
# =========================================================
def binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    y_true_bin = (y_true > 0).astype(int)

    def _safe(f, *args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception:
            return np.nan

    auc = _safe(roc_auc_score, y_true_bin, y_score)

    uniq = np.unique(y_score)
    if len(uniq) > 200:
        uniq = np.unique(np.quantile(y_score, np.linspace(0, 1, 200)))

    best = {"f1": -1.0, "p": np.nan, "r": np.nan, "thr": 0.0}
    for thr in uniq:
        y_pred = (y_score >= thr).astype(int)
        p, r, f, _ = precision_recall_fscore_support(
            y_true_bin, y_pred, average="binary", zero_division=0
        )
        if f > best["f1"]:
            best = {"f1": float(f), "p": float(p), "r": float(r), "thr": float(thr)}
    return {
        "AUC": float(auc),
        "F1": best["f1"],
        "Precision": best["p"],
        "Recall": best["r"],
        "BestThreshold": best["thr"],
    }


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


# =========================================================
# Splits
# =========================================================
def time_split(df_raw: pd.DataFrame, X: pd.DataFrame, y: pd.DataFrame, dt_col: str, val_ratio: float):
    dt = pd.to_datetime(df_raw[dt_col], errors="coerce", format="mixed")
    if dt.isna().any():
        bad = df_raw.loc[dt.isna(), dt_col].head(5).tolist()
        raise ValueError(f"[time_split] DateTime parse failed. examples={bad}")
    cutoff = dt.quantile(1 - val_ratio)
    idx_tr, idx_va = (dt <= cutoff), (dt > cutoff)
    return X[idx_tr], X[idx_va], y[idx_tr], y[idx_va]


def group_split(df_raw: pd.DataFrame, X: pd.DataFrame, y: pd.DataFrame, group_col: str, val_ratio: float, seed: int):
    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    tr_idx, va_idx = next(gss.split(X, y, groups=df_raw[group_col]))
    return X.iloc[tr_idx], X.iloc[va_idx], y.iloc[tr_idx], y.iloc[va_idx]


# =========================================================
# Pipeline builders
# =========================================================
def build_model_pipeline(
    model_name: str,
    num_cols: List[str],
    cat_cols: List[str],
    tweedie_power: float,
    alpha: float,
    lgbm_params: Dict,
    reg_n_jobs: int = -1,
) -> Pipeline:
    if model_name == "tweedie":
        base = MultiOutputRegressor(
            TweedieRegressor(power=tweedie_power, alpha=alpha, link="log", max_iter=2000),
            n_jobs=reg_n_jobs,
        )
        prep = ColumnTransformer(
            [("num", RobustScaler(), num_cols), ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)]
        )
        return Pipeline([("prep", prep), ("reg", base)])

    if model_name == "lgbm":
        if lgb is None:
            raise RuntimeError("lightgbm 미설치. pip install lightgbm")
        base = MultiOutputRegressor(
            lgb.LGBMRegressor(**lgbm_params, n_jobs=reg_n_jobs),
            n_jobs=reg_n_jobs,
        )
        prep = ColumnTransformer(
            [("num", "passthrough", num_cols), ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)]
        )
        return Pipeline([("prep", prep), ("reg", base)])

    raise ValueError("지원하지 않는 모델 이름")


# =========================================================
# MC Scenario Generation
# =========================================================
def generate_demand_scenarios(
    d_hat: np.ndarray,
    residuals: np.ndarray,
    n_scenarios: int,
    seed: int = 2025
) -> np.ndarray:
    """
    residual bootstrap 기반 시나리오 생성.
    ✅ horizon별 독립 샘플링이 아니라, 잔차 '벡터(행)'를 통째로 resample하여
    horizon 간 상관관계를 보존한다.

    d_hat: (n, k)
    residuals: (m, k)  (y_true - y_pred), horizon별 mean=0 권장(바이어스 완화)
    return: (S, n, k)
    """
    rng = np.random.default_rng(seed)
    S = int(n_scenarios)
    if S <= 0:
        raise ValueError("n_scenarios must be positive")

    d_hat = np.asarray(d_hat, dtype=float)
    residuals = np.asarray(residuals, dtype=float)

    n, k = d_hat.shape
    if residuals.ndim != 2 or residuals.shape[1] != k:
        raise ValueError(f"residuals shape must be (m, {k}), got {residuals.shape}")

    m = residuals.shape[0]
    if m <= 0:
        raise ValueError("residual pool is empty")

    idx = rng.integers(0, m, size=(S, n))
    E = residuals[idx, :]                  # (S, n, k)
    scenarios = np.maximum(0.0, d_hat[None, :, :] + E)
    return scenarios


def snapshot_latest_by_product(
    df: pd.DataFrame,
    prod_col: str,
    dt_col: str,
    value_cols: List[str],
) -> pd.DataFrame:
    """
    Product별 최신 DateTime 스냅샷.
    - dt_col 없거나 파싱 불가면 product별 평균으로 대체
    반환: [prod_col] + value_cols
    """
    if prod_col not in df.columns:
        raise ValueError(f"'{prod_col}' column not found")
    for c in value_cols:
        if c not in df.columns:
            raise ValueError(f"'{c}' column not found for snapshot")

    out = df.copy()

    if dt_col not in out.columns:
        grp = out.groupby(prod_col, as_index=False)[value_cols].mean(numeric_only=True)
        return grp

    out[dt_col] = pd.to_datetime(out[dt_col], errors="coerce")
    out = out.dropna(subset=[dt_col]).copy()
    if out.empty:
        grp = df.groupby(prod_col, as_index=False)[value_cols].mean(numeric_only=True)
        return grp

    latest = out.groupby(prod_col, as_index=False)[dt_col].max().rename(columns={dt_col: "_LatestDT"})
    merged = out.merge(latest, on=prod_col, how="inner")
    picked = merged[merged[dt_col] == merged["_LatestDT"]].copy()

    for c in value_cols:
        picked[c] = pd.to_numeric(picked[c], errors="coerce").fillna(0.0)

    snap = picked.groupby(prod_col, as_index=False)[value_cols].mean(numeric_only=True)
    return snap


def scenarios_to_long_csv(
    scenarios: np.ndarray,
    products: List[str],
    out_csv: str,
    product_col_name: str = "Product_Number",
) -> None:
    """
    scenarios: (S, P, D)
    저장: scenario_id, day_idx, Product_Number, demand
    """
    S, P, D = scenarios.shape
    if len(products) != P:
        raise ValueError(f"products length mismatch: {len(products)} vs P={P}")

    rows = []
    for s in range(S):
        for i, p in enumerate(products):
            for d in range(D):
                rows.append({
                    "scenario_id": int(s),
                    "day_idx": int(d),
                    product_col_name: str(p),
                    "demand": float(scenarios[s, i, d]),
                })
    df_long = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df_long.to_csv(out_csv, index=False, encoding="utf-8-sig")


# =========================================================
# Train / Validate / Predict
# =========================================================
def train_validate(
    df_raw, X, y, model,
    split="time", val_size=0.2, seed=2025,
    dt_col=DEFAULT_DT_COL, prod_col=DEFAULT_PROD_COL,
    log_target=False
):
    if split == "time":
        X_tr, X_va, y_tr, y_va = time_split(df_raw, X, y, dt_col, val_size)
    elif split == "group":
        X_tr, X_va, y_tr, y_va = group_split(df_raw, X, y, prod_col, val_size, seed)
    else:
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=val_size, random_state=seed)

    model.fit(X_tr, y_tr.values)
    pred = np.maximum(0.0, np.asarray(model.predict(X_va), dtype=float))

    if log_target:
        pred = np.expm1(pred)
        y_va = np.expm1(y_va)

    rows = {}
    for i, t in enumerate(y.columns):
        yt = y_va[t].values
        pt = pred[:, i]
        rows[t] = {
            "MAE": mean_absolute_error(yt, pt),
            "RMSE": rmse(yt, pt),
            "R2": r2_score(yt, pt),
            "SMAPE": smape(yt, pt),
            **binary_metrics(yt, pt),
        }

    y_true = y_va.values.astype(float)
    y_pred = pred.astype(float)
    residuals = y_true - y_pred
    return model, pd.DataFrame(rows).T, residuals


def predict_all(model, X_all, df_raw, prod_col, dt_col, target_cols):
    pred = np.maximum(0.0, np.asarray(model.predict(X_all), dtype=float))
    out = pd.DataFrame(pred, columns=target_cols, index=X_all.index)
    out[prod_col] = df_raw.loc[X_all.index, prod_col].values
    if dt_col in df_raw.columns:
        out[dt_col] = df_raw.loc[X_all.index, dt_col].values
        cols = [prod_col, dt_col] + target_cols
    else:
        cols = [prod_col] + target_cols
    return out[cols]


# =========================================================
# Tuning helpers
# =========================================================
def average_metric(metrics_df: pd.DataFrame, target_cols: List[str], metric: str, emphasize: Optional[Dict[str, float]] = None) -> float:
    if metric not in metrics_df.columns:
        raise ValueError(f"metrics_df missing metric='{metric}'. cols={list(metrics_df.columns)}")
    w = {t: 1.0 for t in target_cols}
    if emphasize:
        for k, v in emphasize.items():
            if k in w:
                w[k] = float(v)
    total_w = sum(w.values())
    return float((metrics_df.loc[target_cols, metric] * pd.Series(w)).sum() / total_w)


def _parse_seed_list(seed_list_str: Optional[str]) -> Optional[List[int]]:
    if not seed_list_str:
        return None
    parts = [p.strip() for p in seed_list_str.split(",") if p.strip()]
    if not parts:
        return None
    return [int(p) for p in parts]


def tune_params(args, df, X, y, num_cols, cat_cols, target_cols):
    if optuna is None:
        print("Optuna 미설치: 튜닝을 건너뜁니다.")
        return None
    if args.model != "lgbm":
        raise ValueError("tune_params는 현재 lgbm 튜닝만 지원합니다. (--model lgbm)")

    seed_list = _parse_seed_list(args.tune_seed_list)
    if seed_list is None:
        seed_list = [int(args.seed + i) for i in range(int(args.tune_seeds))]
    if len(seed_list) < 1:
        seed_list = [int(args.seed)]

    agg = str(args.tune_agg).lower()
    if agg not in ("mean", "median"):
        raise ValueError("--tune_agg must be one of: mean, median")

    w_mae = float(args.w_mae)
    w_sm = float(args.w_smape)
    if w_mae < 0 or w_sm < 0 or (w_mae + w_sm) <= 0:
        raise ValueError("weights must be non-negative and not both zero. (w_mae + w_smape > 0)")

    mean_y = float(np.asarray(y.values, dtype=float).mean()) + 1e-8

    def objective(trial):
        params = dict(
            objective="tweedie",
            tweedie_variance_power=trial.suggest_float("power", 1.1, 1.6),
            learning_rate=trial.suggest_float("lr", 0.01, 0.1, log=True),
            n_estimators=trial.suggest_int("n_estimators", 400, 2000),
            num_leaves=trial.suggest_int("num_leaves", 31, 255),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 120),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.0, 20.0),
            random_state=args.seed,
        )
        scores = []

        for sd in seed_list:
            params_sd = dict(params)
            params_sd["random_state"] = int(sd)

            if args.deterministic:
                params_sd["deterministic"] = True
                params_sd["force_row_wise"] = True

            reg_n_jobs = 1 if args.deterministic else -1
            model = build_model_pipeline(
                "lgbm", num_cols, [args.prod_col],
                tweedie_power=params_sd["tweedie_variance_power"],
                alpha=0.5,
                lgbm_params=params_sd,
                reg_n_jobs=reg_n_jobs,
            )

            _, mdf, _ = train_validate(
                df, X, y, model,
                split=args.split, val_size=args.val_size,
                seed=int(sd), dt_col=args.dt_col, prod_col=args.prod_col,
                log_target=args.log_target
            )

            mae = average_metric(mdf, target_cols, "MAE")
            smp = average_metric(mdf, target_cols, "SMAPE")

            norm_mae = mae / mean_y
            score = w_mae * norm_mae + w_sm * smp
            scores.append(float(score))

        if agg == "median":
            return float(np.median(scores))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=int(args.trials), show_progress_bar=False)
    print("Best params:", study.best_params)
    print("Best value:", study.best_value)
    return study.best_params


# =========================================================
# Residual pool + residual clip
# =========================================================
def _clip_residuals(residuals: np.ndarray, q: float) -> np.ndarray:
    if q is None or q <= 0:
        return residuals
    q = float(q)
    if not (0.5 < q < 1.0):
        raise ValueError("--res_clip_q는 (0.5, 1.0) 범위여야 합니다. 예: 0.995, 0.99")
    lo_q = 1.0 - q
    hi_q = q

    res = np.asarray(residuals, dtype=float)
    out = res.copy()
    for j in range(out.shape[1]):
        lo = np.quantile(out[:, j], lo_q)
        hi = np.quantile(out[:, j], hi_q)
        out[:, j] = np.clip(out[:, j], lo, hi)
    return out


def build_residual_pool(
    args, df, X, y, model_factory,
    repeats: int = 1
) -> Tuple[pd.DataFrame, np.ndarray]:
    repeats = int(max(1, repeats))
    all_res = []
    metrics_last = None

    if args.residual_pool == "multi" and args.split == "time":
        print("[WARN] split=time 에서는 residual_pool=multi가 큰 의미가 없을 수 있습니다. (cutoff 고정)")

    for r in range(repeats):
        seed_r = int(args.seed + r)
        m = model_factory(seed_override=seed_r)

        _, metrics_df, residuals = train_validate(
            df, X, y, m,
            split=args.split, val_size=args.val_size,
            seed=seed_r, dt_col=args.dt_col, prod_col=args.prod_col,
            log_target=args.log_target
        )
        metrics_last = metrics_df
        all_res.append(residuals)

    res_pool = np.vstack(all_res) if len(all_res) > 1 else all_res[0]
    return metrics_last, res_pool


# =========================================================
# CLI main
# =========================================================
def main():
    ap = argparse.ArgumentParser(description="Leakage-safe Forecast (planned target default) + refit + MC")

    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--metrics_out", default=None)

    ap.add_argument("--prod_col", default=DEFAULT_PROD_COL)
    ap.add_argument("--dt_col", default=DEFAULT_DT_COL)

    ap.add_argument("--val_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--split", default="time", choices=["time", "group", "random"])
    ap.add_argument("--model", default="lgbm", choices=["tweedie", "lgbm"])
    ap.add_argument("--log_target", action="store_true")

    # ✅ y(타깃) 정의
    ap.add_argument("--target_kind", default="planned", choices=["planned", "expected", "both"],
                    help="planned(예정) | expected(예상) | both(예상+예정). 기본 planned 권장")
    ap.add_argument("--horizons", default="1,2,3,4",
                    help="예측할 horizon 인덱스. 예: '1,2,3,4' (기본: T+1~T+4). '0,1,2,3,4'도 가능")

    # ✅ feature에서 예상 수주량 활용 여부(기본: planned 타깃이면 True)
    ap.add_argument("--allow_expected_as_feature", action="store_true",
                    help="planned 타깃일 때, non-past '예상 수주량'을 feature로 허용(권장).")
    ap.add_argument("--disallow_expected_as_feature", action="store_true",
                    help="planned 타깃이어도 '예상 수주량'을 feature에서 제외(보수적).")

    # tuning
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--tune_seeds", type=int, default=1)
    ap.add_argument("--tune_seed_list", type=str, default=None)
    ap.add_argument("--tune_agg", type=str, default="mean", choices=["mean", "median"])
    ap.add_argument("--w_mae", type=float, default=0.5)
    ap.add_argument("--w_smape", type=float, default=0.5)

    ap.add_argument("--deterministic", action="store_true")

    ap.add_argument("--best_params_path", default="./configs/best_params.json")
    ap.add_argument("--save_best_params", action="store_true")

    # residual pool options
    ap.add_argument("--residual_pool", default="val", choices=["val", "multi"])
    ap.add_argument("--residual_runs", type=int, default=1)
    ap.add_argument("--res_clip_q", type=float, default=0.0)

    # MC options
    ap.add_argument("--mc_scenarios", type=int, default=0)
    ap.add_argument("--mc_out", default=None)
    ap.add_argument("--mc_mode", default="product", choices=["raw", "product"])
    ap.add_argument("--mc_out_csv", default=None)

    args = ap.parse_args()

    horizons = _parse_horizons(args.horizons)

    # allow_expected_as_feature 기본값 결정
    if args.target_kind == "planned":
        allow_expected_as_feature = True
        if args.disallow_expected_as_feature:
            allow_expected_as_feature = False
        if args.allow_expected_as_feature:
            allow_expected_as_feature = True
    else:
        allow_expected_as_feature = False

    df = pd.read_csv(args.inp)

    # ✅ 타깃 탐지 (작년/전년/지난해 자동 제외 + horizon 필터)
    target_cols = find_target_cols(df, target_kind=args.target_kind, horizons=horizons)
    if not target_cols:
        raise RuntimeError(f"Target columns not found. target_kind={args.target_kind}, horizons={horizons}")

    print(f"[INFO] target_kind={args.target_kind} | horizons={horizons} | targets={len(target_cols)}")
    if args.target_kind == "planned":
        print(f"[INFO] allow_expected_as_feature={allow_expected_as_feature}")

    X, y, num_cols, cat_cols, excluded = build_xy(
        df=df,
        prod_col=args.prod_col,
        target_cols=target_cols,
        log_target=args.log_target,
        target_kind=args.target_kind,
        allow_expected_as_feature=allow_expected_as_feature,
    )

    # -------------------------
    # (A) 튜닝 or load params
    # -------------------------
    best_params = None
    if args.tune:
        best_params = tune_params(args, df, X, y, num_cols, [args.prod_col], target_cols)
        if best_params and args.save_best_params:
            save_best_params(args.best_params_path, best_params)
    else:
        loaded = load_best_params(args.best_params_path)
        if loaded:
            best_params = loaded

    reg_n_jobs = 1 if args.deterministic else -1

    # -------------------------
    # (B) 모델 파라미터 구성
    # -------------------------
    if args.model == "lgbm":
        if lgb is None:
            raise RuntimeError("lightgbm 미설치. pip install lightgbm")

        bp = best_params or {}
        lgbm_params_base = dict(
            objective="tweedie",
            tweedie_variance_power=bp.get("power", bp.get("tweedie_variance_power", 1.3)),
            learning_rate=bp.get("lr", bp.get("learning_rate", 0.05)),
            n_estimators=bp.get("n_estimators", 1000),
            num_leaves=bp.get("num_leaves", 63),
            min_child_samples=bp.get("min_child_samples", 50),
            subsample=bp.get("subsample", 0.8),
            colsample_bytree=bp.get("colsample_bytree", 0.8),
            reg_lambda=bp.get("reg_lambda", 5.0),
            random_state=args.seed,
        )
        if args.deterministic:
            lgbm_params_base.update(dict(deterministic=True, force_row_wise=True))

        def model_factory(seed_override: Optional[int] = None) -> Pipeline:
            params = dict(lgbm_params_base)
            if seed_override is not None:
                params["random_state"] = int(seed_override)
            return build_model_pipeline(
                "lgbm",
                num_cols=num_cols,
                cat_cols=[args.prod_col],
                tweedie_power=params["tweedie_variance_power"],
                alpha=0.5,
                lgbm_params=params,
                reg_n_jobs=reg_n_jobs
            )

        print("🔧 Final params (base):", lgbm_params_base)

    else:
        bp = best_params or {}
        power = bp.get("power", 1.3)
        alpha = bp.get("alpha", 0.5)

        def model_factory(seed_override: Optional[int] = None) -> Pipeline:
            return build_model_pipeline("tweedie", num_cols, [args.prod_col], power, alpha, {}, reg_n_jobs)

        print("🔧 Final Tweedie params:", {"power": power, "alpha": alpha})

    # -------------------------
    # (C) 검증 metric + residual pool
    # -------------------------
    if args.residual_pool == "multi":
        metrics_df, residuals_pool = build_residual_pool(
            args, df, X, y, model_factory,
            repeats=args.residual_runs
        )
        print(f"[INFO] residual_pool=multi, runs={args.residual_runs}, pool_shape={residuals_pool.shape}")
    else:
        m = model_factory(seed_override=args.seed)
        _, metrics_df, residuals_pool = train_validate(
            df, X, y, m,
            split=args.split, val_size=args.val_size,
            seed=args.seed, dt_col=args.dt_col, prod_col=args.prod_col,
            log_target=args.log_target
        )
        print(f"[INFO] residual_pool=val, pool_shape={residuals_pool.shape}")

    residuals_pool = _clip_residuals(residuals_pool, args.res_clip_q)

    # ✅ horizon별 residual mean 제거 → MC 평균 바이어스 완화
    residuals_pool = residuals_pool - residuals_pool.mean(axis=0, keepdims=True)

    print("Validation metrics")
    print(metrics_df.to_string())

    if args.metrics_out:
        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(args.metrics_out, encoding="utf-8-sig")
        print(f"저장: {args.metrics_out}")

    # -------------------------
    # (D) 전체 데이터 refit 후 최종 예측
    # -------------------------
    final_model = model_factory(seed_override=args.seed)
    final_model.fit(X, y.values)
    print("[OK] Refit on FULL data for final predictions.")

    pred_all = predict_all(final_model, X, df, args.prod_col, args.dt_col, target_cols)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pred_all.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"예측 저장: {args.out}")

    # -------------------------
    # (E) MC 시나리오 생성
    # -------------------------
    if args.mc_scenarios and args.mc_scenarios > 0:
        products_for_save = None

        if args.mc_mode == "product":
            snap = snapshot_latest_by_product(
                pred_all,
                prod_col=args.prod_col,
                dt_col=args.dt_col,
                value_cols=target_cols,
            )
            d_hat = snap[target_cols].to_numpy(dtype=float)     # (P, D)
            products = snap[args.prod_col].astype(str).tolist()
            products_for_save = products

            scenarios = generate_demand_scenarios(
                d_hat=d_hat,
                residuals=residuals_pool.astype(float),
                n_scenarios=args.mc_scenarios,
                seed=args.seed,
            )  # (S, P, D)

            mc_out_csv = args.mc_out_csv if args.mc_out_csv else args.out.replace(".csv", "_mc.csv")
            scenarios_to_long_csv(
                scenarios=scenarios,
                products=products,
                out_csv=mc_out_csv,
                product_col_name=args.prod_col,
            )
            print(f"MC scenarios (long CSV) saved: {mc_out_csv}")

        else:
            d_hat = pred_all[target_cols].to_numpy(dtype=float)
            scenarios = generate_demand_scenarios(
                d_hat=d_hat,
                residuals=residuals_pool.astype(float),
                n_scenarios=args.mc_scenarios,
                seed=args.seed,
            )
            print("[INFO] mc_mode=raw: generated scenarios over raw rows (no long CSV).")

        mc_out = args.mc_out if args.mc_out is not None else args.out.replace(".csv", "_mc.npz")
        Path(mc_out).parent.mkdir(parents=True, exist_ok=True)

        save_kwargs = dict(
            scenarios=scenarios,
            target_cols=np.array(target_cols),
            prod_col=args.prod_col,
            dt_col=args.dt_col,
            mc_mode=args.mc_mode,
            target_kind=args.target_kind,
        )
        if products_for_save is not None:
            save_kwargs["products"] = np.array(products_for_save, dtype=object)

        np.savez_compressed(mc_out, **save_kwargs)
        print(f"MC scenarios saved: {mc_out}")

    # =====================================================
    # ✅ planner 입력용 제품단위 예측 저장: 최신 스냅샷 기반으로 통일
    # =====================================================
    prod_snap = snapshot_latest_by_product(
        pred_all,
        prod_col=args.prod_col,
        dt_col=args.dt_col,
        value_cols=target_cols,
    )
    prod_agg_path = args.out.replace(".csv", "_by_product.csv")
    prod_snap.to_csv(prod_agg_path, index=False, encoding="utf-8-sig")
    print(f"제품별 최신 스냅샷 저장(=planner input): {prod_agg_path}")


if __name__ == "__main__":
    main()