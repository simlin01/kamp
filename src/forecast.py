#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
forecast.py — Multi-Target Tweedie/LGBM Forecast (feat.csv → predict)

[핵심 수정(정합성/안정성)]
1) ✅ Target 탐지 통일: "예상/예정 수주량" 모두 지원 + horizon 정렬 강건화
   - utils.py의 규칙과 동일한 정규식 기반 파싱으로 통일

2) ✅ Leakage 방지 규칙 보강:
   - is_future_plan / is_any_prediction 에서 "예정/예상" 모두 제외
   - 작년/전년/지난해는 예외 처리 유지

3) ✅ planner 입력(pred_final_by_product.csv)과 MC 기준(d_hat)의 "시점 기준"을 일치
   - snapshot_latest_by_product()를 기본으로 사용
   - aggregate_by_product()는 참고용 유지

4) ✅ MC 시나리오: horizon 상관 보존(벡터 resample) + 평균 바이어스 완화(센터링)
   - residual clip 후 horizon별 평균 0으로 center

(그 외 로직/CLI/파일 포맷은 최대한 그대로 유지)
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Optional
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

# ✅ 예상/예정 모두 기본 키워드로
TARGET_KEYWORDS  = ["예상 수주량", "예정 수주량"]


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
# Target / Feature utils
# =========================================================
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()

def _extract_horizon_index(col: str) -> Optional[int]:
    """
    - "T일 ..." -> 0
    - "T+3일 ..." -> 3
    - 축약형 "T" -> 0
    - 축약형 "T+3" -> 3
    """
    c = _norm(col)
    if re.fullmatch(r"T", c):
        return 0
    m = re.fullmatch(r"T\+(\d+)", c)
    if m:
        return int(m.group(1))
    if "T일" in c:
        return 0
    m2 = re.search(r"T\+(\d+)\s*일", c)
    if m2:
        return int(m2.group(1))
    m3 = re.search(r"T\+(\d+)", c)
    if m3:
        return int(m3.group(1))
    return None

def find_target_cols(df: pd.DataFrame, keywords: List[str]) -> List[str]:
    """
    keywords(예: ["예상 수주량","예정 수주량"]) 중 하나라도 포함 + T일/T+ 포함 컬럼을 타깃으로 간주.
    horizon index 기준으로 안정 정렬.
    """
    cands = []
    for c in df.columns:
        cc = _norm(c)
        if any(k in cc for k in keywords) and (("T일" in cc) or ("T+" in cc) or re.search(r"\bT(\+\d+)?\b", cc)):
            cands.append(c)

    with_h = []
    for c in cands:
        h = _extract_horizon_index(c)
        if h is not None:
            with_h.append((h, c))
    no_h = [c for c in cands if _extract_horizon_index(c) is None]

    out = [c for h, c in sorted(with_h, key=lambda x: x[0])] + no_h
    return out

def select_feature_columns(df: pd.DataFrame, prod_col: str, target_cols: List[str]) -> Tuple[List[str], List[str]]:
    numeric_all = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    excluded: List[str] = []

    def is_future_plan(col: str) -> bool:
        # ✅ "예정/예상" 둘 다 미래 계획/예측 계열로 간주 (누출 방지)
        if ("예정 수주량" in col) or ("예상 수주량" in col):
            if any(tag in col for tag in ["작년", "전년", "지난해"]):
                return False
            return True
        return False

    for c in list(numeric_all):
        if c in target_cols or c == prod_col or is_future_plan(c):
            excluded.append(c)

    num_cols = [c for c in numeric_all if c not in set(excluded + [prod_col])]
    return num_cols, excluded

def build_xy(df: pd.DataFrame, prod_col: str, target_cols: List[str], log_target: bool=False):
    if prod_col not in df.columns:
        raise ValueError(f"'{prod_col}' 컬럼이 없습니다.")
    y = df[target_cols].astype(float).clip(lower=0)
    if log_target:
        y = np.log1p(y)

    num_cols, excluded = select_feature_columns(df, prod_col, target_cols)
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
    residuals: (m, k)  (y_true - y_pred), horizon별 mean=0 (권장)
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

def aggregate_by_product(pred_df, prod_col):
    # (참고용) 전체 기간 평균 집계. planner/MC 일관성을 위해 기본 사용은 권장하지 않음.
    tcols = [c for c in pred_df.columns if c not in [prod_col, DEFAULT_DT_COL]]
    return pred_df.groupby(prod_col)[tcols].mean(numeric_only=True).reset_index()


# =========================================================
# Tuning helpers
# =========================================================
def average_metric(metrics_df: pd.DataFrame, target_cols: List[str], metric: str, emphasize: Optional[Dict[str, float]]=None) -> float:
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
    out = []
    for p in parts:
        out.append(int(p))
    return out

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
    w_sm  = float(args.w_smape)
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
# Residual pool (OPTION) + residual clip
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
    ap = argparse.ArgumentParser(description="Leakage-safe Forecast (multi-seed tuning + MAE/SMAPE + refit + MC)")
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
    ap.add_argument("--mc_mode", default="raw", choices=["raw", "product"])
    ap.add_argument("--mc_out_csv", default=None)

    args = ap.parse_args()

    df = pd.read_csv(args.inp)

    target_cols = find_target_cols(df, TARGET_KEYWORDS)
    if not target_cols:
        raise RuntimeError(f"Target columns not found. keywords={TARGET_KEYWORDS}")

    X, y, num_cols, cat_cols, excluded = build_xy(df, args.prod_col, target_cols, args.log_target)

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