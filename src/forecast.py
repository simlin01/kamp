#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
forecast.py — Multi-Target Tweedie/LGBM Forecast (feat.csv → predict)
- 누출 방지: 'T+숫자일 예정 수주량' 완전 제외
- DateTime/제품키 포함 저장
- (--tune, --trials) 하이퍼파라미터 튜닝 지원 (Optuna 있으면 사용)
- 학습에 실제 적용된 파라미터 요약 출력
- 완전 재현용 결정론 모드(--deterministic) 옵션 제공
- 튜닝 결과를 JSON으로 저장/로드(--best_params_path, --save_best_params)

CLI - 기본:
python src/forecast.py \
  --in ./data/feat.csv \
  --out ./outputs/pred_final.csv \
  --metrics_out ./outputs/metrics_final.csv \
  --prod_col Product_Number \
  --dt_col DateTime \
  --model lgbm \
  --split time \
  --val_size 0.2 \
  --seed 2025 \
  --best_params_path ./configs/best_params.json

CLI - Tune:
python src/forecast.py \
  --in ./data/feat.csv \
  --out ./outputs/pred_tuned.csv \
  --metrics_out ./outputs/metrics_tuned.csv \
  --prod_col Product_Number \
  --dt_col DateTime \
  --model lgbm \
  --split time \
  --val_size 0.2 \
  --seed 2025 \
  --tune \
  --trials 50 \
  --save_best_params \
  --best_params_path ./configs/best_params.json

CLI - deterministic:
python src/forecast.py \
  --in ./data/feat.csv \
  --out ./outputs/pred_deterministic.csv \
  --metrics_out ./outputs/metrics_deterministic.csv \
  --prod_col Product_Number \
  --dt_col DateTime \
  --model lgbm \
  --split time \
  --val_size 0.2 \
  --seed 2025 \
  --best_params_path ./configs/best_params.json \
  --deterministic

"""

from typing import List, Tuple, Dict
import argparse, re, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import TweedieRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score, average_precision_score, precision_recall_fscore_support
import json
from pathlib import Path

# =============== Helper for params save/load ===============
def load_best_params(path: str | Path) -> Dict | None:
    try:
        p = Path(path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                params = json.load(f)
            print(f"📥 Loaded best params from: {p}")
            return params
    except Exception as e:
        print(f"⚠️ Failed to load best params ({path}): {e}")
    return None

def save_best_params(path: str | Path, params: Dict) -> None:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved best params to: {p}")
    except Exception as e:
        print(f"⚠️ Failed to save best params ({path}): {e}")

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
TARGET_KEYWORDS  = ["예상 수주량"]

# =============== Target / Feature utils ===============
def find_target_cols(df: pd.DataFrame, keywords: List[str]) -> List[str]:
    cols = [c for c in df.columns if any(k in c for k in keywords)]
    def _key(x: str):
        m = re.search(r"T\+?(\d*)일", x)
        return int(m.group(1) or 0) if m else 0
    return sorted(cols, key=_key)

def select_feature_columns(df: pd.DataFrame, prod_col: str, target_cols: List[str]) -> Tuple[List[str], List[str]]:
    numeric_all = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    excluded: List[str] = []

    def is_future_plan(col: str) -> bool:
        if "예정 수주량" in col:
            if any(tag in col for tag in ["작년", "전년", "지난해"]):
                return False
            return True
        return False

    def is_any_prediction(col: str) -> bool:
        return ("예상 수주량" in col)

    for c in list(numeric_all):
        if c in target_cols or c == prod_col or is_future_plan(c) or is_any_prediction(c):
            excluded.append(c)

    num_cols = [c for c in numeric_all if c not in set(excluded + [prod_col])]
    return num_cols, excluded

def build_xy(df: pd.DataFrame, prod_col: str, target_cols: List[str], log_target: bool=False):
    if prod_col not in df.columns:
        raise ValueError(f"'{prod_col}' 컬럼이 없습니다.")
    y = df[target_cols].astype(float).clip(lower=0)
    if log_target: y = np.log1p(y)
    num_cols, excluded = select_feature_columns(df, prod_col, target_cols)
    cat_cols = [prod_col]
    if len(num_cols)==0: raise RuntimeError("사용 가능한 수치형 피처가 없습니다. features.py를 확인하세요.")
    X = df[num_cols + cat_cols].copy()
    print(f"🧩 Features used: {len(num_cols)} numeric + {len(cat_cols)} categorical")
    if excluded: print(f"🚫 Excluded (leakage/targets): {len(excluded)} cols")
    return X, y, num_cols, cat_cols, excluded

# =============== Metrics ===============
def binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    y_true_bin = (y_true > 0).astype(int)
    def _safe(f, *args, **kwargs):
        try: return f(*args, **kwargs)
        except Exception: return np.nan
    auc = _safe(roc_auc_score, y_true_bin, y_score)
    ap  = _safe(average_precision_score, y_true_bin, y_score)
    uniq = np.unique(y_score)
    if len(uniq) > 200:
        uniq = np.unique(np.quantile(y_score, np.linspace(0,1,200)))
    best = {"f1":-1.0,"p":np.nan,"r":np.nan,"thr":0.0}
    for thr in uniq:
        y_pred = (y_score >= thr).astype(int)
        p,r,f,_ = precision_recall_fscore_support(y_true_bin,y_pred,average="binary",zero_division=0)
        if f>best["f1"]: best={"f1":float(f),"p":float(p),"r":float(r),"thr":float(thr)}
    return {"AUC":float(auc),"AP":float(ap),"F1":best["f1"],"Precision":best["p"],"Recall":best["r"],"BestThreshold":best["thr"]}

def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))

def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + eps)))

# =============== Splits ===============
def time_split(df_raw: pd.DataFrame, X: pd.DataFrame, y: pd.DataFrame, dt_col: str, val_ratio: float):
    dt = pd.to_datetime(df_raw[dt_col])
    cutoff = dt.quantile(1 - val_ratio)
    idx_tr, idx_va = (dt <= cutoff), (dt > cutoff)
    return X[idx_tr], X[idx_va], y[idx_tr], y[idx_va]

def group_split(df_raw: pd.DataFrame, X: pd.DataFrame, y: pd.DataFrame, group_col: str, val_ratio: float, seed: int):
    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    tr_idx, va_idx = next(gss.split(X, y, groups=df_raw[group_col]))
    return X.iloc[tr_idx], X.iloc[va_idx], y.iloc[tr_idx], y.iloc[va_idx]

# =============== Pipeline ===============
def build_model_pipeline(model_name: str, num_cols: List[str], cat_cols: List[str],
                         tweedie_power: float, alpha: float, lgbm_params: Dict,
                         reg_n_jobs: int = -1) -> Pipeline:
    if model_name == "tweedie":
        base = MultiOutputRegressor(
            TweedieRegressor(power=tweedie_power, alpha=alpha, link="log", max_iter=2000),
            n_jobs=reg_n_jobs
        )
        prep = ColumnTransformer(
            [("num", RobustScaler(), num_cols), ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)]
        )
        return Pipeline([("prep", prep), ("reg", base)])
    elif model_name == "lgbm":
        if lgb is None: raise RuntimeError("lightgbm 미설치. pip install lightgbm")
        base = MultiOutputRegressor(
            lgb.LGBMRegressor(**lgbm_params, n_jobs=reg_n_jobs),
            n_jobs=reg_n_jobs
        )
        prep = ColumnTransformer(
            [("num", "passthrough", num_cols), ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)]
        )
        return Pipeline([("prep", prep), ("reg", base)])
    else:
        raise ValueError("지원하지 않는 모델 이름")

# =============== Train / Validate / Predict ===============
def train_validate(df_raw, X, y, model, split="time", val_size=0.2, seed=2025, dt_col=DEFAULT_DT_COL, prod_col=DEFAULT_PROD_COL, log_target=False):
    if split=="time":
        X_tr, X_va, y_tr, y_va = time_split(df_raw, X, y, dt_col, val_size)
    elif split=="group":
        X_tr, X_va, y_tr, y_va = group_split(df_raw, X, y, prod_col, val_size, seed)
    else:
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=val_size, random_state=seed)

    model.fit(X_tr, y_tr.values)
    pred = np.maximum(0.0, np.asarray(model.predict(X_va), dtype=float))
    if log_target:
        pred, y_va = np.expm1(pred), np.expm1(y_va)

    rows = {}
    for i, t in enumerate(y.columns):
        yt = y_va[t].values
        pt = pred[:, i]
        rows[t] = {
            "MAE": mean_absolute_error(yt, pt),
            "R2":  r2_score(yt, pt),
            "SMAPE": smape(yt, pt),
            "MAPE": mape(yt, pt),
            **binary_metrics(yt, pt)
        }
    return model, pd.DataFrame(rows).T

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
    tcols = [c for c in pred_df.columns if c not in [prod_col, DEFAULT_DT_COL]]
    return pred_df.groupby(prod_col)[tcols].mean(numeric_only=True).reset_index()

# =============== Tuning ===============
def average_mae(metrics_df, target_cols, emphasize=None):
    w = {t:1.0 for t in target_cols}
    if emphasize:
        for k,v in emphasize.items():
            if k in w: w[k]=float(v)
    total_w = sum(w.values())
    return float((metrics_df.loc[target_cols, "MAE"] * pd.Series(w)).sum() / total_w)

def tune_params(args, df, X, y, num_cols, cat_cols, target_cols):
    if optuna is None:
        print("⚠️ Optuna 미설치: 튜닝을 건너뜁니다.")
        return None
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
        model = build_model_pipeline("lgbm", num_cols, cat_cols,
                                     tweedie_power=params["tweedie_variance_power"],
                                     alpha=0.5, lgbm_params=params)
        _, mdf = train_validate(df, X, y, model, split=args.split, val_size=args.val_size,
                                seed=args.seed, dt_col=args.dt_col, prod_col=args.prod_col,
                                log_target=args.log_target)
        return average_mae(mdf, target_cols)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
    print("🏆 Best params:", study.best_params)
    return study.best_params

# =============== CLI ===============
def main():
    ap = argparse.ArgumentParser(description="Leakage-safe Forecast")
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
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--best_params_path", default="./configs/best_params.json")
    ap.add_argument("--save_best_params", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    target_cols = find_target_cols(df, TARGET_KEYWORDS)
    X, y, num_cols, cat_cols, excluded = build_xy(df, args.prod_col, target_cols, args.log_target)

    best_params = None
    if args.tune:
        best_params = tune_params(args, df, X, y, num_cols, cat_cols, target_cols)
        if best_params and args.save_best_params:
            save_best_params(args.best_params_path, best_params)
    else:
        loaded = load_best_params(args.best_params_path)
        if loaded: best_params = loaded

    reg_n_jobs = 1 if args.deterministic else -1
    if args.model == "lgbm":
        bp = best_params or {}
        lgbm_params = dict(
            objective="tweedie",
            tweedie_variance_power=bp.get("power", 1.3),
            learning_rate=bp.get("lr", 0.05),
            n_estimators=bp.get("n_estimators", 1000),
            num_leaves=bp.get("num_leaves", 63),
            min_child_samples=bp.get("min_child_samples", 50),
            subsample=bp.get("subsample", 0.8),
            colsample_bytree=bp.get("colsample_bytree", 0.8),
            reg_lambda=bp.get("reg_lambda", 5.0),
            random_state=args.seed,
            deterministic=args.deterministic,
            force_row_wise=args.deterministic,
        )
        model = build_model_pipeline("lgbm", num_cols, cat_cols,
                                     lgbm_params["tweedie_variance_power"], 0.5, lgbm_params, reg_n_jobs)
        print("🔧 Final params:", lgbm_params)
    else:
        power = (best_params or {}).get("power", 1.3)
        alpha = (best_params or {}).get("alpha", 0.5)
        model = build_model_pipeline("tweedie", num_cols, cat_cols, power, alpha, {}, reg_n_jobs)
        print("🔧 Final Tweedie params:", {"power": power, "alpha": alpha})

    model, metrics_df = train_validate(df, X, y, model, split=args.split, val_size=args.val_size,
                                       seed=args.seed, dt_col=args.dt_col, prod_col=args.prod_col,
                                       log_target=args.log_target)
    print("📊 Validation metrics")
    print(metrics_df.to_string())

    if args.metrics_out:
        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(args.metrics_out, encoding="utf-8-sig")
        print(f"💾 저장: {args.metrics_out}")

    pred_all = predict_all(model, X, df, args.prod_col, args.dt_col, target_cols)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pred_all.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"💾 예측 저장: {args.out}")

    prod_agg = aggregate_by_product(pred_all, args.prod_col)
    prod_agg.to_csv(args.out.replace(".csv","_by_product.csv"), index=False, encoding="utf-8-sig")
    print(f"💾 제품별 평균 저장: {args.out.replace('.csv','_by_product.csv')}")

if __name__ == "__main__":
    main()