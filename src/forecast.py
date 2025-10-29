from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None  # 선택 설치

def build_features(
    df: pd.DataFrame,
    prod_col: str,
    target_cols: List[str],
    last_year_cols: List[str],
    numeric_covars: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    # 피처: 전년도 수주량 + 기타 수치 피처
    feature_cols = list(last_year_cols) + list(numeric_covars)

    # 만약 피처가 전무하면(최소구동) 타깃 평균으로 대체 피처 생성
    if len(feature_cols) == 0:
        for c in target_cols:
            df[f"mean_{c}"] = df[c].fillna(df[c].mean())
        feature_cols = [f"mean_{c}" for c in target_cols]

    X = df[feature_cols + [prod_col]].copy()
    y = df[target_cols].copy()
    num_cols = [c for c in feature_cols if np.issubdtype(df[c].dtype, np.number)]
    cat_cols = [prod_col]
    return X, y, num_cols, cat_cols

def get_model(model_type: str):
    if model_type == "xgboost":
        if XGBRegressor is None:
            raise RuntimeError("xgboost가 설치되어 있지 않습니다. pip install xgboost 후 사용하세요.")
        return XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
    # default: linear
    return LinearRegression()

def train_and_validate(
    X: pd.DataFrame,
    y: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    model_type: str = "linear",
    val_size: float = 0.2,
    random_state: int = 42
) -> Tuple[Dict[str, Pipeline], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=val_size, random_state=random_state)

    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop"
    )

    models: Dict[str, Pipeline] = {}
    metrics = {}
    for t in y.columns:
        model = get_model(model_type)
        pipe = Pipeline([("prep", preprocess), ("reg", model)])
        pipe.fit(X_tr, y_tr[t].values)
        pred = pipe.predict(X_va)
        metrics[t] = {
            "MAE": float(mean_absolute_error(y_va[t].values, pred)),
            "R2": float(r2_score(y_va[t].values, pred)),
        }
        models[t] = pipe

    metrics_df = pd.DataFrame(metrics).T
    return models, metrics_df, X_va, y_va

def forecast_all(
    models: Dict[str, Pipeline],
    X_all: pd.DataFrame,
    prod_col: str
) -> pd.DataFrame:
    out = {t: models[t].predict(X_all) for t in models.keys()}
    pred = pd.DataFrame(out)
    pred[prod_col] = X_all[prod_col].values
    # 음수는 0으로 클리핑
    for c in pred.columns:
        if c != prod_col:
            pred[c] = np.maximum(0.0, pred[c].astype(float))
    return pred

def aggregate_by_product(pred_df: pd.DataFrame, prod_col: str) -> pd.DataFrame:
    return pred_df.groupby(prod_col).mean(numeric_only=True).reset_index()
