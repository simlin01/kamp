#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, os, pandas as pd
from config import Config
from src import utils
from src import forecast as F
from src import planner as P
from src import planner_opt as POPT
from src import report_llm as RLLM

DEFAULT_PROD_COL = "Product_Number"
DEFAULT_DT_COL = "DateTime"

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="features.py 출력 CSV (feat.csv)")
    ap.add_argument("--out_dir", type=str, default="./outputs")

    # ✅ 모델 선택은 forecast.py 규격으로 정규화 (tweedie | lgbm)
    ap.add_argument("--model", type=str, default="lgbm",
                    choices=["linear", "xgboost", "lgbm", "elastic", "tweedie"])

    ap.add_argument("--encoding", type=str, default="utf-8")
    ap.add_argument("--daily_capacity", type=int, default=5000)
    ap.add_argument("--min_lot_size", type=int, default=0)
    ap.add_argument("--safety_stock", type=int, default=0)

    # Planner / Reporter
    ap.add_argument("--planner", type=str, default="ortools",
                    choices=["heuristic", "ortools"])
    ap.add_argument("--reporter", type=str, default="llm",
                    choices=["template", "llm"])

    # OR-Tools 옵션
    ap.add_argument("--int_production", action="store_true",
                    help="생산량을 정수로 강제")

    # 🔧 (구)옵션 — 현재 버전에서 미사용. 호환성 유지를 위해 경고만 출력
    ap.add_argument("--tune", action="store_true", help="(미사용) Optuna 튜닝")
    ap.add_argument("--use_zeromodel", action="store_true", help="(미사용) 제로 인플레 2단계")
    ap.add_argument("--quantiles", nargs="*", type=float, default=[], help="(미사용) 분위수")

    # 검증/누출 방지 관련
    ap.add_argument("--split", type=str, default="time", choices=["random", "time", "group"],
                    help="검증 분할 방식: time(권장), group, random")
    ap.add_argument("--prod_col", type=str, default=DEFAULT_PROD_COL)
    ap.add_argument("--dt_col", type=str, default=DEFAULT_DT_COL)
    ap.add_argument("--val_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    # Tweedie/LGBM 하이퍼파라미터
    ap.add_argument("--tweedie_power", type=float, default=1.3)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--lgbm_estimators", type=int, default=800)
    ap.add_argument("--lgbm_lr", type=float, default=0.05)
    ap.add_argument("--lgbm_leaves", type=int, default=63)
    ap.add_argument("--lgbm_min_child", type=int, default=50)
    ap.add_argument("--lgbm_subsample", type=float, default=0.8)
    ap.add_argument("--lgbm_colsample", type=float, default=0.8)
    ap.add_argument("--lgbm_lambda", type=float, default=5.0)
    ap.add_argument("--lgbm_power", type=float, default=1.3)

    # 리포트/메트릭 저장
    ap.add_argument("--metrics_out", type=str, default=None)
    ap.add_argument("--feature_report", type=str, default=None)
    ap.add_argument("--llm_model", type=str, default="gpt-4o-mini")
    ap.add_argument("--max_chars", type=int, default=16000, help="CSV 요약 입력 길이 컷")
    return ap.parse_args()


def normalize_model_name(m: str) -> str:
    """main의 모델명을 forecast.py 규격으로 매핑."""
    m = (m or "").lower()
    if m in ["tweedie", "linear", "elastic"]:
        if m in ["linear", "elastic"]:
            print("⚠️ --model", m, "→ 'tweedie'로 매핑합니다.")
        return "tweedie"
    if m in ["lgbm", "xgboost"]:
        if m == "xgboost":
            print("⚠️ --model xgboost는 현재 구현에서 'lgbm'으로 매핑합니다.")
        return "lgbm"
    # 기본
    return "tweedie"


def main():
    args = parse_args()
    cfg = Config(
        daily_capacity=args.daily_capacity,
        min_lot_size=args.min_lot_size,
        safety_stock=args.safety_stock,
        model_type=normalize_model_name(args.model),
        encoding=args.encoding,
    )
    utils.ensure_dir(args.out_dir)

    # 📝 미사용 옵션 경고
    if args.tune: print("⚠️ --tune: 현재 버전에서 미사용(무시).")
    if args.use_zeromodel: print("⚠️ --use_zeromodel: 현재 버전에서 미사용(무시).")
    if args.quantiles: print("⚠️ --quantiles: 현재 버전에서 미사용(무시).")

    # 1) Load data (feat.csv 권장)
    df = utils.load_data(args.data, encoding=cfg.encoding)
    df.columns = [c.strip() for c in df.columns]

    # 2) 타깃 자동 탐색 (예상 수주량)
    target_cols = F.find_target_cols(df, ["예상 수주량"])
    if len(target_cols) == 0:
        raise RuntimeError("타깃(예상 수주량) 컬럼이 없습니다. 열 이름에 '예상 수주량'이 포함되어야 합니다.")
    print("🎯 Targets:", target_cols)

    # 3) X, y 구성(누출 방지: 미래 '예정 수주량' & 모든 '예상 수주량'은 X에서 제외)
    X, y, num_cols, cat_cols, excluded = F.build_xy(df, prod_col=args.prod_col, target_cols=target_cols)

    # 4) 모델 파이프라인
    model_name = normalize_model_name(args.model)
    lgbm_params = dict(
        tweedie_variance_power=args.lgbm_power,
        n_estimators=args.lgbm_estimators,
        learning_rate=args.lgbm_lr,
        num_leaves=args.lgbm_leaves,
        min_child_samples=args.lgbm_min_child,
        subsample=args.lgbm_subsample,
        colsample_bytree=args.lgbm_colsample,
        reg_lambda=args.lgbm_lambda,
        random_state=args.seed,
    )
    model = F.build_model_pipeline(
        model_name=model_name,
        num_cols=num_cols,
        cat_cols=cat_cols,
        tweedie_power=args.tweedie_power,
        alpha=args.alpha,
        lgbm_params=lgbm_params
    )

    # 5) Train & Validate (split: time/group/random)
    model, metrics_df, X_va, y_va = F.train_validate(
        df_raw=df, X=X, y=y, model=model,
        split=args.split, val_size=args.val_size, seed=args.seed,
        dt_col=args.dt_col, prod_col=args.prod_col
    )
    print("\n📊 Validation metrics (per target)")
    print(metrics_df.to_string())

    metrics_path = os.path.join(args.out_dir, "forecast_validation_metrics.csv")
    if args.metrics_out:
        metrics_path = args.metrics_out
    utils.save_csv(metrics_df, metrics_path)

    # 6) Feature report (선택)
    if args.feature_report:
        pd.DataFrame({"used_numeric_features": pd.Series(num_cols)}).to_csv(
            args.feature_report, index=False, encoding="utf-8-sig"
        )
        pd.DataFrame({"excluded_features": pd.Series(excluded)}).to_csv(
            args.feature_report.replace(".csv", "_excluded.csv"), index=False, encoding="utf-8-sig"
        )
        print(f"💾 피처 리포트 저장: {args.feature_report} / {args.feature_report.replace('.csv', '_excluded.csv')}")

    # 7) 전체 예측 → 제품별 집계
    pred_df = F.predict_all(model, X_all=X, prod_col=args.prod_col, target_cols=target_cols)
    forecast_by_product = F.aggregate_by_product(pred_df, args.prod_col)

    forecast_path = os.path.join(args.out_dir, "forecast_by_product.csv")
    utils.save_csv(forecast_by_product, forecast_path)

    # 8) Planning
    if args.planner == "heuristic":
        plan_df = P.plan_production(
            forecast_by_product=forecast_by_product,
            horizons=target_cols,
            prod_col=args.prod_col,
            daily_capacity=cfg.daily_capacity,
            min_lot_size=cfg.min_lot_size,
            safety_stock=cfg.safety_stock
        )
    else:
        plan_df = POPT.optimize_plan(
            forecast_by_product=forecast_by_product,
            horizons=target_cols,
            prod_col=args.prod_col,
            daily_capacity=cfg.daily_capacity,
            min_lot_size=cfg.min_lot_size,
            safety_stock=cfg.safety_stock,
            int_production=args.int_production
        )
    plan_path = os.path.join(args.out_dir, "production_plan.csv")
    utils.save_csv(plan_df, plan_path)

    # 9) Reporting
    if args.reporter == "template":
        raise NotImplementedError("템플릿 리포터는 현재 주석 처리되어 있습니다. --reporter llm을 사용하세요.")
    else:
        report_txt = RLLM.build_report_with_llm(
            plan_csv=plan_path,
            forecast_csv=forecast_path,
            metrics_csv=metrics_path,
            model_name=args.llm_model,
            max_chars=args.max_chars
        )
    report_path = os.path.join(args.out_dir, "weekly_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_txt)

    print("[✅ Done]")
    print(f"- Forecast metrics: {metrics_path}")
    print(f"- Forecast by product: {forecast_path}")
    print(f"- Production plan: {plan_path}")
    print(f"- Weekly report: {report_path}")


if __name__ == "__main__":
    main()