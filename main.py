import argparse, os, pandas as pd
from config import Config
from src import utils
from src import forecast as F
from src import planner as P
from src import planner_opt as PORT
# from src import report as R
from src import planner_opt as POPT
from src import report_llm as RLLM

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="./outputs")
    ap.add_argument("--model", type=str, default="linear", choices=["linear", "xgboost"])
    ap.add_argument("--encoding", type=str, default="utf-8")
    ap.add_argument("--daily_capacity", type=int, default=5000)
    ap.add_argument("--min_lot_size", type=int, default=0)
    ap.add_argument("--safety_stock", type=int, default=0)

    # 새로 추가된 스위치
    ap.add_argument("--planner", type=str, default="heuristic", choices=["heuristic", "ortools"])
    ap.add_argument("--reporter", type=str, default="template", choices=["template", "llm"])

    # OR-Tools 옵션 예시
    ap.add_argument("--int_production", action="store_true", help="생산량을 정수로 강제")

    # LLM 옵션 예시
    ap.add_argument("--llm_model", type=str, default="gpt-4o-mini")
    ap.add_argument("--max_chars", type=int, default=16000, help="CSV 요약 입력 길이 컷")
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = Config(
        daily_capacity=args.daily_capacity,
        min_lot_size=args.min_lot_size,
        safety_stock=args.safety_stock,
        model_type=args.model,
        encoding=args.encoding,
    )
    utils.ensure_dir(args.out_dir)

    # 1) Load
    df = utils.load_data(args.data, encoding=cfg.encoding)
    df.columns = [c.strip() for c in df.columns]
    prod_col, target_cols, last_year_cols, numeric_covars = utils.detect_columns(df)
    if len(target_cols) == 0:
        raise RuntimeError("예측 대상 컬럼('T일 예정 수주량' / 'T+N일 예정 수주량')을 찾지 못했습니다.")
    df = df.dropna(subset=target_cols, how="all").reset_index(drop=True)

    # 2) Features
    X_all, y_all, num_cols, cat_cols = F.build_features(df, prod_col, target_cols, last_year_cols, numeric_covars)

    # 3) Train & Validate
    models, metrics_df, _, _ = F.train_and_validate(X_all, y_all, num_cols, cat_cols, model_type=cfg.model_type)
    metrics_path = os.path.join(args.out_dir, "forecast_validation_metrics.csv")
    utils.save_csv(metrics_df, metrics_path)

    # 4) Forecast → by product
    pred_df = F.forecast_all(models, X_all, prod_col)
    forecast_by_product = F.aggregate_by_product(pred_df, prod_col)
    forecast_path = os.path.join(args.out_dir, "forecast_by_product.csv")
    utils.save_csv(forecast_by_product, forecast_path)

    # 5) Planning (선택)
    if args.planner == "heuristic":
        plan_df = P.plan_production(
            forecast_by_product=forecast_by_product,
            horizons=target_cols,
            prod_col=prod_col,
            daily_capacity=cfg.daily_capacity,
            min_lot_size=cfg.min_lot_size,
            safety_stock=cfg.safety_stock
        )
    else:
        # OR-Tools 최적화
        plan_df = POPT.optimize_plan(
            forecast_by_product=forecast_by_product,
            horizons=target_cols,
            prod_col=prod_col,
            daily_capacity=cfg.daily_capacity,
            min_lot_size=cfg.min_lot_size,
            safety_stock=cfg.safety_stock,
            int_production=args.int_production
        )
    plan_path = os.path.join(args.out_dir, "production_plan.csv")
    utils.save_csv(plan_df, plan_path)

    # 6) Reporting (선택)
    if args.reporter == "template":
        report_txt = R.build_report_text(
            plan_df=plan_df,
            prod_col=prod_col,
            metrics_df=metrics_df,
            max_days=len(target_cols),
            daily_capacity=cfg.daily_capacity
        )
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

    print("[OK] Done")
    print(f"- {metrics_path}")
    print(f"- {forecast_path}")
    print(f"- {plan_path}")        # ✅ 수정
    print(f"- {report_path}")

if __name__ == "__main__":
    main()
