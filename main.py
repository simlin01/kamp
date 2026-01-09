#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — End-to-end SCM planning pipeline (Features → Forecast → Planning → Metrics → Evaluator → Report)

CLI 예시:
python main.py \
  --data ./data/data.csv \
  --out_dir ./outputs \
  --prod_col Product_Number \
  --daily_capacity 5000 \
  --int_production \
  --model gpt-4o-mini \
  --mc_scenarios 200 \
  --mc_alpha 0.9 \
  --mc_seed 2025
"""

from __future__ import annotations

import os, sys
import json
import argparse
from datetime import datetime
import pandas as pd
import numpy as np

# src 패키지 import (루트에 main.py가 있고, 모듈은 src/ 폴더에 있는 구조)
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.append(os.path.abspath("."))

from src import features as FE
from src import forecast as FO
from src import planner_opt as PO
from src import metrics as M
from src import evaluator as EV
from src import report_llm as RL


# =========================================================
# 유틸
# =========================================================
def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def save_json(path: str, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _now_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# =========================================================
# Forecast helper (Python API)
# - main.py에서 forecast를 "직접 호출"하는 이유:
#   * subprocess 대신 residual / validation 정보를 잡아 MC 생성에 사용 가능
# =========================================================
def _run_forecast_python_api(
    feat_csv: str,
    out_pred_csv: str,
    out_metrics_csv: str,
    out_pred_by_product_csv: str,
    prod_col: str,
    dt_col: str | None,
    best_params_path: str,
    seed: int = 2025,
    split: str = "time",
    val_size: float = 0.2,
    deterministic: bool = False,
    log_target: bool = False,
):
    """
    forecast.py를 subprocess가 아닌 Python API로 실행.

    반환:
      pred_all_df: (row-level) 예측 결과 + [prod_col, dt_col]
      metrics_df: horizon별 metrics
      residual_df_val: validation row-level residual + [prod_col, dt_col]
      target_cols: horizon 컬럼 목록
      dt_use: 사용된 dt column
    """
    df = pd.read_csv(feat_csv)

    target_cols = FO.find_target_cols(df, ["예상 수주량"])
    if not target_cols:
        raise RuntimeError(f"Target columns not found in feat.csv. Columns: {list(df.columns)[:50]}")

    dt_use = dt_col or (FO.DEFAULT_DT_COL if hasattr(FO, "DEFAULT_DT_COL") else "DateTime")

    X, y, num_cols, cat_cols, _excluded = FO.build_xy(df, prod_col, target_cols, log_target)

    # best params load (forecast.py와 동일한 key mapping)
    bp = FO.load_best_params(best_params_path) or {}

    reg_n_jobs = 1 if deterministic else -1
    if FO.lgb is None:
        raise RuntimeError("lightgbm 미설치. pip install lightgbm")

    lgbm_params = dict(
        objective="tweedie",
        tweedie_variance_power=bp.get("power", bp.get("tweedie_variance_power", 1.3)),
        learning_rate=bp.get("lr", bp.get("learning_rate", 0.05)),
        n_estimators=bp.get("n_estimators", 1000),
        num_leaves=bp.get("num_leaves", 63),
        min_child_samples=bp.get("min_child_samples", 50),
        subsample=bp.get("subsample", 0.8),
        colsample_bytree=bp.get("colsample_bytree", 0.8),
        reg_lambda=bp.get("reg_lambda", 5.0),
        random_state=seed,
        deterministic=deterministic,
        force_row_wise=deterministic,
    )
    print("Loaded best params from:", best_params_path)
    print("🔧 Final params:", lgbm_params)

    model = FO.build_model_pipeline(
        "lgbm",
        num_cols=num_cols,
        cat_cols=cat_cols,
        tweedie_power=lgbm_params["tweedie_variance_power"],
        alpha=0.5,
        lgbm_params=lgbm_params,
        reg_n_jobs=reg_n_jobs,
    )

    # ---- split
    if split == "time":
        X_tr, X_va, y_tr, y_va = FO.time_split(df, X, y, dt_use, val_size)
    elif split == "group":
        X_tr, X_va, y_tr, y_va = FO.group_split(df, X, y, prod_col, val_size, seed)
    else:
        from sklearn.model_selection import train_test_split
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=val_size, random_state=seed)

    model.fit(X_tr, y_tr.values)
    pred_va = np.maximum(0.0, np.asarray(model.predict(X_va), dtype=float))

    if log_target:
        pred_va = np.expm1(pred_va)
        y_va2 = np.expm1(y_va)
    else:
        y_va2 = y_va

    # metrics
    rows = {}
    for i, t in enumerate(y.columns):
        yt = y_va2[t].values
        pt = pred_va[:, i]
        rows[t] = {
            "MAE": float(np.mean(np.abs(yt - pt))),
            "RMSE": float(FO.rmse(yt, pt)),
            "R2": float(FO.r2_score(yt, pt)) if hasattr(FO, "r2_score") else float(0.0),
            "SMAPE": float(FO.smape(yt, pt)),
            **FO.binary_metrics(yt, pt),
        }
    metrics_df = pd.DataFrame(rows).T
    print("Validation metrics")
    print(metrics_df.to_string())

    # residual pool on validation (y_true - y_pred)
    residuals_val = y_va2.values.astype(float) - pred_va.astype(float)  # (n_val, k)

    # predict all rows
    pred_all = FO.predict_all(model, X, df, prod_col, dt_use, target_cols)

    # save outputs
    ensure_dir(os.path.dirname(out_pred_csv))
    pred_all.to_csv(out_pred_csv, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(out_metrics_csv, encoding="utf-8-sig")
    print(f"예측 저장: {out_pred_csv}")
    print(f"저장: {out_metrics_csv}")

    # by-product average (forecast.py helper)
    prod_agg = FO.aggregate_by_product(pred_all, prod_col)
    prod_agg.to_csv(out_pred_by_product_csv, index=False, encoding="utf-8-sig")
    print(f"제품별 평균 저장: {out_pred_by_product_csv}")

    # residual df for validation rows (need prod/dt + residual columns)
    df_val = df.loc[X_va.index].copy()
    res_df = pd.DataFrame(residuals_val, columns=target_cols, index=X_va.index)
    keep_cols = [prod_col] + ([dt_use] if dt_use in df_val.columns else [])
    residual_df_val = pd.concat([df_val[keep_cols], res_df], axis=1)

    return pred_all, metrics_df, residual_df_val, target_cols, dt_use


# =========================================================
# 파이프라인
# =========================================================
def run_pipeline(
    data_csv: str,
    out_dir: str,
    prod_col: str = "Product_Number",
    dt_col: str | None = None,
    daily_capacity: int = 10000,
    lambda_smooth: float = 1.0,
    initial_inventory: float = 0.0,
    int_production: bool = True,
    model_name: str = "gpt-4o-mini",
    policy_path: str | None = None,
    skip_llm: bool = False,
    best_params_path: str = "./configs/best_params.json",
    mc_scenarios: int = 0,
    mc_alpha: float = 0.9,
    mc_seed: int = 2025,
):
    """
    전체 파이프라인:
      1) features: data.csv -> feat.csv
      2) forecast: feat.csv -> pred_final.csv, metrics_final.csv, pred_final_by_product.csv
      2.5) (옵션) MC scenarios 생성: mc_scenarios.csv (scenario_id, day_idx, Product_Number, demand)
      3) planner_opt: pred_final.csv + feat.csv -> production_plan.csv
      3.5) (옵션) MC validate: evaluator.mc_validate_plan -> mc_validation.json
      4) metrics(planning): production_plan.csv -> planning_metrics.json
      5) evaluator: audit_and_learn -> audit_result.json (+ policy.json)
      6) report_llm: weekly_report.md/json (옵션, mc_json 포함)
    """
    ensure_dir(out_dir)

    # ---------- 0) 경로 ----------
    artifacts_dir = os.path.join(out_dir, "outputs")
    reports_dir = os.path.join(out_dir, "reports")
    ensure_dir(artifacts_dir)
    ensure_dir(reports_dir)

    feat_csv = os.path.join(artifacts_dir, "feat.csv")
    forecast_csv = os.path.join(artifacts_dir, "pred_final.csv")
    forecast_metrics_csv = os.path.join(artifacts_dir, "metrics_final.csv")
    forecast_by_product_csv = os.path.join(artifacts_dir, "pred_final_by_product.csv")

    plan_csv = os.path.join(artifacts_dir, "production_plan.csv")
    planning_metrics_json = os.path.join(artifacts_dir, "planning_metrics.json")

    policy_json = policy_path or os.path.join(out_dir, "policy.json")
    audit_json = os.path.join(out_dir, "audit_result.json")

    mc_scenarios_csv = os.path.join(artifacts_dir, "mc_scenarios.csv")
    mc_validation_json = os.path.join(artifacts_dir, "mc_validation.json")

    # ---------- 1) Feature ----------
    print("[1/6] Building features ...")
    raw_df = pd.read_csv(data_csv, encoding="utf-8")
    feat_df, clus_summary = FE.build_features(raw_df)
    ensure_dir(os.path.dirname(feat_csv))
    feat_df.to_csv(feat_csv, index=False, encoding="utf-8")

    if clus_summary is not None and not clus_summary.empty:
        clus_csv_path = os.path.join(artifacts_dir, "cluster_summary.csv")
        clus_summary.to_csv(clus_csv_path, index=False, encoding="utf-8")

    if not os.path.exists(feat_csv):
        raise RuntimeError(f"feat.csv not found: {feat_csv}")
    print(f"  -> {feat_csv}")

    # ---------- 2) Forecast ----------
    print("[2/6] Forecasting via Python API ...")
    pred_all, metrics_df, residual_df_val, target_cols, dt_use = _run_forecast_python_api(
        feat_csv=feat_csv,
        out_pred_csv=forecast_csv,
        out_metrics_csv=forecast_metrics_csv,
        out_pred_by_product_csv=forecast_by_product_csv,
        prod_col=prod_col,
        dt_col=dt_col,
        best_params_path=best_params_path,
        seed=mc_seed,
        split="time",
        val_size=0.2,
        deterministic=False,
        log_target=False,
    )
    print(f"  -> {forecast_csv}")
    print(f"  -> {forecast_metrics_csv}")
    print(f"  -> {forecast_by_product_csv}")

    # ---------- 2.5) MC scenarios 생성 (옵션) ----------
    if mc_scenarios and mc_scenarios > 0:
        print(f"[2.5/6] Generating MC demand scenarios (S={mc_scenarios}) ...")

        # 제품별 최신 DateTime 스냅샷 기반 수요(P×D)
        pred_snap = PO.preprocess_forecast(pred_all.copy())
        horizons = PO.detect_horizons(pred_snap)

        # validation residual도 동일 스냅샷 기준(P×D)
        res_snap = PO.preprocess_forecast(residual_df_val.copy())

        keep_cols = [prod_col] + horizons
        for df_ in (pred_snap, res_snap):
            for c in keep_cols:
                if c not in df_.columns:
                    raise RuntimeError(
                        f"MC scenario build failed: missing col '{c}' in snapshot df. "
                        f"cols={list(df_.columns)[:30]}"
                    )

        d_hat = pred_snap[horizons].to_numpy(dtype=float)      # (P, D)
        res_pool = res_snap[horizons].to_numpy(dtype=float)    # (P, D)  pool size=P

        scenarios = FO.generate_demand_scenarios(
            d_hat=d_hat,
            residuals=res_pool,
            n_scenarios=int(mc_scenarios),
            seed=int(mc_seed),
        )  # (S, P, D)

        prod_list = pred_snap[prod_col].astype(str).tolist()

        rows = []
        S, P, D = scenarios.shape
        for s in range(S):
            for i, p in enumerate(prod_list):
                for d in range(D):
                    rows.append({
                        "scenario_id": int(s),
                        "day_idx": int(d),
                        "Product_Number": str(p),
                        "demand": float(scenarios[s, i, d]),
                    })

        scen_long = pd.DataFrame(rows)
        scen_long.to_csv(mc_scenarios_csv, index=False, encoding="utf-8-sig")
        print(f"  -> {mc_scenarios_csv} (rows={len(scen_long)})")
    else:
        print("[2.5/6] MC scenarios disabled (mc_scenarios=0)")

    # ---------- 3) Planner ----------
    print("[3/6] Planning (CP-SAT) ...")
    cluster_info = PO.load_cluster_info(feat_csv)
    pred_df = pd.read_csv(forecast_csv)
    pred_df = PO.preprocess_forecast(pred_df)
    horizons = PO.detect_horizons(pred_df)

    plan_df = PO.optimize_plan(
        forecast_by_product=pred_df,
        horizons=horizons,
        prod_col=prod_col,
        cluster_info=cluster_info,
        daily_capacity=daily_capacity,
        lambda_smooth=lambda_smooth,
        initial_inventory=initial_inventory,
        int_production=int_production,
        scale=10,
    )

    for c in ["demand", "produce", "end_inventory", "backlog"]:
        if c in plan_df.columns:
            plan_df[c] = pd.to_numeric(plan_df[c], errors="coerce")
    if prod_col in plan_df.columns:
        plan_df[prod_col] = plan_df[prod_col].astype(str).str.replace(r"\.0$", "", regex=True)

    # capa/day 컬럼 부여(보고서/요약용)
    counts = plan_df.groupby("day_idx")[prod_col].transform("count")
    plan_df["capa"] = daily_capacity / counts
    plan_df["day"] = plan_df["day_idx"]

    plan_df.to_csv(plan_csv, index=False)
    print(f"  -> {plan_csv} (rows={len(plan_df)})")

    # ---------- 3.5) MC validation (옵션) ----------
    mc_validation = None
    if mc_scenarios and mc_scenarios > 0 and os.path.exists(mc_scenarios_csv):
        print("[3.5/6] MC validating fixed plan (via evaluator) ...")
        scenarios_df = EV.load_mc_scenarios(mc_scenarios_csv)
        mc_validation = EV.mc_validate_plan(
            plan_df=plan_df,
            scenarios_df=scenarios_df,
            initial_inventory=float(initial_inventory),
            alpha=float(mc_alpha),
        )
        save_json(mc_validation_json, mc_validation)
        print(f"  -> {mc_validation_json}")
    else:
        mc_validation_json = ""  # report_llm에 넘길 때 빈 값

    # ---------- 4) Planning KPI ----------
    print("[4/6] Computing planning metrics ...")
    planning_kpi = M.compute_planning_metrics(
        plan_df=plan_df,
        daily_capacity=daily_capacity,
        feat_df=pd.read_csv(feat_csv),
        product_col=prod_col
    )
    save_json(planning_metrics_json, planning_kpi)
    print(f"  -> {planning_metrics_json}")

    # ---------- 5) Evaluator & Policy Update ----------
    print("[5/6] Evaluating plan & updating policy ...")

    # metrics_summary는 evaluator의 update_policy_from_outcomes가 기대하는 구조를 따름
    metrics_summary = {"planning_metrics": planning_kpi}

    audit = EV.audit_and_learn(
        plan_df=plan_df,
        daily_capacity=daily_capacity,
        metrics_summary=metrics_summary,
        policy_path=policy_json,
        llm_enabled=False
    )

    # MC 요약을 audit_result에 포함
    if isinstance(mc_validation, dict) and mc_validation.get("summary"):
        audit["mc_validation"] = mc_validation.get("summary")

    save_json(audit_json, audit)
    print(f"  -> {audit_json}")

    # ---------- 6) Report (LLM) ----------
    print("[6/6] Building weekly report ...")
    if skip_llm:
        print("  (LLM report skipped)")
    else:
        ts = _now_str()
        out_md = os.path.join(reports_dir, f"weekly_report_{ts}.md")
        out_json = os.path.join(reports_dir, f"weekly_report_{ts}.json")
        out_verify = os.path.join(reports_dir, f"weekly_report_{ts}.verify.txt")

        out = RL.build_report_with_llm(
            plan_csv=plan_csv,
            forecast_csv=forecast_by_product_csv,  # 보고서용은 by_product가 안정적
            metrics_csv=forecast_metrics_csv if os.path.exists(forecast_metrics_csv) else "",
            model_name=model_name,
            auto_regen_on_fail=True,
            feat_csv=feat_csv,
            mc_json=mc_validation_json if (mc_validation_json and os.path.exists(mc_validation_json)) else None,
        )

        if out.get("markdown"):
            ensure_dir(os.path.dirname(out_md))
            with open(out_md, "w", encoding="utf-8") as f:
                f.write(out["markdown"])

        if out.get("json") is not None:
            save_json(out_json, out["json"])

        if out.get("verify"):
            ensure_dir(os.path.dirname(out_verify))
            with open(out_verify, "w", encoding="utf-8") as f:
                v = out["verify"]
                f.write(("OK" if v.get("ok") else "NG") + "\n\n")
                f.write(v.get("report", ""))

        print(f"  -> {out_md}\n  -> {out_json}\n  -> {out_verify}")

    print("\n[DONE] Pipeline finished.")


# =========================================================
# CLI
# =========================================================
def parse_args():
    p = argparse.ArgumentParser(description="End-to-end SCM planning pipeline")

    p.add_argument("--data", required=True, help="원본 데이터 CSV (예: ./data/data.csv)")
    p.add_argument("--out_dir", default="./outputs", help="출력 루트 디렉토리")

    p.add_argument("--prod_col", default="Product_Number")
    p.add_argument("--dt_col", default=None)

    p.add_argument("--daily_capacity", type=int, default=10000)
    p.add_argument("--lambda_smooth", type=float, default=1.0)
    p.add_argument("--initial_inventory", type=float, default=0.0)
    p.add_argument("--int_production", action="store_true")

    p.add_argument("--model", default="gpt-4o-mini", help="LLM 모델명 (report_llm)")
    p.add_argument("--policy_path", default=None, help="정책 파일 경로(없으면 out_dir/policy.json)")
    p.add_argument("--skip_llm", action="store_true", help="LLM 보고서 생성을 스킵")

    p.add_argument("--best_params_path", default="./configs/best_params.json",
                   help="forecast 단계에서 사용할 best_params JSON 경로")

    # MC options
    p.add_argument("--mc_scenarios", type=int, default=0, help="MC scenario count (0 disables)")
    p.add_argument("--mc_alpha", type=float, default=0.9, help="MC summary quantile alpha (0.9=p90)")
    p.add_argument("--mc_seed", type=int, default=2025, help="MC random seed")

    return p.parse_args()

def main():
    args = parse_args()
    run_pipeline(
        data_csv=args.data,
        out_dir=args.out_dir,
        prod_col=args.prod_col,
        dt_col=args.dt_col,
        daily_capacity=args.daily_capacity,
        lambda_smooth=args.lambda_smooth,
        initial_inventory=args.initial_inventory,
        int_production=args.int_production,
        model_name=args.model,
        policy_path=args.policy_path,
        skip_llm=args.skip_llm,
        best_params_path=args.best_params_path,
        mc_scenarios=args.mc_scenarios,
        mc_alpha=args.mc_alpha,
        mc_seed=args.mc_seed,
    )

if __name__ == "__main__":
    main()