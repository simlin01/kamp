#!/usr/bin/env python3
# -*- coding: utf-8 -*-
''' CLI
python -m main_test \
  --data ./data/data.csv \
  --out_dir ./outputs \
  --prod_col Product_Number \
  --daily_capacity 5000 \
  --int_production \
  --model gpt-4o-mini
'''
from __future__ import annotations

import os, sys
import json
import argparse
from datetime import datetime
import pandas as pd
import subprocess
import shlex

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

# ---------------------------------------------------------
# 유틸
# ---------------------------------------------------------
def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def save_json(path: str, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _now_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# ---------------------------------------------------------
# 파이프라인
# ---------------------------------------------------------
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
    scenario_names: list[str] | None = None,
    policy_path: str | None = None,
    skip_llm: bool = False,
    best_params_path: str = "./configs/best_params.json",
):
    """
    전체 파이프라인:
      1) features: data.csv -> feat.csv
      2) forecast: feat.csv -> pred_final.csv, metrics_final.csv
      3) planner_opt: pred_final.csv + feat.csv -> production_plan.csv
      4) metrics(planning): production_plan.csv -> planning_metrics.json
      5) evaluator: verify + policy update -> policy.json (옵션)
      6) report_llm: weekly_report.md/json (옵션)
    """
    ensure_dir(out_dir)

    # ---------- 0) 경로 준비 ----------
    artifacts_dir = os.path.join(out_dir, "outputs")
    reports_dir = os.path.join(out_dir, "reports")
    ensure_dir(artifacts_dir)
    ensure_dir(reports_dir)

    feat_csv = os.path.join(artifacts_dir, "feat.csv")
    forecast_csv = os.path.join(artifacts_dir, "pred_final.csv")
    forecast_metrics_csv = os.path.join(artifacts_dir, "metrics_final.csv")
    plan_csv = os.path.join(artifacts_dir, "production_plan.csv")
    planning_metrics_json = os.path.join(artifacts_dir, "planning_metrics.json")
    policy_json = policy_path or os.path.join(out_dir, "policy.json")

    # ---------- 1) Feature 생성 ----------
    # data.csv -> feat.csv
    # ---------- 1) Feature 생성 ----------
    # data.csv -> feat.csv
    print("[1/6] Building features ...")

    # 1) 입력 CSV → DataFrame
    raw_df = pd.read_csv(data_csv, encoding="utf-8")

    # 2) 피처 생성 (DataFrame in → DataFrame out)
    feat_df, clus_summary = FE.build_features(raw_df)

    # 3) 결과 저장 (feat_csv는 위에서 artifacts_dir 기반으로 이미 정의됨)
    #    artifacts_dir = os.path.join(out_dir, "outputs")  # 위에서 선언됨
    ensure_dir(os.path.dirname(feat_csv))
    feat_df.to_csv(feat_csv, index=False, encoding="utf-8")

    # (선택) 클러스터 요약도 저장
    if clus_summary is not None and not clus_summary.empty:
        clus_csv_path = os.path.join(artifacts_dir, "cluster_summary.csv")
        clus_summary.to_csv(clus_csv_path, index=False, encoding="utf-8")

    if not os.path.exists(feat_csv):
        raise RuntimeError(f"feat.csv not found: {feat_csv}")
    print(f"  -> {feat_csv}")

    # ---------- 2) Forecast ----------
    # feat.csv -> pred_final.csv, metrics_final.csv
    # ---------- 2) Forecast (CLI 호출) ----------
    print("[2/6] Forecasting via CLI ...")
    os.makedirs(os.path.dirname(forecast_csv), exist_ok=True)
    if forecast_metrics_csv:
        os.makedirs(os.path.dirname(forecast_metrics_csv), exist_ok=True)

    cmd = [
        sys.executable, "-m", "src.forecast",
        "--in", feat_csv,
        "--out", forecast_csv,
        "--metrics_out", forecast_metrics_csv,
        "--prod_col", prod_col,
        "--model", "lgbm",
        "--split", "time",
        "--val_size", "0.2",
        "--seed", "2025",
        "--best_params_path", best_params_path
    ]
    # dt_col이 있으면 추가
    if dt_col:
        cmd.extend(["--dt_col", dt_col])

    print("  CMD:", " ".join(shlex.quote(x) for x in cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)

    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr)
        raise RuntimeError(f"forecast.py failed with code {res.returncode}")

    print(res.stdout.strip() or "[forecast] done")
    if not os.path.exists(forecast_csv):
        raise RuntimeError(f"pred_final.csv not found: {forecast_csv}")
    if not os.path.exists(forecast_metrics_csv):
        print("[WARN] forecast metrics file missing; continuing:", forecast_metrics_csv)
    print(f"  -> {forecast_csv}")
    print(f"  -> {forecast_metrics_csv}")

    # ---------- 3) Planner (CP-SAT) ----------
    print("[3/6] Planning (CP-SAT) ...")
    cluster_info = PO.load_cluster_info(feat_csv)
    pred_df = pd.read_csv(forecast_csv)
    pred_df = PO.preprocess_forecast(pred_df)

    # horizons 자동 감지 (T, T+1, 또는 'T+1일 예상 수주량' 형태)
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
        scale=10,  # 소수 수요 정수화 스케일
        # 정책 맵을 policy에서 불러와 주입할 수도 있음
        # min_lot_map=..., safety_stock_map=..., weight_map=...
    )
    # 문자열 포매팅(.2f)로 생산된 경우 수치형으로도 하나 더 저장하고 싶으면 여기서 캐스팅
    # (리포트/LNS용은 문자열도 괜찮지만, KPI 집계는 숫자가 편함)
    for c in ["demand", "produce", "end_inventory", "backlog"]:
        if c in plan_df.columns and plan_df[c].dtype == object:
            try:
                plan_df[c] = plan_df[c].astype(float)
            except Exception:
                pass

    plan_df.to_csv(plan_csv, index=False)
    print(f"  -> {plan_csv} (rows={len(plan_df)})")

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
    metrics_summary = {
        "forecast_metrics_csv": forecast_metrics_csv,
        "planning_metrics": planning_kpi
    }

    # (옵션) 과거 policy를 불러와 업데이트
    policy = EV.load_policy(policy_json) if policy_json else None
    audit = EV.audit_and_learn(
        plan_df=plan_df,
        daily_capacity=daily_capacity,
        metrics_summary={"planning_metrics": planning_kpi},
        policy_path=policy_json,  # 전달하면 저장까지
        llm_enabled=False  # 여기서는 규칙기반만, 필요시 True
    )
    save_json(os.path.join(out_dir, "audit_result.json"), audit)
    print(f"  -> {os.path.join(out_dir, 'audit_result.json')}")

    # ---------- 6) Report (LLM) ----------
    print("[6/6] Building weekly report ...")
    if skip_llm:
        print("  (LLM report skipped)")
    else:
        out_md = os.path.join(reports_dir, f"weekly_report_{_now_str()}.md")
        out_json = os.path.join(reports_dir, f"weekly_report_{_now_str()}.json")
        out_verify = os.path.join(reports_dir, f"weekly_report_{_now_str()}.verify.txt")

        # 단일 시나리오(기본). 여러 계획 파일 비교 시 plans=[...], scenario_names=[...] 로 전달
        out = RL.build_report_with_llm(
            plan_csv=plan_csv,
            forecast_csv=forecast_csv,
            metrics_csv=forecast_metrics_csv if os.path.exists(forecast_metrics_csv) else "",
            model_name=model_name,
            auto_regen_on_fail=True
        )
        if out.get("markdown"):
            with open(out_md, "w", encoding="utf-8") as f:
                f.write(out["markdown"])
        if out.get("json") is not None:
            save_json(out_json, out["json"])
        if out.get("verify"):
            with open(out_verify, "w", encoding="utf-8") as f:
                v = out["verify"]
                f.write(("OK" if v.get("ok") else "NG") + "\n\n")
                f.write(v.get("report", ""))

        print(f"  -> {out_md}\n  -> {out_json}\n  -> {out_verify}")

    print("\n[DONE] Pipeline finished.")

# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
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
    p.add_argument("--best_params_path", default="./configs/best_params.json", help="forecast.py에 넘길 최적 파라미터 JSON 경로")
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
        skip_llm=args.skip_llm
    )

if __name__ == "__main__":
    main()
