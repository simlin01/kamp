#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — End-to-end SCM planning pipeline (ONE-SHOT)
Features → Forecast → (MC scenarios) → Planner(CVaR+Hint) → MC Validate → Metrics → Evaluator → Report

권장 실행 예시:
python -m src.main \
  --data ./data/data.csv \
  --out_dir ./outputs \
  --daily_capacity 10000 \
  --lambda_smooth 0.1 \
  --int_production \
  --mc_scenarios 200 \
  --mc_alpha 0.9 \
  --mc_seed 2025 \
  --use_cvar_planner \
  --hint_plan_csv ./outputs/plan_S200_minlotfix_only.csv \
  --lambda_cvar 0.07 \
  --lambda_scale 1000000 \
  --loss_scale 2000 \
  --planner_max_time 1800 \
  --planner_gap 0.2 \
  --planner_workers 8 \
  --planner_seed 42 \
  --min_lot_map '{"0":30,"1":30,"2":0,"3":50}' \
  --fail_threshold 0.55
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

# =========================================================
# Import path (robust for "python -m src.main" and "python src/main.py")
# =========================================================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]  # .../kamp
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import features as FE
from src import forecast as FO
from src import planner_opt as PO
from src import metrics as M
from src import evaluator as EV
from src import report_llm as RL


# =========================================================
# Utils
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


def _load_json_map_or_none(s: str | None) -> dict | None:
    """
    s가 JSON string이면 dict로, '@path.json'이면 파일 읽어서 dict로.
    """
    if not s:
        return None
    s = str(s).strip()
    if not s:
        return None
    if s.startswith("@"):
        p = s[1:]
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(s)


def _as_int_key_float_map(d: dict | None) -> dict[int, float] | None:
    if d is None:
        return None
    out: dict[int, float] = {}
    for k, v in d.items():
        out[int(k)] = float(v)
    return out


# =========================================================
# Forecast helper (Python API)
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
    df = pd.read_csv(feat_csv)

    target_cols = FO.find_target_cols(df, ["예상 수주량"])
    if not target_cols:
        raise RuntimeError(f"Target columns not found in feat.csv. Columns: {list(df.columns)[:50]}")

    dt_use = dt_col or (FO.DEFAULT_DT_COL if hasattr(FO, "DEFAULT_DT_COL") else "DateTime")
    X, y, num_cols, cat_cols, _excluded = FO.build_xy(df, prod_col, target_cols, log_target)

    bp = FO.load_best_params(best_params_path) or {}

    reg_n_jobs = 1 if deterministic else -1
    if getattr(FO, "lgb", None) is None:
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

    # split
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

    residuals_val = y_va2.values.astype(float) - pred_va.astype(float)
    pred_all = FO.predict_all(model, X, df, prod_col, dt_use, target_cols)

    ensure_dir(os.path.dirname(out_pred_csv))
    pred_all.to_csv(out_pred_csv, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(out_metrics_csv, encoding="utf-8-sig")
    print(f"예측 저장: {out_pred_csv}")
    print(f"저장: {out_metrics_csv}")

    prod_agg = FO.aggregate_by_product(pred_all, prod_col)
    prod_agg.to_csv(out_pred_by_product_csv, index=False, encoding="utf-8-sig")
    print(f"제품별 평균 저장: {out_pred_by_product_csv}")

    df_val = df.loc[X_va.index].copy()
    res_df = pd.DataFrame(residuals_val, columns=target_cols, index=X_va.index)
    keep_cols = [prod_col] + ([dt_use] if dt_use in df_val.columns else [])
    residual_df_val = pd.concat([df_val[keep_cols], res_df], axis=1)

    return pred_all, metrics_df, residual_df_val, target_cols, dt_use


# =========================================================
# Pipeline
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
    mc_wb: float = 1.0,
    mc_wi: float = 0.2,
    fail_threshold: float = 0.55,  # ✅ CLI로 제어
    scale: int = 10,
    # ---- Planner ----
    use_cvar_planner: bool = False,  # ✅ 기본은 False. 켤 때만 --use_cvar_planner
    hint_plan_csv: str | None = None,
    lambda_cvar: float = 0.07,
    lambda_scale: int = 1_000_000,
    loss_scale: int = 2000,
    planner_max_time: float = 1800.0,
    planner_gap: float = 0.2,
    planner_workers: int = 8,
    planner_seed: int = 42,
    min_lot_map_json: str | None = None,
):
    ensure_dir(out_dir)

    # ---------- output paths (ALL under out_dir) ----------
    reports_dir = os.path.join(out_dir, "reports")
    ensure_dir(reports_dir)

    feat_csv = os.path.join(out_dir, "feat.csv")
    forecast_csv = os.path.join(out_dir, "pred_final.csv")
    forecast_metrics_csv = os.path.join(out_dir, "metrics_final.csv")
    forecast_by_product_csv = os.path.join(out_dir, "pred_final_by_product.csv")

    mc_scenarios_csv = os.path.join(out_dir, "pred_final_mc.csv")

    plan_csv = os.path.join(out_dir, "production_plan.csv")
    planning_metrics_json = os.path.join(out_dir, "planning_metrics.json")

    policy_json = policy_path or os.path.join(out_dir, "policy.json")
    audit_json = os.path.join(out_dir, "governance_audit.json")
    mc_validation_json = os.path.join(out_dir, "mc_validation.json")

    # ---------- 1) Features ----------
    print("[1/6] Building features ...")
    raw_df = pd.read_csv(data_csv, encoding="utf-8")
    feat_df, clus_summary = FE.build_features(raw_df)
    feat_df.to_csv(feat_csv, index=False, encoding="utf-8-sig")
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

    # ---------- 2.5) MC scenarios ----------
    if mc_scenarios and mc_scenarios > 0:
        print(f"[2.5/6] Generating MC demand scenarios (S={mc_scenarios}) ...")

        pred_snap = PO.preprocess_forecast(pred_all.copy())
        horizons = PO.detect_horizons(pred_snap)
        res_snap = PO.preprocess_forecast(residual_df_val.copy())

        keep_cols = [prod_col] + horizons
        for df_ in (pred_snap, res_snap):
            for c in keep_cols:
                if c not in df_.columns:
                    raise RuntimeError(
                        f"MC scenario build failed: missing col '{c}' in snapshot df. "
                        f"cols={list(df_.columns)[:30]}"
                    )

        d_hat = pred_snap[horizons].to_numpy(dtype=float)  # (P, D)
        res_pool = res_snap[horizons].to_numpy(dtype=float)  # (P, D)

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
                    rows.append(
                        {
                            "scenario_id": int(s),
                            "day_idx": int(d),
                            prod_col: str(p),
                            "demand": float(scenarios[s, i, d]),
                        }
                    )

        scen_long = pd.DataFrame(rows)
        scen_long.to_csv(mc_scenarios_csv, index=False, encoding="utf-8-sig")
        print(f"  -> {mc_scenarios_csv} (rows={len(scen_long)})")
    else:
        print("[2.5/6] MC scenarios disabled (mc_scenarios=0)")

    # ---------- 3) Planner ----------
    print("[3/6] Planning (CP-SAT) ...")

    cluster_info = PO.load_cluster_info(feat_csv)

    # planner 입력: product-level snapshot
    pred_df = pd.read_csv(forecast_by_product_csv)
    pred_df = PO.preprocess_forecast(pred_df)
    horizons = PO.detect_horizons(pred_df)

    # load min_lot_map
    min_lot_map = _as_int_key_float_map(_load_json_map_or_none(min_lot_map_json)) if min_lot_map_json else None

    # hint plan
    hint_plan = None
    if hint_plan_csv and os.path.exists(hint_plan_csv):
        if hasattr(PO, "load_hint_plan_csv"):
            hint_plan = PO.load_hint_plan_csv(hint_plan_csv, scale=scale)
            print(f"[HINT] loaded: {hint_plan_csv} | keys={len(hint_plan)}")
        else:
            print("[HINT] planner_opt.load_hint_plan_csv not found. Hint disabled.")
    elif hint_plan_csv:
        print(f"[HINT] not found: {hint_plan_csv} (continue without hint)")

    # MC scenarios (reordered) for CVaR planner
    scenarios_re = None
    if use_cvar_planner:
        if not (mc_scenarios and mc_scenarios > 0 and os.path.exists(mc_scenarios_csv)):
            raise RuntimeError(
                "use_cvar_planner=True 인데 pred_final_mc.csv가 없습니다. "
                "먼저 --mc_scenarios 200 처럼 MC 생성이 필요합니다."
            )

        if not (hasattr(PO, "load_mc_scenarios") and hasattr(PO, "reorder_mc_scenarios_to_forecast")):
            raise RuntimeError("planner_opt에 load_mc_scenarios / reorder_mc_scenarios_to_forecast가 필요합니다.")

        scenarios, mc_products = PO.load_mc_scenarios(mc_npz=None, mc_csv=mc_scenarios_csv, product_col=prod_col)
        pred_df[prod_col] = pred_df[prod_col].astype(str).str.replace(r"\.0$", "", regex=True)
        forecast_products = pred_df[prod_col].tolist()
        scenarios_re = PO.reorder_mc_scenarios_to_forecast(scenarios, mc_products, forecast_products)

    # optimize
    if use_cvar_planner:
        plan_df, diag = PO.optimize_plan(
            forecast_by_product=pred_df,
            horizons=horizons,
            prod_col=prod_col,
            cluster_info=cluster_info,
            daily_capacity=daily_capacity,
            lambda_smooth=lambda_smooth,
            initial_inventory=initial_inventory,
            int_production=int_production,
            scale=scale,
            min_lot_map=min_lot_map,
            hint_plan=hint_plan,
            use_cvar_obj=True,
            mc_scenarios=scenarios_re,
            cvar_alpha=float(mc_alpha),
            lambda_cvar=float(lambda_cvar),
            lambda_scale=int(lambda_scale),
            mc_wb=float(mc_wb),
            mc_wi=float(mc_wi),
            loss_scale=int(loss_scale),
            solver_seed=int(planner_seed),
            max_time=float(planner_max_time),
            workers=int(planner_workers),
            gap=float(planner_gap),
            log_progress=True,
        )
        print("[Planner CVaR diag]", diag)
    else:
        plan_df, diag = PO.optimize_plan(
            forecast_by_product=pred_df,
            horizons=horizons,
            prod_col=prod_col,
            cluster_info=cluster_info,
            daily_capacity=daily_capacity,
            lambda_smooth=lambda_smooth,
            initial_inventory=initial_inventory,
            int_production=int_production,
            scale=scale,
            min_lot_map=min_lot_map,
            hint_plan=hint_plan,
            use_cvar_obj=False,
            solver_seed=int(planner_seed),
            max_time=float(planner_max_time),
            workers=int(planner_workers),
            gap=float(planner_gap),
            log_progress=True,
        )
        print("[Planner BASIC diag]", diag)

    # normalize + save
    for c in ["demand", "produce", "end_inventory", "backlog", "shortage"]:
        if c in plan_df.columns:
            plan_df[c] = pd.to_numeric(plan_df[c], errors="coerce")
    if prod_col in plan_df.columns:
        plan_df[prod_col] = plan_df[prod_col].astype(str).str.replace(r"\.0$", "", regex=True)

    plan_df.to_csv(plan_csv, index=False, encoding="utf-8-sig")
    print(f"  -> {plan_csv} (rows={len(plan_df)})")

    # ---------- 3.5) MC validation ----------
    mc_validation = None
    mc_validation_path_for_report = ""
    if mc_scenarios and mc_scenarios > 0 and os.path.exists(mc_scenarios_csv):
        print("[3.5/6] MC validating fixed plan (via evaluator) ...")
        scenarios_df = EV.load_mc_scenarios(mc_scenarios_csv)
        mc_validation = EV.mc_validate_plan(
            plan_df=plan_df,
            scenarios_df=scenarios_df,
            initial_inventory=float(initial_inventory),
            daily_capacity=float(daily_capacity),
            alpha=float(mc_alpha),
            w_b=float(mc_wb),
            w_i=float(mc_wi),
            fail_threshold=float(fail_threshold),  # ✅ FIX (필수 인자)
        )
        save_json(mc_validation_json, mc_validation)
        mc_validation_path_for_report = mc_validation_json
        print(f"  -> {mc_validation_json}")
    else:
        print("[3.5/6] MC validation skipped (no scenarios)")

    # ---------- 4) Planning metrics ----------
    print("[4/6] Computing planning metrics ...")
    planning_kpi = M.compute_planning_metrics(
        plan_df=plan_df,
        daily_capacity=daily_capacity,
        feat_df=pd.read_csv(feat_csv),
        product_col=prod_col,
    )
    save_json(planning_metrics_json, planning_kpi)
    print(f"  -> {planning_metrics_json}")

    # ---------- 5) Evaluator ----------
    print("[5/6] Evaluating plan & updating policy ...")
    metrics_summary = {"planning_metrics": planning_kpi}

    audit = EV.audit_and_learn(
        plan_df=plan_df,
        daily_capacity=daily_capacity,
        metrics_summary=metrics_summary,
        policy_path=policy_json,
        llm_enabled=False,
    )

    if isinstance(mc_validation, dict) and mc_validation.get("summary"):
        audit["mc_validation"] = mc_validation.get("summary")

    save_json(audit_json, audit)
    print(f"  -> {audit_json}")

    # ---------- 6) Report ----------
    print("[6/6] Building weekly report ...")
    if skip_llm:
        print("  (LLM report skipped)")
        print("\n[DONE] Pipeline finished.")
        return

    ts = _now_str()
    out_md = os.path.join(reports_dir, f"weekly_report_{ts}.md")
    out_json = os.path.join(reports_dir, f"weekly_report_{ts}.json")
    out_verify = os.path.join(reports_dir, f"weekly_report_{ts}.verify.txt")

    out = RL.build_report_with_llm(
        plan_csv=plan_csv,
        forecast_csv=forecast_by_product_csv,
        metrics_csv=forecast_metrics_csv if os.path.exists(forecast_metrics_csv) else "",
        model_name=model_name,
        auto_regen_on_fail=True,
        feat_csv=feat_csv,
        mc_json=mc_validation_path_for_report,
        planning_metrics_json=planning_metrics_json if os.path.exists(planning_metrics_json) else "",
    )

    # save md/json/verify
    if isinstance(out, dict) and out.get("markdown"):
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(out["markdown"])

    if isinstance(out, dict) and out.get("json") is not None:
        save_json(out_json, out["json"])

    if isinstance(out, dict) and out.get("verify"):
        ensure_dir(os.path.dirname(out_verify))
        with open(out_verify, "w", encoding="utf-8") as f:
            v = out["verify"]
            f.write(("OK" if v.get("ok") else "NG") + "\n\n")
            f.write(v.get("report", ""))

    # html export (if helper exists)
    if out_md and os.path.exists(out_md):
        out_html = str(Path(out_md).with_suffix(".html"))
        try:
            facts_for_charts = out.get("facts") if isinstance(out, dict) else None
            if hasattr(RL, "md_to_html_with_charts"):
                RL.md_to_html_with_charts(out_md, out_html, facts=facts_for_charts, title="주간 운영 계획 보고서")
            else:
                RL.md_to_html(out_md, out_html, title="주간 운영 계획 보고서")
            print(f"[OK] Saved HTML: - {out_html}")
        except Exception as e:
            print("[WARN] HTML export skipped:", repr(e))

    print(f"  -> {out_md}\n  -> {out_json}\n  -> {out_verify}")
    print("\n[DONE] Pipeline finished.")


# =========================================================
# CLI
# =========================================================
def parse_args():
    p = argparse.ArgumentParser(description="End-to-end SCM planning pipeline (one-shot)")

    p.add_argument("--data", required=True, help="원본 데이터 CSV (예: ./data/data.csv)")
    p.add_argument("--out_dir", default="./outputs", help="출력 디렉토리 (산출물은 모두 여기 직하위)")

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

    # MC
    p.add_argument("--mc_scenarios", type=int, default=0, help="MC scenario count (0 disables)")
    p.add_argument("--mc_alpha", type=float, default=0.9, help="MC quantile alpha (0.9≈p90)")
    p.add_argument("--mc_seed", type=int, default=2025, help="MC random seed")
    p.add_argument("--mc_wb", type=float, default=1.0, help="MC Loss weight (shortage/backlog)")
    p.add_argument("--mc_wi", type=float, default=0.2, help="MC Loss weight (inventory)")
    p.add_argument("--fail_threshold", type=float, default=0.55, help="MC validation fail threshold (Loss 기준)")

    # Planner scaling
    p.add_argument("--scale", type=int, default=10, help="planner integer scaling factor")

    # Planner(CVaR) knobs
    p.add_argument("--use_cvar_planner", action="store_true", help="planner_opt를 CVaR 모드로 실행")
    p.add_argument("--hint_plan_csv", default=None, help="Warm-start hint plan CSV 경로(선택)")
    p.add_argument("--lambda_cvar", type=float, default=0.07)
    p.add_argument("--lambda_scale", type=int, default=1_000_000)
    p.add_argument("--loss_scale", type=int, default=2000)
    p.add_argument("--planner_max_time", type=float, default=1800.0)
    p.add_argument("--planner_gap", type=float, default=0.2)
    p.add_argument("--planner_workers", type=int, default=8)
    p.add_argument("--planner_seed", type=int, default=42)
    p.add_argument("--min_lot_map", type=str, default=None, help='예: \'{"0":30,"1":30,"2":0,"3":50}\'')

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
        mc_wb=args.mc_wb,
        mc_wi=args.mc_wi,
        fail_threshold=args.fail_threshold,
        scale=args.scale,
        use_cvar_planner=bool(args.use_cvar_planner),
        hint_plan_csv=args.hint_plan_csv,
        lambda_cvar=args.lambda_cvar,
        lambda_scale=args.lambda_scale,
        loss_scale=args.loss_scale,
        planner_max_time=args.planner_max_time,
        planner_gap=args.planner_gap,
        planner_workers=args.planner_workers,
        planner_seed=args.planner_seed,
        min_lot_map_json=args.min_lot_map,
    )


if __name__ == "__main__":
    main()