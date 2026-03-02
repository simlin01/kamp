#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — End-to-end SCM planning pipeline (ONE-SHOT)
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

# =========================================================
# Import path
# =========================================================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import features as FE
from src import forecast as FO
from src import planner_opt as PO
from src import metrics as M
from src import evaluator as EV
from src import report_llm as RL

# =========================================================
# Constants / helpers
# =========================================================
PAST_MARKERS = ["작년", "전년", "지난해"]


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


def _is_past_ref(col: str) -> bool:
    return any(m in str(col) for m in PAST_MARKERS)


def _extract_horizon_idx(col: str) -> Optional[int]:
    import re

    s = str(col).strip()
    if s == "T":
        return 0
    m = re.fullmatch(r"T\+(\d+)", s)
    if m:
        return int(m.group(1))
    if "T일" in s:
        return 0
    m2 = re.search(r"T\+(\d+)\s*일", s)
    if m2:
        return int(m2.group(1))
    m3 = re.search(r"T\+(\d+)", s)
    if m3:
        return int(m3.group(1))
    return None


def _select_planner_horizons(
    df: pd.DataFrame,
    *,
    prefer: str = "예상",
    allow_scheduled: bool = False,
    max_h: int = 4
) -> List[str]:
    cols = list(df.columns)

    def ok(c: str) -> bool:
        if _is_past_ref(c):
            return False
        idx = _extract_horizon_idx(c)
        if idx is None:
            return False
        if idx < 0 or idx > int(max_h):
            return False

        s = str(c)
        if prefer == "예상":
            if "예상" in s:
                return True
            if allow_scheduled and ("예정" in s):
                return True
            return False
        return False

    picked = [c for c in cols if ok(c)]
    picked = sorted(
        picked,
        key=lambda x: (_extract_horizon_idx(x) or 10_000, 0 if "예상" in str(x) else 1, str(x)),
    )
    if not picked:
        raise RuntimeError(
            "planner horizons selection failed.\n"
            f"- prefer={prefer}, allow_scheduled={allow_scheduled}, max_h={max_h}\n"
            f"- columns sample={cols[:30]}"
        )
    return picked


def _horizon_idx_list(max_h: int) -> List[int]:
    return list(range(0, int(max_h) + 1))


# =========================================================
# Forecast helper (forecast.py 최신 시그니처와 정합)
# =========================================================
def _run_forecast_python_api(
    feat_csv: str,
    out_pred_csv: str,
    out_metrics_csv: str,
    out_pred_by_product_csv: str,
    prod_col: str,
    dt_col: str | None,
    best_params_path: str,
    planner_max_h: int,
    seed: int = 2025,
    split: str = "time",
    val_size: float = 0.2,
    deterministic: bool = False,
    log_target: bool = False,
):
    df = pd.read_csv(feat_csv)

    dt_use = dt_col or (FO.DEFAULT_DT_COL if hasattr(FO, "DEFAULT_DT_COL") else "DateTime")

    horizons_idx = _horizon_idx_list(planner_max_h)
    target_cols = FO.find_target_cols(df, target_kind="expected", horizons=horizons_idx)

    if not target_cols:
        raise RuntimeError(
            f"Target columns not found in feat.csv. (expected) horizons={horizons_idx}\n"
            f"columns(head)={list(df.columns)[:40]}"
        )

    X, y, num_cols, cat_cols, _excluded = FO.build_xy(
        df=df,
        prod_col=prod_col,
        target_cols=target_cols,
        target_kind="expected",
        allow_expected_as_feature=False,
        log_target=log_target,
    )

    bp = FO.load_best_params(best_params_path) or {}

    reg_n_jobs = 1 if deterministic else -1
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
    )
    if deterministic:
        lgbm_params.update(dict(deterministic=True, force_row_wise=True))

    model = FO.build_model_pipeline(
        model_name="lgbm",
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

    # metrics (per-horizon)
    rows: Dict[str, dict] = {}
    for i, t in enumerate(y.columns):
        yt = y_va2[t].values
        pt = pred_va[:, i]
        rows[t] = {
            "MAE": float(np.mean(np.abs(yt - pt))),
            "RMSE": float(FO.rmse(yt, pt)),
            "R2": float(r2_score(yt, pt)),
            "SMAPE": float(FO.smape(yt, pt)),
            **FO.binary_metrics(yt, pt),
        }
    metrics_df = pd.DataFrame(rows).T

    # residual pool (validation rows)
    residuals_val = y_va2.values.astype(float) - pred_va.astype(float)

    # full prediction
    pred_all = FO.predict_all(model, X, df, prod_col, dt_use, target_cols)

    ensure_dir(os.path.dirname(out_pred_csv))
    pred_all.to_csv(out_pred_csv, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(out_metrics_csv, encoding="utf-8-sig")

    # 제품단(plan 입력)은 "최신 스냅샷"
    prod_snap = FO.snapshot_latest_by_product(
        pred_all,
        prod_col=prod_col,
        dt_col=dt_use,
        value_cols=target_cols,
    )
    prod_snap.to_csv(out_pred_by_product_csv, index=False, encoding="utf-8-sig")

    # residual_df_val: validation row 전체 + target_cols 잔차
    df_val = df.loc[X_va.index].copy()
    res_df = pd.DataFrame(residuals_val, columns=target_cols, index=X_va.index)
    keep_cols = [prod_col] + ([dt_use] if dt_use in df_val.columns else [])
    residual_df_val = pd.concat([df_val[keep_cols], res_df], axis=1)

    return pred_all, metrics_df, residual_df_val, target_cols, dt_use


# =========================================================
# Tail-focused scenario sampling
# =========================================================
def _sample_scenarios_for_optimization(
    scenarios_re: np.ndarray,
    daily_capacity: float,
    n_sample: int = 30,
    tail_frac: float = 0.4,
    seed: int = 2025,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    S = scenarios_re.shape[0]
    if S <= n_sample:
        return scenarios_re

    D = scenarios_re.shape[2]
    total_demands = scenarios_re.sum(axis=(1, 2))  # (S,)
    cap_total = float(daily_capacity) * float(D)
    proxy_short = np.maximum(total_demands - cap_total, 0.0)

    tail_k = int(max(1, round(n_sample * float(tail_frac))))
    tail_idx = np.argsort(proxy_short)[-tail_k:]

    remain = n_sample - tail_k
    if remain > 0:
        sorted_idx = np.argsort(total_demands)
        q_idx = sorted_idx[np.linspace(0, S - 1, remain, dtype=int)]
    else:
        q_idx = np.array([], dtype=int)

    picked = np.unique(np.concatenate([tail_idx, q_idx], axis=0))
    if picked.size < n_sample:
        rest = np.setdiff1d(np.arange(S), picked, assume_unique=False)
        if rest.size > 0:
            add = rng.choice(rest, size=min(n_sample - picked.size, rest.size), replace=False)
            picked = np.unique(np.concatenate([picked, add], axis=0))

    if picked.size > n_sample:
        sub_sorted = picked[np.argsort(proxy_short[picked])]
        picked = sub_sorted[-n_sample:]

    return scenarios_re[picked]


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
    cvar_alpha: float | None = None,
    mc_seed: int = 2025,
    mc_wb: float = 1.0,
    mc_wi: float = 0.2,
    fail_threshold: float = 0.25,
    scale: int = 10,
    use_cvar_planner: bool = False,
    hint_plan_csv: str | None = None,
    lambda_cvar: float = 0.07,
    lambda_scale: int = 1_000_000,
    loss_scale: int = 2000,
    planner_max_time: float = 1800.0,
    planner_gap: float = 0.2,
    planner_workers: int = 8,
    planner_seed: int = 42,
    min_lot_map_json: str | None = None,
    safety_stock_map_json: str | None = None,
    weight_map_json: str | None = None,
    planner_use_scheduled: bool = False,
    planner_max_h: int = 4,
):
    if cvar_alpha is None:
        cvar_alpha = mc_alpha

    ensure_dir(out_dir)
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

    mc_validation = None

    # ---------- 1) Features ----------
    print("[1/6] Building features ...")
    raw_df = pd.read_csv(data_csv, encoding="utf-8")
    feat_df, _clus_summary = FE.build_features(raw_df)
    feat_df.to_csv(feat_csv, index=False, encoding="utf-8-sig")

    # ---------- 2) Forecast ----------
    print("[2/6] Forecasting (expected-only targets) ...")
    pred_all, metrics_df, residual_df_val, target_cols_all, dt_use = _run_forecast_python_api(
        feat_csv=feat_csv,
        out_pred_csv=forecast_csv,
        out_metrics_csv=forecast_metrics_csv,
        out_pred_by_product_csv=forecast_by_product_csv,
        prod_col=prod_col,
        dt_col=dt_col,
        best_params_path=best_params_path,
        planner_max_h=planner_max_h,
        seed=mc_seed,
    )

    # ---------- 2.5) MC scenarios (Full S) ----------
    if mc_scenarios and mc_scenarios > 0:
        print(f"[2.5/6] Generating MC demand scenarios (S={mc_scenarios}) ...")

        # (A) d_hat: 제품별 최신 스냅샷(=planner 입력 기준)
        pred_snap = pd.read_csv(forecast_by_product_csv)
        pred_snap = PO.preprocess_forecast(pred_snap, prod_col=prod_col, dt_col=dt_use)
        pred_snap[prod_col] = pred_snap[prod_col].astype(str).str.replace(r"\.0$", "", regex=True)

        # (B) residual pool: validation row 전체 사용 (축약 금지)
        res_pool_df = residual_df_val.copy()
        if prod_col in res_pool_df.columns:
            res_pool_df[prod_col] = res_pool_df[prod_col].astype(str).str.replace(r"\.0$", "", regex=True)

        # (C) MC horizon: planner와 동일 규칙으로 "예상"만 선택
        horizons_mc = _select_planner_horizons(
            pred_snap,
            prefer="예상",
            allow_scheduled=planner_use_scheduled,
            max_h=planner_max_h,
        )

        missing_cols = [c for c in horizons_mc if c not in res_pool_df.columns]
        if missing_cols:
            raise RuntimeError(
                "Residual pool missing some horizon columns for MC.\n"
                f"missing={missing_cols}\n"
                f"available(head)={list(res_pool_df.columns)[:50]}"
            )

        d_hat = pred_snap[horizons_mc].to_numpy(dtype=float)        # (P,D)
        res_pool = res_pool_df[horizons_mc].to_numpy(dtype=float)   # (N_val,D)

        # mean-centering
        res_pool = res_pool - np.nanmean(res_pool, axis=0, keepdims=True)

        scenarios = FO.generate_demand_scenarios(
            d_hat=d_hat,
            residuals=res_pool,
            n_scenarios=int(mc_scenarios),
            seed=int(mc_seed),
        )  # (S,P,D)

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

        pd.DataFrame(rows).to_csv(mc_scenarios_csv, index=False, encoding="utf-8-sig")
        print(f"[OK] Saved MC scenarios: {mc_scenarios_csv} | shape=(S={S}, P={P}, D={D})")
    else:
        print("[2.5/6] MC scenarios disabled")

    # ---------- 3) Planner ----------
    print("[3/6] Planning (CP-SAT) ...")
    cluster_info = PO.load_cluster_info(feat_csv, prod_col=prod_col)

    pred_df = pd.read_csv(forecast_by_product_csv)
    pred_df = PO.preprocess_forecast(pred_df, prod_col=prod_col, dt_col=dt_use)
    pred_df[prod_col] = pred_df[prod_col].astype(str).str.replace(r"\.0$", "", regex=True)

    horizons_planner = _select_planner_horizons(
        pred_df,
        prefer="예상",
        allow_scheduled=planner_use_scheduled,
        max_h=planner_max_h,
    )

    min_lot_map = _as_int_key_float_map(_load_json_map_or_none(min_lot_map_json))
    safety_stock_map = _as_int_key_float_map(_load_json_map_or_none(safety_stock_map_json))
    weight_map = _as_int_key_float_map(_load_json_map_or_none(weight_map_json))

    hint_plan = None
    if hint_plan_csv and os.path.exists(hint_plan_csv):
        try:
            hint_plan = PO.load_hint_plan_csv(hint_plan_csv, scale=scale, prod_col=prod_col)
        except TypeError:
            hint_plan = PO.load_hint_plan_csv(hint_plan_csv, scale=scale)

    scenarios_for_opt = None
    if use_cvar_planner:
        if not (mc_scenarios and mc_scenarios > 0) or not os.path.exists(mc_scenarios_csv):
            raise RuntimeError(
                "use_cvar_planner=True 인데 MC 시나리오 파일이 없습니다. "
                "먼저 --mc_scenarios > 0 으로 pred_final_mc.csv 를 생성해야 합니다."
            )

        scenarios, mc_products = PO.load_mc_scenarios(
            mc_npz=None,
            mc_csv=mc_scenarios_csv,
            product_col=prod_col,
        )

        forecast_products = pred_df[prod_col].tolist()
        scenarios_re = PO.reorder_mc_scenarios_to_forecast(scenarios, mc_products, forecast_products)

        if scenarios_re.shape[2] != len(horizons_planner):
            raise RuntimeError(
                f"MC horizon D({scenarios_re.shape[2]}) != planner horizons D({len(horizons_planner)}).\n"
                f"planner horizons={horizons_planner}"
            )

        if scenarios_re.shape[0] > 30:
            print(f"  -> Sampling 30 scenarios from {scenarios_re.shape[0]} for optimization (TAIL-AWARE)...")
            scenarios_for_opt = _sample_scenarios_for_optimization(
                scenarios_re=scenarios_re,
                daily_capacity=float(daily_capacity),
                n_sample=30,
                tail_frac=0.4,
                seed=int(mc_seed),
            )
        else:
            scenarios_for_opt = scenarios_re

    plan_df, diag = PO.optimize_plan(
        forecast_by_product=pred_df,
        horizons=horizons_planner,
        prod_col=prod_col,
        cluster_info=cluster_info,
        daily_capacity=daily_capacity,
        lambda_smooth=lambda_smooth,
        initial_inventory=initial_inventory,
        int_production=int_production,
        scale=scale,
        min_lot_map=min_lot_map,
        safety_stock_map=safety_stock_map,
        weight_map=weight_map,
        hint_plan=hint_plan,
        use_cvar_obj=use_cvar_planner,
        mc_scenarios=scenarios_for_opt,
        cvar_alpha=float(cvar_alpha),
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

    for c in ["demand", "produce", "end_inventory", "backlog", "shortage"]:
        if c in plan_df.columns:
            plan_df[c] = pd.to_numeric(plan_df[c], errors="coerce")
    if prod_col in plan_df.columns:
        plan_df[prod_col] = plan_df[prod_col].astype(str).str.replace(r"\.0$", "", regex=True)

    plan_df.to_csv(plan_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved plan: {plan_csv} | rows={len(plan_df)}")

    # ---------- 3.5) MC validation (Full S) ----------
    if mc_scenarios and mc_scenarios > 0:
        print(f"[3.5/6] MC validating fixed plan with FULL {mc_scenarios} scenarios ...")
        scenarios_df_full = EV.load_mc_scenarios(mc_scenarios_csv, product_col=prod_col)
        mc_validation = EV.mc_validate_plan(
            plan_df=plan_df,
            scenarios_df=scenarios_df_full,
            initial_inventory=float(initial_inventory),
            daily_capacity=float(daily_capacity),
            alpha=float(mc_alpha),
            w_b=float(mc_wb),
            w_i=float(mc_wi),
            fail_threshold=float(fail_threshold),
            product_col=prod_col,
        )
        save_json(mc_validation_json, mc_validation)
        print(f"[OK] Saved MC validation: {mc_validation_json}")

    # ---------- 4) Planning metrics ----------
    planning_kpi = M.compute_planning_metrics(
        plan_df=plan_df,
        daily_capacity=float(daily_capacity),
        feat_df=pd.read_csv(feat_csv),
        product_col=prod_col,
    )
    save_json(planning_metrics_json, planning_kpi)
    print(f"[OK] Saved planning metrics: {planning_metrics_json}")

    # ---------- 5) Evaluator / Governance ----------
    audit = EV.audit_and_learn(
        plan_df=plan_df,
        daily_capacity=float(daily_capacity),
        metrics_summary={"planning_metrics": planning_kpi, "daily_capacity": float(daily_capacity)},
        policy_path=policy_json,
        product_col=prod_col,
    )
    if isinstance(mc_validation, dict) and mc_validation.get("summary"):
        audit["mc_validation"] = mc_validation.get("summary")
    save_json(audit_json, audit)
    print(f"[OK] Saved audit: {audit_json}")

    # ---------- 6) Report ----------
    if not skip_llm:
        print("[6/6] Building weekly report ...")
        out = RL.build_report_with_llm(
            plan_csv=plan_csv,
            forecast_csv=forecast_by_product_csv,
            metrics_csv=forecast_metrics_csv,
            model_name=model_name,
            auto_regen_on_fail=True,
            feat_csv=feat_csv,
            mc_json=mc_validation_json if (mc_scenarios and mc_scenarios > 0) else None,
            planning_metrics_json=planning_metrics_json,
        )
        if isinstance(out, dict) and out.get("markdown"):
            ts = _now_str()
            out_md = os.path.join(reports_dir, f"weekly_report_{ts}.md")
            with open(out_md, "w", encoding="utf-8") as f:
                f.write(out["markdown"])
            out_html = str(Path(out_md).with_suffix(".html"))
            RL.md_to_html_with_charts(out_md, out_html, out.get("facts"), "주간 운영 계획 보고서")
            print(f"[OK] Saved report:\n- {out_md}\n- {out_html}")

    print("\n[DONE] Pipeline finished.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out_dir", default="./outputs")
    p.add_argument("--prod_col", default="Product_Number")
    p.add_argument("--dt_col", default=None)

    p.add_argument("--daily_capacity", type=int, default=10000)
    p.add_argument("--lambda_smooth", type=float, default=1.0)
    p.add_argument("--initial_inventory", type=float, default=0.0)
    p.add_argument("--int_production", action="store_true")

    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--policy_path", default=None)
    p.add_argument("--skip_llm", action="store_true")

    p.add_argument("--best_params_path", default="./configs/best_params.json")

    # MC / CVaR
    p.add_argument("--mc_scenarios", type=int, default=0)
    p.add_argument("--mc_alpha", type=float, default=0.9)
    p.add_argument("--cvar_alpha", type=float, default=None)
    p.add_argument("--mc_seed", type=int, default=2025)
    p.add_argument("--mc_wb", type=float, default=1.0)
    p.add_argument("--mc_wi", type=float, default=0.2)

    p.add_argument("--fail_threshold", type=float, default=0.25)

    # planner
    p.add_argument("--scale", type=int, default=10)
    p.add_argument("--use_cvar_planner", action="store_true")
    p.add_argument("--hint_plan_csv", default=None)
    p.add_argument("--lambda_cvar", type=float, default=0.07)
    p.add_argument("--lambda_scale", type=int, default=1_000_000)
    p.add_argument("--loss_scale", type=int, default=2000)
    p.add_argument("--planner_max_time", type=float, default=1800.0)
    p.add_argument("--planner_gap", type=float, default=0.2)
    p.add_argument("--planner_workers", type=int, default=8)
    p.add_argument("--planner_seed", type=int, default=42)
    p.add_argument("--min_lot_map", type=str, default=None)
    p.add_argument("--safety_stock_map", type=str, default=None)
    p.add_argument("--weight_map", type=str, default=None)

    # horizon 정책
    p.add_argument("--planner_use_scheduled", action="store_true")
    p.add_argument("--planner_max_h", type=int, default=4)

    return p.parse_args()


if __name__ == "__main__":
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
        cvar_alpha=args.cvar_alpha,
        mc_seed=args.mc_seed,
        mc_wb=args.mc_wb,
        mc_wi=args.mc_wi,
        fail_threshold=args.fail_threshold,
        scale=args.scale,
        use_cvar_planner=args.use_cvar_planner,
        hint_plan_csv=args.hint_plan_csv,
        lambda_cvar=args.lambda_cvar,
        lambda_scale=args.lambda_scale,
        loss_scale=args.loss_scale,
        planner_max_time=args.planner_max_time,
        planner_gap=args.planner_gap,
        planner_workers=args.planner_workers,
        planner_seed=args.planner_seed,
        min_lot_map_json=args.min_lot_map,
        safety_stock_map_json=args.safety_stock_map,
        weight_map_json=args.weight_map,
        planner_use_scheduled=args.planner_use_scheduled,
        planner_max_h=args.planner_max_h,
    )