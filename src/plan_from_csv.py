# src/plan_from_csv.py
import argparse, os, sys, re, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path: sys.path.append(HERE)

from planner import plan_production  # 기본 휴리스틱 사용

def smart_read_csv(path: str, encoding: str = "utf-8"):
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding=encoding)
    except Exception:
        for enc in ("utf-8-sig", "cp949"):
            try:
                return pd.read_csv(path, sep=None, engine="python", encoding=enc)
            except Exception:
                pass
        return pd.read_csv(path, encoding="utf-8")

def detect_cols(df: pd.DataFrame):
    # 제품컬럼 후보
    prod_col = next((c for c in ["Product_Number","product","SKU","품번","제품","상품코드","상품"] if c in df.columns), df.columns[0])
    # horizon: T일 / T+N일 + (예정|예상) + 수주량
    pat = re.compile(r"^(T(?:\+\d+)?일)\s*(예정|예상)\s*수주량$")
    hcols = [c for c in df.columns if pat.match(str(c).strip())]
    if not hcols:
        # 백업
        hcols = [c for c in df.columns if "수주량" in str(c) and ("T일" in str(c) or "T+" in str(c))]
    if not hcols:
        raise RuntimeError("horizon 컬럼을 찾지 못했습니다. (예: 'T일 예상 수주량' 등)")

    def _hidx(c):
        s = str(c)
        if "T일" in s and "T+" not in s: return 0
        m = re.search(r"T\+(\d+)일", s); return int(m.group(1)) if m else 9999
    hcols = sorted(hcols, key=_hidx)
    return prod_col, hcols

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", default="./outputs/production_plan.csv")
    ap.add_argument("--planner", default="heuristic", choices=["heuristic","ortools"])
    ap.add_argument("--daily_capacity", type=int, default=6000)
    ap.add_argument("--min_lot_size", type=int, default=0)
    ap.add_argument("--safety_stock", type=int, default=0)
    ap.add_argument("--int_production", action="store_true")
    ap.add_argument("--encoding", default="utf-8")
    ap.add_argument("--feat_csv", required=True, help="feat.csv 경로 (Cluster 포함)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df = smart_read_csv(args.in_csv, encoding=args.encoding)

    # 음수 방지
    for c in df.columns:
        if c != "Product_Number" and hasattr(df[c], "dtype") and df[c].dtype.kind in "if":
            df[c] = df[c].clip(lower=0)

    prod_col, horizons = detect_cols(df)
    print(f"[INFO] prod_col={prod_col}")
    print(f"[INFO] horizons={horizons}")

    if args.planner == "heuristic":
        plan_df = plan_production(df, horizons, prod_col,
                                  daily_capacity=args.daily_capacity,
                                  min_lot_size=args.min_lot_size,
                                  safety_stock=args.safety_stock)
    else:
        from planner_opt import optimize_plan, load_cluster_info
        cluster_info = load_cluster_info(args.feat_csv)
        plan_df = optimize_plan(
                                forecast_by_product=df,
                                horizons=horizons,
                                prod_col=prod_col,
                                cluster_info=cluster_info,
                                daily_capacity=args.daily_capacity,
                                lambda_smooth=1.0,
                                initial_inventory=0,
                                int_production=True,
                                scale=100
                            )

    plan_df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print("[OK] wrote plan ->", args.out_csv)

if __name__ == "__main__":
    main()
