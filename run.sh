#!/usr/bin/env bash
# =============================================================================
# AgentFactory E2E Runner for kamp/
# - venv 생성/활성화 → 의존성 설치 → 파이프라인 실행
# - 데이터 경로: kamp/data/data.csv
# - 결과물: kamp/outputs/
# =============================================================================
set -euo pipefail

# -----------------------------
# 0) 사용자 환경 변수 (필요 시 수정)
# -----------------------------
# LLM 리포팅을 쓰려면 OPENAI_API_KEY가 필요합니다.
# export OPENAI_API_KEY="sk-..."   # <== 필요 시 주석 해제하거나, 쉘에서 미리 export 해두기

# 실행 파라미터
DATA_PATH="${DATA_PATH:-data/data.csv}"        # 루트(kamp) 기준 경로
OUT_DIR="${OUT_DIR:-outputs}"                  # 결과물 폴더
MODEL="${MODEL:-xgboost}"                      # linear | xgboost
PLANNER="${PLANNER:-ortools}"                  # heuristic | ortools
REPORTER="${REPORTER:-llm}"                    # template | llm
LLM_MODEL="${LLM_MODEL:-gpt-4o-mini}"          # LLM 리포팅 모델명
ENCODING="${ENCODING:-utf-8}"                  # utf-8 | cp949
DAILY_CAPACITY="${DAILY_CAPACITY:-6000}"       # 일일 총 CAPA
MIN_LOT_SIZE="${MIN_LOT_SIZE:-0}"              # 최소 로트 크기
SAFETY_STOCK="${SAFETY_STOCK:-0}"              # 안전재고
INT_PRODUCTION="${INT_PRODUCTION:-true}"       # true면 정수생산 강제(OR-Tools)

# -----------------------------
# 1) 경로 설정
# -----------------------------
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"      # kamp/ 절대경로
CDIR="$ROOT_DIR"
AF_DIR="$ROOT_DIR/agentfactory"                # 코드 폴더(README대로 agentfactory/ 하위에 소스가 있다고 가정)
REQ_FILE="$AF_DIR/requirements.txt"
MAIN_PY="$AF_DIR/main.py"

echo "[INFO] ROOT_DIR: $ROOT_DIR"
echo "[INFO] AgentFactory DIR: $AF_DIR"
echo "[INFO] Data: $DATA_PATH"
echo "[INFO] Outputs: $OUT_DIR"

# 사전 체크
if [[ ! -f "$MAIN_PY" ]]; then
  echo "[ERROR] main.py not found at $MAIN_PY"
  exit 1
fi

if [[ ! -f "$REQ_FILE" ]]; then
  echo "[ERROR] requirements.txt not found at $REQ_FILE"
  exit 1
fi

if [[ ! -f "$ROOT_DIR/$DATA_PATH" ]]; then
  echo "[ERROR] Data file not found at $ROOT_DIR/$DATA_PATH"
  exit 1
fi

# -----------------------------
# 2) Python 선택/가상환경 준비
# -----------------------------
PY_BIN="${PY_BIN:-python3}"
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  echo "[WARN] $PY_BIN not found. Trying 'python' fallback."
  PY_BIN="python"
fi

# venv 생성
VENV_DIR="$ROOT_DIR/.venv"
if [[ ! -d "$VENV_DIR" ]]; then
  echo "[INFO] Creating venv at $VENV_DIR"
  "$PY_BIN" -m venv "$VENV_DIR"
fi

# venv 활성화
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

# pip 최신화 및 의존성 설치
python -m pip install --upgrade pip
pip install -r "$REQ_FILE"

# -----------------------------
# 3) 실행 옵션 구성
# -----------------------------
INT_FLAG=""
if [[ "$INT_PRODUCTION" == "true" ]] && [[ "$PLANNER" == "ortools" ]]; then
  INT_FLAG="--int_production"
fi

# -----------------------------
# 4) 파이프라인 실행
# -----------------------------
echo "[INFO] Running AgentFactory pipeline..."
set -x
python "$MAIN_PY" \
  --data "$DATA_PATH" \
  --out_dir "$OUT_DIR" \
  --model "$MODEL" \
  --planner "$PLANNER" \
  --reporter "$REPORTER" \
  --llm_model "$LLM_MODEL" \
  --encoding "$ENCODING" \
  --daily_capacity "$DAILY_CAPACITY" \
  --min_lot_size "$MIN_LOT_SIZE" \
  --safety_stock "$SAFETY_STOCK" \
  $INT_FLAG
set +x

echo "[OK] All done."
echo "Outputs written to: $OUT_DIR"
echo " - forecast_validation_metrics.csv"
echo " - forecast_by_product.csv"
echo " - production_plan.csv"
echo " - weekly_report.txt"

# -----------------------------
# 5) 유용한 프리셋(참고용, 실행 안 함)
# -----------------------------
: <<'__PRESETS__'
# (A) 빠른 베이스라인(휴리스틱 + 템플릿 리포트)
python agentfactory/main.py \
  --data data/data.csv \
  --out_dir outputs \
  --model linear \
  --planner heuristic \
  --reporter template

# (B) XGBoost + OR-Tools(정수 생산) + LLM 보고
python agentfactory/main.py \
  --data data/data.csv \
  --out_dir outputs \
  --model xgboost \
  --planner ortools \
  --int_production \
  --reporter llm \
  --llm_model gpt-4o-mini \
  --daily_capacity 6000
__PRESETS__
