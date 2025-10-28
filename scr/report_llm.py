# src/report_llm.py
import os
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

def _read_clip(path: str, max_chars: int) -> str:
    if not os.path.exists(path):
        return f"[MISSING] {path}"
    txt = pd.read_csv(path).to_csv(index=False)
    if len(txt) > max_chars:
        txt = txt[:max_chars] + f"\n...[truncated to {max_chars} chars]"
    return txt

def build_report_with_llm(
    plan_csv: str,
    forecast_csv: str,
    metrics_csv: str,
    model_name: str = "gpt-4o-mini",
    max_chars: int = 16000
) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 환경변수를 설정하세요.")

    plan_txt = _read_clip(plan_csv, max_chars)
    forecast_txt = _read_clip(forecast_csv, max_chars)
    metrics_txt = _read_clip(metrics_csv, max_chars)

    sys = SystemMessage(content=(
        "You are an operations planning analyst AI. "
        "Summarize supply-chain production plan and forecasting quality for a weekly executive report. "
        "Be concise, numeric, and actionable. Provide bullet points and sections."
    ))

    user = HumanMessage(content=f"""
[FILES]
[PRODUCTION_PLAN_CSV]
{plan_txt}

[FORECAST_BY_PRODUCT_CSV]
{forecast_txt}

[FORECAST_METRICS_CSV]
{metrics_txt}

[TASK]
1) 요약(기간, CAPA, 총 수요/생산/백로그)
2) 제품 Top5: 증산 필요 / 과다생산
3) 예측성능: Horizon별 MAE/R2 요약
4) 액션아이템: 3~5개 (CAPA 재배분, 로트/안전재고 조정 등)
5) 리스크/가정: 데이터 품질/제약 가정 간단 표기
""")

    llm = ChatOpenAI(model=model_name, temperature=0.2)
    out = llm([sys, user]).content
    return out
