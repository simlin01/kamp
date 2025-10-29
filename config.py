## agentfactory/config.py
from dataclasses import dataclass

@dataclass
class Config:
    # Planning horizon is inferred from columns (T, T+1, ..., T+k)
    daily_capacity: int = 5000          # 총 일일 생산능력
    min_lot_size: int = 0               # 최소 로트 크기(선택)
    safety_stock: int = 0               # 안전재고(선택)
    model_type: str = "linear"          # "linear" | "xgboost"
    encoding: str = "utf-8"             # 입력 데이터 인코딩
