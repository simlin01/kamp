# ## agentfactory/config.py
# from dataclasses import dataclass

# @dataclass
# class Config:
#     # Planning horizon is inferred from columns (T, T+1, ..., T+k)
#     daily_capacity: int = 5000          # 총 일일 생산능력
#     min_lot_size: int = 0               # 최소 로트 크기(선택)
#     safety_stock: int = 0               # 안전재고(선택)
#     model_type: str = "linear"          # "linear" | "xgboost"
#     encoding: str = "utf-8"             # 입력 데이터 인코딩

# KAMP/config.py
from dataclasses import dataclass
from typing import Optional
import os

try:
    # python-dotenv가 설치되어 있으면 .env 자동 로드
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # 패키지가 없을 때도 문제 없이 동작하게 무시
    pass


@dataclass
class Config:
    """
    공통 설정 컨테이너
    - 생산/예측 파라미터는 dataclass 필드로 관리
    - OpenAI API 키는 .env(OPENAI_API_KEY)에서 읽어서 보관
    """
    # 생산/예측 관련 파라미터
    daily_capacity: int = 5000          # 총 일일 생산능력
    min_lot_size: int = 0               # 최소 로트
    safety_stock: int = 0               # 안전재고
    model_type: str = "linear"          # "linear" | "xgboost"
    encoding: str = "utf-8"             # 입력 CSV 인코딩

    # API 키 (옵션: 제공 안 하면 .env에서 읽음)
    openai_api_key: Optional[str] = None

    def __post_init__(self):
        # 인자로 안 들어오면 환경변수에서 읽어 채움
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")

    def ensure_api_key(self) -> str:
        """
        LLM 리포터 사용 시 유용: 키가 없으면 친절한 에러를 던짐.
        """
        if not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY가 설정되지 않았습니다. "
                ".env 파일에 OPENAI_API_KEY=sk-... 형식으로 추가하세요."
            )
        return self.openai_api_key