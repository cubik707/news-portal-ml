import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

DB_HOST: str = os.getenv("DB_HOST", "localhost")
DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
DB_NAME: str = os.getenv("DB_NAME", "news_portal")
DB_USER: str = os.getenv("DB_USERNAME", os.getenv("DB_USER", "postgres"))
DB_PASSWORD: str = os.getenv("DB_PASSWORD", "postgres")

ALPHA: float = float(os.getenv("ALPHA", "0.7"))
LAMBDA_DECAY: float = float(os.getenv("LAMBDA_DECAY", "0.05"))

ML_PORT: int = int(os.getenv("ML_PORT", "8000"))
