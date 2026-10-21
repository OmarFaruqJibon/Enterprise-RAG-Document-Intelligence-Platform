from functools import lru_cache
from pathlib import Path

from qdrant_client import QdrantClient


BASE_DIR = Path(__file__).resolve().parents[2]
QDRANT_PATH = BASE_DIR / "qdrant_data"


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:

    QDRANT_PATH.mkdir(parents=True, exist_ok=True)

    return QdrantClient(
        path=str(QDRANT_PATH)
    )