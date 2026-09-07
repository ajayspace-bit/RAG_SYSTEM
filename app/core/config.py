from pathlib import Path
import os

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_GENERATION_MODEL = os.getenv("GEMINI_GENERATION_MODEL", "gemini-3.6-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
CHROMA_PATH = PROJECT_ROOT / os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "employee_handbook")
ASSETS_PATH = PROJECT_ROOT / os.getenv("ASSETS_PATH", "./assets")

# The existing workspace uses `asset`; keep it usable while the documented
# project layout uses `assets`.
LEGACY_ASSETS_PATH = PROJECT_ROOT / "asset"


def get_assets_path() -> Path:
    """Return the configured assets directory, with a legacy fallback."""
    if list(ASSETS_PATH.glob("*.pdf")):
        return ASSETS_PATH
    if list(LEGACY_ASSETS_PATH.glob("*.pdf")):
        return LEGACY_ASSETS_PATH
    return ASSETS_PATH