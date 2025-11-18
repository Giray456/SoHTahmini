from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "nasa_dataset"
MODELS_DIR = PROJECT_ROOT / "models"

MODELS_DIR.mkdir(exist_ok=True)

def get_mat_files():
    return sorted(DATA_DIR.glob("*.mat"))