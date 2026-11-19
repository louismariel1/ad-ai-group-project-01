from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_DIR = PROJECT_ROOT / "config"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = PROJECT_ROOT / "results"
FILES_DIR = PROJECT_ROOT / "files"
DATA_DIR = PROJECT_ROOT / "data"

PROJECT_CONFIG_PATH = CONFIG_DIR / "project_config.json"
PIPELINES_PATH = CONFIG_DIR / "pipelines.json"
TARGET_SPLIT_PATH = CONFIG_DIR / "dataset_splits.json"
SSL_SPLIT_PATH = CONFIG_DIR / "ssl_dataset_splits.json"

SSL_CKPT_DIR = CHECKPOINTS_DIR / "ssl"
TEACHER_CKPT_DIR = CHECKPOINTS_DIR / "teachers"
STUDENT_CKPT_DIR = CHECKPOINTS_DIR / "students"

FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
TABLES_DIR = RESULTS_DIR / "tables"

LAYOUT_DIRS = [
    CONFIG_DIR,
    SSL_CKPT_DIR,
    TEACHER_CKPT_DIR,
    STUDENT_CKPT_DIR,
    EXPERIMENTS_DIR / "RQ1",
    EXPERIMENTS_DIR / "RQ2",
    EXPERIMENTS_DIR / "RQ3",
    RESULTS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    TABLES_DIR,
    FILES_DIR,
    DATA_DIR,
]


def ensure_layout() -> None:
    for path in LAYOUT_DIRS:
        path.mkdir(parents=True, exist_ok=True)
        if path != CONFIG_DIR:
            gitkeep = path / ".gitkeep"
            if not gitkeep.exists():
                gitkeep.touch()
