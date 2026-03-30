from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_YAML = PROJECT_ROOT / "data" / "data.yaml"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RUNS_DIR = OUTPUTS_DIR / "runs"
PREDICTIONS_DIR = OUTPUTS_DIR / "sample_predictions"

MODEL_NAME = "yolov8n.pt"
RUN_NAME = "indoor_objects_yolov8n_gpu_768_more_epochs"

IMAGE_SIZE = 768
BATCH_SIZE = 16
EPOCHS = 60

DEVICE = 0 if torch.cuda.is_available() else "cpu"
CONF_THRESHOLD = 0.25