from ultralytics import YOLO
from config import DATA_YAML, RUNS_DIR, RUN_NAME
from utils import ensure_path_exists


def main() -> None:
    weights_path = RUNS_DIR / RUN_NAME / "weights" / "best.pt"

    ensure_path_exists(DATA_YAML)
    ensure_path_exists(weights_path)

    model = YOLO(str(weights_path))
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        device=0
    )

    print(metrics)


if __name__ == "__main__":
    main()