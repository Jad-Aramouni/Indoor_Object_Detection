from ultralytics import YOLO
from config import RUNS_DIR, RUN_NAME, PREDICTIONS_DIR, CONF_THRESHOLD, PROJECT_ROOT
from utils import ensure_path_exists


def main() -> None:
    weights_path = RUNS_DIR / RUN_NAME / "weights" / "best.pt"
    source_path = PROJECT_ROOT / "data" / "test" / "images"

    ensure_path_exists(weights_path)
    ensure_path_exists(source_path)

    model = YOLO(str(weights_path))
    model.predict(
        source=str(source_path),
        save=True,
        project=str(PREDICTIONS_DIR.parent),
        name=PREDICTIONS_DIR.name,
        conf=CONF_THRESHOLD,
        device=0
    )

    print(f"Predictions saved to: {PREDICTIONS_DIR}")


if __name__ == "__main__":
    main()