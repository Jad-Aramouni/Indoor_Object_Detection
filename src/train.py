from ultralytics import YOLO
from config import DATA_YAML, RUNS_DIR, MODEL_NAME, RUN_NAME, IMAGE_SIZE, BATCH_SIZE, EPOCHS, DEVICE
from utils import print_device_info, ensure_path_exists


def main() -> None:
    print_device_info()
    ensure_path_exists(DATA_YAML)

    model = YOLO(MODEL_NAME)

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        project=str(RUNS_DIR),
        name=RUN_NAME,
        device=DEVICE,
        pretrained=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()