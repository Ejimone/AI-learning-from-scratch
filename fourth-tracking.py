import os
from pathlib import Path

import torch
from ultralytics import YOLO
import roboflow


def load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE entries without overriding shell variables."""
    if not path.is_file():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("\"'")
        os.environ.setdefault(key.strip(), value)


# Device verification
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Milestone 5: Custom Dataset Fine-Tuning ({device}) ---")

load_dotenv(Path(__file__).resolve().parent / ".env")
roboflow_api_key = os.getenv("ROBOFLOW_API_KEY") or os.getenv("roboflow_api_key")
if not roboflow_api_key:
    raise RuntimeError(
        "ROBOFLOW_API_KEY is not configured. Set it in the environment or in a "
        "project .env file before downloading the dataset."
    )
roboflow_workspace = os.getenv("ROBOFLOW_WORKSPACE", "model-examples")
roboflow_project = os.getenv(
    "ROBOFLOW_PROJECT", "football-players-obj-detection"
)
roboflow_version = int(os.getenv("ROBOFLOW_VERSION", "2"))

# 2\. Download Custom Football Dataset from Roboflow Universe
dataset_root = Path(__file__).resolve().parent / "datasets"
dataset_url = dataset_root / "football-players-obj-detection-2"

if not dataset_url.exists():
    print("Downloading custom football dataset from Roboflow...")
    dataset_root.mkdir(parents=True, exist_ok=True)

    rf = roboflow.Roboflow(api_key=roboflow_api_key)
    project = rf.workspace(roboflow_workspace).project(roboflow_project)
    dataset = project.version(roboflow_version).download(
        "yolov8", location=str(dataset_url)
    )
    dataset_path = Path(dataset.location)
else:
    dataset_path = dataset_url

yaml_path = dataset_path / "data.yaml"


# 3\. Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # Load the pre-trained YO


# fine-tune the model on the custom dataset
model.train(data=yaml_path, epochs=100, imgsz=640, batch=16, device=device, project="football_model", name="yolov8n_football", exist_ok=True)
best_model_path = Path(model.trainer.save_dir) / "weights" / "best.pt"
if not best_model_path.is_file():
    raise FileNotFoundError(f"Training completed without creating {best_model_path}")



# 5\. Run Validation Benchmarking on Validation Split
print("Running validation benchmarking on the validation split...")

val_model = YOLO(best_model_path)  # Load the fine-tuned model
metrics = val_model.val(data=yaml_path)


#  6\. Print Benchmark Performance Summary
print("\\n==============================================")
print(" CUSTOM MODEL BENCHMARK RESULTS ")
print("==============================================")

print(f"Precision (P): {metrics.results_dict['metrics/precision(B)']:.4f}")
print(f"Recall (R): {metrics.results_dict['metrics/recall(B)']:.4f}")
print(f"mAP@0.5 (mAP50): {metrics.results_dict['metrics/mAP50(B)']:.4f}")
print(f"mAP@0.5:0.95 (mAP50-95): {metrics.results_dict['metrics/mAP50-95(B)']:.4f}")


print("\\n==============================================")
# Print Per-Class Performance
for i, name in enumerate(metrics.names):
    precision, recall, map50, map50_95 = metrics.class_result(i)
    print(f"Class: {name}")
    print(f"  Precision (P): {precision:.4f}")
    print(f"  Recall (R): {recall:.4f}")
    print(f"  mAP@0.5 (mAP50): {map50:.4f}")
    print(f"  mAP@0.5:0.95 (mAP50-95): {map50_95:.4f}")
    print("----------------------------------------------")


for i, name in val_model.names.items():
    print(f"Class [{name}]: Evaluation completed.")