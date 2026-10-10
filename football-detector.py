import os
import urllib.request
from ultralytics import YOLO


# 1. Verify GPU Availability & install dependencies
print("--- Milestone 3: Custom YOLO Object Detection ---")

# 2. Download a sample image to run predictions on
data_dir = "football_dataset"
sample_path = os.path.join(data_dir, "sample.jpg")

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if not os.path.exists(sample_path):
    print("Downloading sample image...")
    urllib.request.urlretrieve(
        "https://media.roboflow.com/notebooks/examples/dog.jpeg",
        sample_path
    )

# 3. Load pretrained YOLO model (YOLOv8 Nano)
model = YOLO("yolov8n.pt")  # loads official COCO pretrained weights

# 4. Run Pretrained Prediction Sample
print("\n--- Running Pretrained Prediction ---")
results = model.predict(source=sample_path, conf=0.25, save=True)
for r in results:
    print("Detected Classes:", [model.names[int(c)] for c in r.boxes.cls])
    print("Confidence Scores:", r.boxes.conf.tolist())
    print("Bounding Box Coordinates (xyxy):\n", r.boxes.xyxy)

# 5. Train YOLO on Custom Dataset (COCO8 Example Dataset for Instant Execution)
print("\n--- Starting Custom Model Training ---")
# Uses coco8.yaml built-in dataset to verify complete training loop instantly
model.train(data="coco8.yaml", epochs=3, imgsz=640, batch=8, project="runs/detect", name="custom_football")

# 6. Validate Custom Model & Print mAP Metrics
print("\n--- Evaluating Model Performance (Validation) ---")
val_results = model.val(data="coco8.yaml")

print("\n--- Validation Metrics ---")
print(f"Precision (P): {val_results.results_dict['metrics/precision(B)']:.4f}")
print(f"Recall (R): {val_results.results_dict['metrics/recall(B)']:.4f}")
print(f"mAP50: {val_results.results_dict['metrics/mAP50(B)']:.4f}")
print(f"mAP50-95: {val_results.results_dict['metrics/mAP50-95(B)']:.4f}")
