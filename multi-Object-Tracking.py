import cv2
import numpy as np
import torch
from ultralytics import YOLO
import urllib.request
import os
import argparse


# command-line options: run the full video by default, optionally without the preview window
parser = argparse.ArgumentParser()
parser.add_argument("--max-frames", type=int, default=0, help="stop after this many frames (0 = whole video)")
parser.add_argument("--no-show", action="store_true", help="don't open the live preview window")
args = parser.parse_args()


# verify GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = YOLO("yolov8n.pt")  # Load a pretrained YOLOv8n model
model.to(device)  # Move the model to the GPU if available


# download sample video streams for testing
video_url = "https://media.geeksforgeeks.org/wp-content/uploads/20250910161721597148/sample.mp4"
base_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(base_dir, "sample_tracking.mp4")

if not os.path.exists(video_path):
    print("Downloading sample video stream...")
    urllib.request.urlretrieve(video_url, video_path)

# setup video processing and output writer
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out_path = os.path.join(base_dir, "annotated_tracking.avi")
out=cv2.VideoWriter(out_path, fourcc, fps, (width, height))


# Track unique IDs and history trails
unique_track_ids = set()

track_history = {}  # Dictionary to store the history of each track ID
print("\n--- Running ByteTrack Tracking Loop ---")
frame_count = 0

# Run tracking generator across video frames
results = model.track(source=video_path, show=not args.no_show, stream=True, tracker="bytetrack.yaml", conf=0.25)



for result in results:
    frame = result.orig_img
    boxes = result.boxes
    tracks = boxes.id if boxes is not None else None

    if boxes is not None and tracks is not None:
        # Extract bounding box coordinates and assigned tracking IDs
        xyxy = boxes.xyxy.cpu().numpy()  # Convert to numpy array
        track_ids = tracks.cpu().numpy()  # Convert to numpy array
        classes = boxes.cls.cpu().numpy()  # Convert to numpy array
        confs = boxes.conf.cpu().numpy()  # Convert to numpy array

        for box, track_id, cls, conf in zip(xyxy, track_ids, classes, confs):
            x1, y1, x2, y2 = map(int, box)
            track_id = int(track_id)
            cls = int(cls)
            unique_track_ids.add(track_id)

            # Center coordinates for trajectory history
            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            if track_id not in track_history:
                track_history[track_id] = []
            track_history[track_id].append((center_x, center_y))

            # Keep last 20 trajectory points
            if len(track_history[track_id]) > 20:
                track_history[track_id].pop(0)

            # Draw trajectory trail
            points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=(0, 255, 255), thickness=2)

            # Draw bounding box and track ID label
            label = f"ID:{track_id} {model.names[cls]} ({conf:.2f})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    # Overlay total unique tracks counter
    cv2.putText(
        frame,
        f"Total Unique IDs: {len(unique_track_ids)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )
    out.write(frame)
    frame_count += 1
    if frame_count % 100 == 0:
        print(f"Frame {frame_count}: {len(unique_track_ids)} unique IDs so far", flush=True)
    if args.max_frames and frame_count >= args.max_frames:
        break

cap.release()
out.release()
print(f"\nTracking Complete! Processed {frame_count} frames.")
print(f"Total Unique Object IDs Tracked: {len(unique_track_ids)}")
print(f"Annotated tracking video saved to '{out_path}'.")