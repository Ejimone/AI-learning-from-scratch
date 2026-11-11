import cv2
import numpy as np
import argparse
from ultralytics import YOLO

# load pretrained detector
print("-- Milestone 6: Realtime Team Color Classification on Live Camera Feed --")
model = YOLO("yolov8n.pt")  # load a pretrained YOLOv8n model

# 2. Team Color Classification Function
def classify_team_color(crop):
    if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
        return (0, 255, 0), "Unknown"  # Return green for empty crops

    # crop upper 50% (Torso)
    h, w = crop.shape[:2]
    torso = crop[0 : int(h * 0.5), :]

    # Convert to BGR to HSV & mask out green grass
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Extract non-green pixels
    foreground_pixels = torso[mask == 0]


    if len(foreground_pixels) == 0:
        return (0, 255, 0), "Unknown"  # Return green for empty crops


    # compute mean BGR color of foreground jersey
    mean_color = np.mean(foreground_pixels, axis=0) # [B, G, R]

    # Simple heuristic split: Red dominance vs Blue dominance
    b, g, r = mean_color
    if r > g and r > b:
        return (0, 0, 255), "Red Team"
    elif b > g and b > r:
        return (255, 0, 0), "Blue Team"
    else:
        return (0, 255, 0), "Unknown"  # Return green for unknown colors



# process video stream
parser = argparse.ArgumentParser(description="Classify teams from a live camera stream")
parser.add_argument(
    "--source",
    default="0",
    help="Camera index or IP camera stream URL (for example: http://192.168.1.10:8080/video)",
)
args = parser.parse_args()
video_source = int(args.source) if args.source.isdigit() else args.source
cap = cv2.VideoCapture(video_source)


frame_idx = 0

print("\n---Running Live Player Detection and Team Classification---")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # run YOLO player detection
    results = model.predict(frame, conf=0.5, verbose=False)


    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]

            # classify team color
            team_color, team_name = classify_team_color(crop)

            # draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), team_color, 2)
            cv2.putText(
                frame,
                team_name,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                team_color,
                2,
            )
    frame_idx += 1


    # display the frame
    cv2.imshow("Live Player Detection and Team Classification", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
print(f"Successfully processed {frame_idx} frames with live team classification.p")