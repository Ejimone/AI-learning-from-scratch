# milestone 2 leatning
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms

# device setup and pretrained ResNet-18 Model Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using Device {device}")


# load pretrained ResNet-18 Weights
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# freeze feature extraction layers
for param in model.parameters():
  param.requires_grad = False

# replace final FC layer for 2 classes (eg., "sports ball", "vehicle")
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model = model.to(device)
model.eval()

# 2. Image Transformations matching ResNet Pretraining
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Synthetic Video Stream Generator (Simulating 30 frames of video)
print("\n--- Generating Simulated Video Stream ---")
frame_height, frame_width = 480, 640
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out_video = cv2.VideoWriter(
    "annotated_output.avi", fourcc, 10.0, (frame_width, frame_height)
)

start_time = time.time()
for frame_idx in range(30):
  # generate synthetic frame with a moving simulated bounding box
  frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
  bx1, by1 = 100 + (frame_idx * 5), 100 + (frame_idx * 2)
  bx2, by2 = bx1 + 120, by1 + 120

  # draw target object on synthetic frame
  cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), -1)


  # extract ROI (Region of interest) and run pytorch inference
  roi = frame[by1:by2, bx1:bx2]
  if roi.size > 0:
    rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    input_tensor = preprocess(rgb_roi).unsqueeze(0).to(device)


    with torch.no_grad():
      output = model(input_tensor)
      pred_class = torch.argmax(output, dim=1).item()


    # draw bounding box and inference overlay

    label = f"target #{pred_class}"
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
    cv2.putText(
        frame,
        label,
        (bx1, by1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
      )

    # write annotated frame to video file

    out_video.write(frame)

out_video.release()
elapsed = time.time() - start_time

print(
    f"successfully processed 30 frames in {elapsed:.2f}s ({30/elapsed:.2f}FPS)"
)

print("Annotated video saved as 'annotated_output.avi'.")