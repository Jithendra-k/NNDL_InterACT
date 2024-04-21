from ultralytics import YOLO
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
# Initialize a YOLO-World model
model = YOLO('yolov8s-worldv2.pt').to(device)  # or choose yolov8m/l-world.pt

# Define custom classes
model.set_classes(["laptop"])

# Execute prediction for specified categories on an image
results = model.predict('room-1.jpeg')
results[0].show()