from ultralytics import YOLO
import torch

import clearml 
clearml.browser_login()

# devicetorunon = "cuda" # Default device to run on the execution

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU {torch.cuda.get_device_name(0)} is available")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU instead")
    devicetorunon = "cpu"

# Load the model. 
model = YOLO('yolov8n.pt')

# Training.
results = model.train(
   data='data_head.yaml',
   imgsz=640,
   epochs=20, 
   batch=128,
   name='yolov8n_v8_head_detector',
   project='Dataset/TrainedModel',
   optimizer='Adam'
)
#    device=devicetorunon,