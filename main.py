from ultralytics import YOLO
import clearml

clearml.browser_login()
devicetorunon = 'cpu'  # Default device to run on the execution
# import torch
# print("Cuda is available: ", torch.cuda.is_available())
# if (torch.cuda.is_available()):
#    devicetorunon = 'gpu'
#    print("GPU is Available, thus running on GPU")
# else:
#    devicetorunon = 'cpu'
#    print("GPU is not available, thus running on CPU")

# Load the model. 
model = YOLO('yolov8n.pt')

# Training.
results = model.train(
   data='data_head.yaml',
   imgsz=640,
   epochs=20,
   batch=8,
   name='yolov8n_v8_head_detector',
   device=devicetorunon,
   project='Dataset/TrainedModel',
   optimizer=
)