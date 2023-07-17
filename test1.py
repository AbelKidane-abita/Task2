import clearml
clearml.browser_login()

# api { 
#     # Abel Kidane's workspace
#     web_server: https://app.clear.ml
#     api_server: https://api.clear.ml
#     files_server: https://files.clear.ml
#     credentials {
#         "access_key" = "TDL48EHB3V6CPLGY7PJF"
#         "secret_key"  = "D0wXBERSgWDSWVImN5nV4hzPWgaB3DqynSm2ceubJjTXjzYYJn"
#     }
# }

from ultralytics import YOLO
 
# Load the model.
model = YOLO('yolov8n.pt')
 
# Training.
results = model.train(
   data='data_head.yaml',
   project='Dataset/TrainedModel',
   imgsz=480,
   epochs=10,
   batch=2,
   name='yolov8n_v8_head_detector',
   device= 'cpu'
)
