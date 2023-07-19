import os
from ultralytics import YOLO
import cv2

cap = cv2.VideoCapture(1)

ret, frame = cap.read()

# trained_model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')
trained_model_path1 = "Dataset/TrainedModel/yolov8n_v8_head_detector__DEFAULT_LEARNING_RATE_0.01/weights/best.pt" #lr=0.01
trained_model_path2 = "Dataset/TrainedModel/yolov8n_v8_head_detector2_LEARNING_RATE_0.1/weights/best.pt" #lr=0.1
trained_model_path3 = "Dataset/TrainedModel/yolov8n_v8_head_detector/weights/best.pt" #lr=0.001

# Load a model
model = YOLO(trained_model_path1)  # load a custom model

threshold = 0.5

while ret:

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()