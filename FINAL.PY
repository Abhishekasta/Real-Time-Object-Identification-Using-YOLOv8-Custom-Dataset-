import cv2
from ultralytics import YOLO


# dataset_path = "Objet Detection.v1i.yolov8"
# data_yaml_path = "data.yaml"


# model = YOLO("yolov8n.pt")

# model.train(data=data_yaml_path, epochs=50)
model = YOLO("runs/detect/train/weights/best.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    results = model(frame)

    
    annotated_frame = results[0].plot()  

    
    cv2.imshow("YOLOv8 Object Detection", annotated_frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
