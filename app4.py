from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
import cvzone
import math
import time

app = Flask(__name__)

def load_model():
    return YOLO("yolov8n.pt")


def calculate_fps(prev_time):
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    return fps, current_time

def detect_objects(model, frame, classNames, object_counts, detected_objects):
    preds = model(frame, stream=False)
    for r in preds:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(frame, (x1, y1, w, h))
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            cvzone.putTextRect(frame, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

            # Verificar si el objeto ya ha sido contado
            class_name = classNames[cls]
            if class_name not in detected_objects:
                detected_objects.append(class_name)
                # Incrementar el contador de objetos por clase
                if class_name in object_counts:
                    object_counts[class_name] += 1

    return frame, object_counts, detected_objects

def generate_frames():
    cap = cv2.VideoCapture(0)  # For Webcam
    model = load_model()
    classNames = [
        "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
        "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
        "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"
    ]
    prev_frame_time = time.time()
    object_counts = {class_name: 0 for class_name in classNames}  # Contador de objetos por clase

    while True:
        ret, frame = cap.read()
        detected_objects = []  # Lista de objetos detectados en la iteración actual
        frame, object_counts, detected_objects = detect_objects(model, frame, classNames, object_counts, detected_objects)
        fps, prev_frame_time = calculate_fps(prev_frame_time)
        print(fps)
        print(object_counts)  # Imprimir los conteos de objetos por clase

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)