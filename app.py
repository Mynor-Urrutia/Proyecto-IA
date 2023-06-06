# Importación de Librerias
from flask import Flask, render_template, Response
import cv2
import mediapipe as np
import torch
import numpy as np
import matplotlib.path as mplPath

# Realizar captura de cámara
cap = cv2.VideoCapture(0)

ZONE = np.array([
    [200, 300],
    [403, 470],
    [476, 655],
    [498, 710],
    [1237, 714],
    [1217, 523],
    [1139, 469],
    [1009, 393],
])

def get_center(bbox):
    # Obtiene el centro del cuadro delimitador
    center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
    return center

def load_model():
    # Carga el modelo de detección de objetos YOLOv5
    model = torch.hub.load("ultralytics/yolov5", model="yolov5s", pretrained=True)
    return model

def get_bboxes(preds):
    # Obtiene los cuadros delimitadores de las detecciones
    df = preds.pandas().xyxy[0]
    df = df[df["confidence"] >= 0.5]
    df = df[df["name"] == "person"]
    return df[["xmin", "ymin", "xmax", "ymax"]].values.astype(int)

def is_valid_detection(xc, yc):
    # Verifica si la detección está dentro de la zona válida
    return mplPath.Path(ZONE).contains_point((xc, yc))

# Función Frames
def gen_frame():
    model = load_model()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        preds = model(frame)
        bboxes = get_bboxes(preds)

        detections = 0
        for box in bboxes:
            xc, yc = get_center(box)

            if is_valid_detection(xc, yc):
                detections += 1

            # Dibuja el centro y el cuadro delimitador en el frame
            cv2.circle(img=frame, center=(xc, yc), radius=5, color=(0, 255, 0), thickness=-1)
            cv2.rectangle(img=frame, pt1=(box[0], box[1]), pt2=(box[2], box[3]), color=(255, 0, 0), thickness=1)

        # Agrega el número de detecciones de personas en el frame
        cv2.putText(img=frame, text=f"Cars: {detections}", org=(100, 100), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3,
                    color=(0, 0, 0), thickness=3)
        # Dibuja la zona válida en el frame
        cv2.polylines(img=frame, pts=[ZONE], isClosed=True, color=(0, 0, 255), thickness=4)

        _, jpeg = cv2.imencode('.jpg', frame)
        frame_data = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

# Creamos la app
app = Flask(__name__)

# Ruta Principal
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Ejecutar index.hmtl
if __name__ == "__main__":
    app.run(debug=True)
