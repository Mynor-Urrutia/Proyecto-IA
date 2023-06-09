# Importación de Librerias
from flask import Flask, render_template, Response, request
import cv2
import mediapipe as mp
import torch
import numpy as np

# Realizar captura de cámara
cap = cv2.VideoCapture(0)
ZONE = None  # Se inicializa ZONE como None

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

# Función Frames
def gen_frame():
    global ZONE  # Declarar la variable ZONE como global

    model = load_model()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if ZONE is None:
            height, width, _ = frame.shape
            ZONE = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.int32)

        preds = model(frame)
        bboxes = get_bboxes(preds)

        detections = 0
        for box in bboxes:
            xc = int((box[0] + box[2]) // 2)
            yc = int((box[1] + box[3]) // 2)

            # Verifica si la detección está dentro de la zona válida
            if cv2.pointPolygonTest(ZONE, (xc, yc), False) >= 0:
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

# Ruta para actualizar ZONE
@app.route('/update_zone', methods=['POST'])
def update_zone():
    global ZONE
    zone_str = request.form.get('zone')
    coordinates = zone_str.split(',')

    if len(coordinates) != 8:
        return 'Error: Las coordenadas deben tener 4 valores separados por comas.'

    try:
        ZONE = np.array([[int(coordinates[0]) * 0.8, int(coordinates[1]) * 0.8],
                         [int(coordinates[2]) * 0.8, int(coordinates[3]) * 0.8],
                         [int(coordinates[4]) * 0.8, int(coordinates[5]) * 0.8],
                         [int(coordinates[6]) * 0.8, int(coordinates[7]) * 0.8]], dtype=np.int32)
        return 'ZONE actualizado correctamente'
    except ValueError:
        return 'Error: Las coordenadas deben ser valores numéricos enteros.'

# Ejecutar index.hmtl
if __name__ == "__main__":
    app.run(debug=True)
