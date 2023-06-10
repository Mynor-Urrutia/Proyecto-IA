# Importación de Librerias
from flask import Flask, render_template, Response, request, redirect, url_for, send_file
import cv2
import mediapipe as mp
import torch
import numpy as np
import time
import csv
import os

# Realizar captura de cámara
cap = cv2.VideoCapture(0)
ZONE = None  # Se inicializa ZONE como None

objetos = ["person", "car", "cell phone"]
selected_name = 0
prueba = 0

data = []  # Lista para almacenar los datos

# Nombre del archivo CSV y encabezados de las columnas
filename = "data/datos.csv"
header = ["Tiempo", "Detecciones", "Tipo de objeto"]

detection_active = True


def load_model():
    # Carga el modelo de detección de objetos YOLOv5
    model = torch.hub.load("ultralytics/yolov5", model="yolov5s", pretrained=True)
    return model


def get_bboxes(preds, selected_name):
    # Obtiene los cuadros delimitadores de las detecciones
    df = preds.pandas().xyxy[0]
    df = df[df["confidence"] >= 0.5]
    df = df[df["name"] == selected_name]
    return df[["xmin", "ymin", "xmax", "ymax"]].values.astype(int)

# Función Frames
def gen_frame(selected_name):
    global ZONE
    global prueba
    global data
    global detection_active  # Agregar la variable de control

    model = load_model()
    start_time = time.time()

    while detection_active:
        ret, frame = cap.read()

        if not ret:
            break

        elapsed_time = int(time.time() - start_time)

        if ZONE is None:
            height, width, _ = frame.shape
            ZONE = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.int32)

        preds = model(frame)
        bboxes = get_bboxes(preds, selected_name)

        detections = 0
        for box in bboxes:
            xc = int((box[0] + box[2]) // 2)
            yc = int((box[1] + box[3]) // 2)

            if cv2.pointPolygonTest(ZONE, (xc, yc), False) >= 0:
                detections += 1

            # Dibuja el centro y el cuadro delimitador en el frame
            cv2.circle(img=frame, center=(xc, yc), radius=5, color=(0, 255, 0), thickness=-1)
            cv2.rectangle(img=frame, pt1=(box[0], box[1]), pt2=(box[2], box[3]), color=(255, 0, 0), thickness=1)

        if detections != prueba:  # Se produjo un cambio en el número de detecciones
            elapsed_time = int(time.time() - start_time)  # Calcula el tiempo transcurrido
            print(f"{selected_name} dentro del area: {detections}, Tiempo transcurrido: {elapsed_time} segundos")
            prueba = detections  # Actualiza el número de detecciones previo

        data.append([selected_name, detections, elapsed_time])

            # Agrega el número de detecciones de personas en el frame
        cv2.putText(img=frame, text=f"{selected_name}: {detections}", org=(100, 100), fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=3,
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
    return render_template('index.html', objetos=objetos)

@app.route('/video')
def video():
    return Response(gen_frame(selected_name), mimetype='multipart/x-mixed-replace; boundary=frame')

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

# Ruta para actualizar el valor seleccionado y obtener los cuadros delimitadores correspondientes
@app.route('/update_name', methods=['POST'])
def update_name():
    global selected_name
    selected_name = request.form['selected_name']
    print("Selected name:", selected_name)  # Mostrar el dato por consola
    return redirect(url_for('index'))

# Función para guardar los datos en el archivo CSV
def save_data_to_csv(data):
    if os.path.exists(filename):
        os.remove(filename)

    # Guardar los nuevos datos en el archivo CSV
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)

    print(f"Los datos se han guardado en el archivo {filename}.")

@app.route('/stop_detection')
def stop_detection():
    global detection_active
    detection_active = False
    return redirect(url_for('index'))

@app.route('/reset_detection')
def reset_detection():
    global detection_active
    global data
    detection_active = True
    data = []
    return redirect(url_for('index'))

@app.route('/download_csv')
def download_csv():
    filename = "data/datos.csv"
    # Guardar los datos en el archivo CSV antes de la descarga
    save_data_to_csv(data)
    return send_file(filename, as_attachment=True)

# Ejecutar index.hmtl
if __name__ == "__main__":
    app.run(debug=True, threaded=True, use_reloader=False)

    # Guardar los datos en el archivo CSV al finalizar la ejecución del programa
    save_data_to_csv(data)
