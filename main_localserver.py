import cv2
import numpy as np
import pyrealsense2 as rs
from flask import Flask, Response
from object_detection import ObjectDetectionProcessor  # Procesador YOLO
from ultralytics import YOLO
import os

app = Flask(__name__)

# Inicializar la cámara RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# Cargar YOLO
yolo_weights = "train_model/train2/weights/best.pt"
yolo_path = "train_model/train2/weights/best_ncnn_model"
if not os.path.exists(yolo_path):
    model = YOLO(yolo_weights)
    model.export(format="ncnn")
processor = ObjectDetectionProcessor(yolo_path)

# Función para escalar mapa de profundidad
def depth_to_colormap(depth_frame):
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    return depth_colormap

# Generador de frames RGB
def generate_rgb_frames():
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        frame = np.asanyarray(color_frame.get_data())
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# Generador de frames de profundidad
def generate_depth_frames():
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue
        depth_colormap = depth_to_colormap(depth_frame)
        _, buffer = cv2.imencode('.jpg', depth_colormap)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# Generador de frames procesados con YOLO
def generate_yolo_frames():
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        frame = np.asanyarray(color_frame.get_data())
        processed_frame = processor.process_image(frame)
        _, buffer = cv2.imencode('.jpg', processed_frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# Rutas para los streams
@app.route('/video_feed_rgb')
def video_feed_rgb():
    return Response(generate_rgb_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_depth')
def video_feed_depth():
    return Response(generate_depth_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_yolo')
def video_feed_yolo():
    return Response(generate_yolo_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Página HTML para ver los tres streams
@app.route('/')
def index():
    return '''
    <html>
        <head>
            <title>Intel RealSense Streaming</title>
            <style>
                body { display: flex; justify-content: center; align-items: center; height: 100vh; background-color: black; color: white; }
                .container { display: flex; flex-direction: row; }
                .stream { margin: 10px; text-align: center; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="stream">
                    <h2>RGB Stream</h2>
                    <img src="/video_feed_rgb" width="640" height="480">
                </div>
                <div class="stream">
                    <h2>Depth Stream</h2>
                    <img src="/video_feed_depth" width="640" height="480">
                </div>
                <div class="stream">
                    <h2>Processed Stream (YOLO)</h2>
                    <img src="/video_feed_yolo" width="640" height="480">
                </div>
            </div>
        </body>
    </html>
    '''

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
