import cv2
import numpy as np
import pyrealsense2 as rs
from flask import Flask, Response
from ultralytics import YOLO

class ObjectDetectionProcessor:
    def __init__(self, pipeline):
        # Guardamos el pipeline para obtener depth_scale
        self.pipeline = pipeline
        profile = pipeline.get_active_profile()
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()  # metros por unidad

        # Cargamos el modelo YOLO para TensorRT
        self.engine_path = "yolo_3rt_model.engine"
        self.model = YOLO(self.engine_path)
        self.class_names = self.model.names

    def calculate_distance(self, depth_frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Accedemos al frame bruto de profundidad y lo escalamos a metros
        raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        depth_m = raw * self.depth_scale

        # Preparamos el parche 5x5, clamp de coordenadas
        h, w = depth_m.shape
        xs = np.clip(np.arange(cx - 2, cx + 3), 0, w - 1)
        ys = np.clip(np.arange(cy - 2, cy + 3), 0, h - 1)
        patch = depth_m[np.ix_(ys, xs)]
        valid = patch[np.isfinite(patch) & (patch > 0.01)]  # >1cm

        if valid.size == 0:
            return None

        # Convertimos a centímetros
        return float(np.median(valid) * 100)

    def process_image(self, cv_image, depth_frame):
        results = self.model(cv_image)

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy().astype(int)

            for i, bbox in enumerate(boxes):
                if confidences[i] < 0.5:
                    continue

                distance_cm = self.calculate_distance(depth_frame, bbox)
                if distance_cm is None:
                    continue

                color = (255, 0, 0) if distance_cm < 40 else (0, 0, 255)
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    cv_image,
                    f"{self.class_names[labels[i]]}: {confidences[i]:.2f}, {distance_cm:.1f}cm",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

        return cv_image


# --- Inicialización de la cámara RealSense y servidor Flask ---
app = Flask(__name__)

# Pipeline y configuración de streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)

# Alineación depth->color
align = rs.align(rs.stream.color)

# Procesador de detección
processor = ObjectDetectionProcessor(pipeline)

# Helper: colormap para profundidad
def depth_to_colormap(depth_frame):
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.03),
        cv2.COLORMAP_JET
    )
    return depth_colormap

# Generadores de frames
def generate_rgb_frames():
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        img = np.asanyarray(color_frame.get_data())
        _, buf = cv2.imencode('.jpg', img)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')


def generate_depth_frames():
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue
        col = depth_to_colormap(depth_frame)
        _, buf = cv2.imencode('.jpg', col)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')


def generate_yolo_frames():
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        img = np.asanyarray(color_frame.get_data())
        processed = processor.process_image(img, depth_frame)
        _, buf = cv2.imencode('.jpg', processed)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

# Rutas Flask
@app.route('/')
def index():
    return '''
<html><head><title>RealSense YOLO Stream</title>
<style>
 body { background: black; color: white; display: flex; justify-content: center; align-items: center; height: 100vh; }
 .container { display: flex; gap: 10px; }
 .stream { text-align: center; }
</style></head><body>
  <div class="container">
    <div class="stream"><h2>RGB</h2><img src="/video_feed_rgb"></div>
    <div class="stream"><h2>Depth</h2><img src="/video_feed_depth"></div>
    <div class="stream"><h2>YOLO</h2><img src="/video_feed_yolo"></div>
  </div>
</body></html>
'''

@app.route('/video_feed_rgb')
def video_feed_rgb():
    return Response(generate_rgb_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_depth')
def video_feed_depth():
    return Response(generate_depth_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_yolo')
def video_feed_yolo():
    return Response(generate_yolo_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
