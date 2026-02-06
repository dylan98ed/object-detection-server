# =============================================
# TODO
# Agregar formato engine y uso de GPU en Jetson
# =============================================


import cv2
import numpy as np
import pyrealsense2 as rs
from flask import Flask, Response, jsonify
from ultralytics import YOLO
import os
import threading
import time
from pathlib import Path

app = Flask(__name__)

# =============================================
# RealSense Camera Configuration
# =============================================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # depth kept, not visualized
pipeline.start(config)

# =============================================
# YOLO Model Initialization
# =============================================
CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parent.parent  # object-detection-server/

YOLO_WEIGHTS_PATH = REPO_ROOT / "train_model/train2/weights/best.pt"
YOLO_EXPORT_PATH = REPO_ROOT / "train_model/train2/weights/best_ncnn_model"

if not YOLO_EXPORT_PATH.exists():
    model = YOLO(YOLO_WEIGHTS_PATH.as_posix())
    model.export(format="ncnn")

# =============================================
# Object Detection Processor Class
# =============================================
class ObjectDetectionProcessor:
    """Handles object detection using YOLO (no distance, no depth visualization)"""

    def __init__(self, yolo_path):
        self.model = YOLO(yolo_path)
        self.class_names = self.model.names
        self.detections = []
        self.last_update_time = time.time()
        self.lock = threading.Lock()

    def process_image(self, color_image):
        new_detections = []

        results = self.model(color_image)

        for result in results:
            inference_ms = result.speed.get("inference", 0.0)

            boxes = result.boxes.xyxy.numpy()
            confidences = result.boxes.conf.numpy()
            labels = result.boxes.cls.numpy().astype(int)

            for i, bbox in enumerate(boxes):
                if confidences[i] < 0.5:
                    continue

                class_name = self.class_names[labels[i]]
                x_min, y_min, x_max, y_max = bbox.astype(int)

                # Draw bounding box
                cv2.rectangle(
                    color_image,
                    (x_min, y_min),
                    (x_max, y_max),
                    (0, 255, 0),
                    2
                )

                label = f"{class_name}: {confidences[i]:.2f}"
                cv2.putText(
                    color_image,
                    label,
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (255, 255, 255),
                    2
                )

                new_detections.append({
                    "class_name": class_name,
                    "confidence": float(confidences[i]),
                    "inference_ms": inference_ms
                })

        with self.lock:
            self.detections = new_detections
            self.last_update_time = time.time()

        return color_image


# =============================================
# Processor Initialization
# =============================================
processor = ObjectDetectionProcessor(YOLO_EXPORT_PATH.as_posix())

# =============================================
# Frame Generators
# =============================================
def generate_rgb_frames():
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if color_frame:
            frame = np.asanyarray(color_frame.get_data())
            _, buffer = cv2.imencode(".jpg", frame)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )

def generate_yolo_frames():
    align = rs.align(rs.stream.color)

    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()

        if color_frame:
            color_img = np.asanyarray(color_frame.get_data())
            processed = processor.process_image(color_img)
            _, buffer = cv2.imencode(".jpg", processed)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )

# =============================================
# Flask Routes
# =============================================
@app.route("/video_feed_rgb")
def video_feed_rgb():
    return Response(
        generate_rgb_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

@app.route("/video_feed_yolo")
def video_feed_yolo():
    return Response(
        generate_yolo_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

@app.route("/detections")
def get_detections():
    with processor.lock:
        return jsonify(
            {
                "detections": processor.detections,
                "timestamp": processor.last_update_time,
            }
        )

@app.route("/")
def index():
    return """
    <html>
    <head>
        <title>YOLO RealSense Server</title>
        <style>
            body { background:black; color:white; text-align:center; }
            .container { display:flex; justify-content:center; }
            .stream { margin:10px; }
            table { border-collapse:collapse; margin:auto; }
            th, td { border:1px solid white; padding:8px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="stream">
                <h3>RGB Stream</h3>
                <img src="/video_feed_rgb" width="640" height="480">
            </div>
            <div class="stream">
                <h3>YOLOv11n Stream</h3>
                <img src="/video_feed_yolo" width="640" height="480">
            </div>
        </div>

        <h2>Detected Objects</h2>
        <table>
            <thead>
                <tr>
                    <th>Object</th>
                    <th>Confidence</th>
                    <th>Inference (ms)</th>
                </tr>
            </thead>
            <tbody id="detections-body"></tbody>
        </table>

        <script>
            let lastTimestamp = 0;

            function updateDetections() {
                fetch("/detections")
                    .then(r => r.json())
                    .then(data => {
                        if (data.timestamp > lastTimestamp) {
                            lastTimestamp = data.timestamp;
                            const tbody = document.getElementById("detections-body");
                            tbody.innerHTML = "";
                            data.detections.forEach(d => {
                                const row = document.createElement("tr");
                                row.innerHTML = `
                                    <td>${d.class_name}</td>
                                    <td>${d.confidence.toFixed(2)}</td>
                                    <td>${d.inference_ms.toFixed(2)}</td>
                                `;
                                tbody.appendChild(row);
                            });
                        }
                    });
            }
            setInterval(updateDetections, 100);
        </script>
    </body>
    </html>
    """

# =============================================
# Main
# =============================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)