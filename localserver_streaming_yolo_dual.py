import cv2
import numpy as np
import pyrealsense2 as rs
from flask import Flask, Response
from object_detection import ObjectDetectionProcessor
from ultralytics import YOLO
import os

app = Flask(__name__)

# Initialize the RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 640, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 640, rs.format.z16, 30)
pipeline.start(config)

# Load YOLO models
model_paths = [
    ("train_model/train/weights/best.pt", "train_model/train/weights/best_ncnn_model"),
    ("train_model/train2/weights/best.pt", "train_model/train2/weights/best_ncnn_model")
]

processors = []
for yolo_weights, yolo_path in model_paths:
    if not os.path.exists(yolo_path):
        model = YOLO(yolo_weights)
        model.export(format="ncnn")
    processors.append(ObjectDetectionProcessor(yolo_path))

# Function to map depth to colors

def depth_to_colormap(depth_frame):
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    return depth_colormap

# Generator for RGB frames
def generate_rgb_frames():
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        frame = np.asanyarray(color_frame.get_data())
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# Generator for depth frames
def generate_depth_frames():
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue
        depth_colormap = depth_to_colormap(depth_frame)
        _, buffer = cv2.imencode('.jpg', depth_colormap)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# Generator for YOLO processed frames
def generate_yolo_frames(processor):
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        frame = np.asanyarray(color_frame.get_data())
        processed_frame = processor.process_image(frame)
        _, buffer = cv2.imencode('.jpg', processed_frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# Routes for video streams
@app.route('/video_feed_rgb')
def video_feed_rgb():
    return Response(generate_rgb_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_depth')
def video_feed_depth():
    return Response(generate_depth_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_yolo')
def video_feed_yolo():
    return Response(generate_yolo_frames(processors[0]), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_yolo2')
def video_feed_yolo2():
    return Response(generate_yolo_frames(processors[1]), mimetype='multipart/x-mixed-replace; boundary=frame')

# HTML page to display streams
@app.route('/')
def index():
    return '''
    <html>
        <head>
            <title>Intel RealSense Streaming</title>
            <style>
                body { display: flex; justify-content: center; align-items: center; height: 100vh; background-color: black; color: white; }
                .container { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
                .stream { text-align: center; }
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
                    <h2>YOLO Model 1</h2>
                    <img src="/video_feed_yolo" width="640" height="480">
                </div>
                <div class="stream">
                    <h2>YOLO Model 2</h2>
                    <img src="/video_feed_yolo2" width="640" height="480">
                </div>
            </div>
        </body>
    </html>
    '''

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)