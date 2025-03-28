import cv2
import numpy as np
import pyrealsense2 as rs
from flask import Flask, Response
from object_detection import ObjectDetectionProcessor
from ultralytics import YOLO
import os

app = Flask(__name__)

# Initialize RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Enable RGB stream
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Enable Depth stream
pipeline.start(config)

# Load YOLO model
yolo_weights = "train_model/train2/weights/best.pt"
yolo_path = "train_model/train2/weights/best_ncnn_model"

# Check if the exported NCNN model exists, otherwise export it
if not os.path.exists(yolo_path):
    model = YOLO(yolo_weights)
    model.export(format="ncnn")

# Initialize object detection processor with YOLO model
processor = ObjectDetectionProcessor(yolo_path)

# Function to scale depth map into a colorized format
def depth_to_colormap(depth_frame):
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    return depth_colormap

# Generator function for RGB frames
def generate_rgb_frames():
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        frame = np.asanyarray(color_frame.get_data())
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# Generator function for depth frames
def generate_depth_frames():
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue
        depth_colormap = depth_to_colormap(depth_frame)
        _, buffer = cv2.imencode('.jpg', depth_colormap)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# Generator function for YOLO-processed frames
def generate_yolo_frames():
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        frame = np.asanyarray(color_frame.get_data())
        processed_frame = processor.process_image(frame)  # Apply YOLO object detection
        _, buffer = cv2.imencode('.jpg', processed_frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# Routes for streaming
@app.route('/video_feed_rgb')
def video_feed_rgb():
    """Route to serve RGB video stream."""
    return Response(generate_rgb_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_depth')
def video_feed_depth():
    """Route to serve depth video stream."""
    return Response(generate_depth_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_yolo')
def video_feed_yolo():
    """Route to serve YOLO-processed video stream."""
    return Response(generate_yolo_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# HTML page to display all streams
@app.route('/')
def index():
    """Main page displaying the three video streams."""
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
    # Start the Flask server
    app.run(host="0.0.0.0", port=5000, debug=False)
