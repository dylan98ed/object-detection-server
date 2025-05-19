import cv2
import numpy as np
import pyrealsense2 as rs
from flask import Flask, Response, jsonify
from ultralytics import YOLO
import os
import threading
import time

app = Flask(__name__)

# =============================================
# RealSense Camera Configuration
# =============================================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# =============================================
# YOLO Model Initialization
# =============================================
YOLO_WEIGHTS_PATH = "train_model/train2/weights/best.pt"
YOLO_EXPORT_PATH = "train_model/train2/weights/best_ncnn_model"

# Export YOLO model to NCNN format if not exists
if not os.path.exists(YOLO_EXPORT_PATH):
    model = YOLO(YOLO_WEIGHTS_PATH)
    model.export(format="ncnn")

# =============================================
# Object Detection Processor Class
# =============================================
class ObjectDetectionProcessor:
    """Handles object detection and distance calculation using YOLO and RealSense depth data"""
    
    def __init__(self, yolo_path, depth_scale):
        """
        Initialize object detector with YOLO model and depth sensor parameters
        
        Args:
            yolo_path (str): Path to YOLO model weights
            depth_scale (float): Depth sensor scale factor from RealSense
        """
        self.model_path = yolo_path
        self.model = YOLO(self.model_path)
        self.depth_scale = depth_scale
        self.class_names = self.model.names
        self.detections = []
        self.last_update_time = time.time()
        self.lock = threading.Lock()

        # Geometric calibration parameters
        self.real_width = 0.06  # Known object width in meters (stop sign)
        self.focal_length = 480  # Camera focal length in pixels

    def calculate_geometric_distance(self, bbox_width):
        """Calculate distance using object width and perspective projection
        
        Args:
            bbox_width (float): Detected object width in pixels
            
        Returns:
            float: Calculated distance in centimeters
        """
        distance_m = (self.real_width * self.focal_length) / bbox_width
        return distance_m * 100  # Convert to centimeters

    def process_image(self, color_image, depth_image):
        """Process frame for object detection and annotation
        
        Args:
            color_image (np.array): RGB input frame
            depth_image (np.array): Depth frame data
            
        Returns:
            np.array: Annotated output frame
        """
        new_detections = []
        results = self.model(color_image)

        for result in results:
            boxes = result.boxes.xyxy.numpy()
            confidences = result.boxes.conf.numpy()
            labels = result.boxes.cls.numpy().astype(int)

            for i, bbox in enumerate(boxes):
                if confidences[i] < 0.5:
                    continue  # Skip low-confidence detections

                # Extract detection metadata
                class_name = self.class_names[labels[i]]
                bbox = bbox.astype(int)
                x_min, y_min, x_max, y_max = bbox
                bbox_width = x_max - x_min

                # Calculate geometric distance
                geom_distance = self.calculate_geometric_distance(bbox_width)

                # Calculate depth-based distance with outlier rejection
                depth_distance = self.calculate_robust_depth_distance(
                    depth_image, x_min, y_min, x_max, y_max
                )

                # Annotate frame
                self._draw_annotations(
                    color_image, bbox, class_name, confidences[i], geom_distance
                )

                new_detections.append({
                    'class_name': class_name,
                    'confidence': float(confidences[i]),
                    'distance_calculated': geom_distance,
                    'distance_mapped': depth_distance
                })

        # Update shared detection data
        with self.lock:
            self.detections = new_detections
            self.last_update_time = time.time()

        return color_image

    def calculate_robust_depth_distance(self, depth_image, x_min, y_min, x_max, y_max):
        """Calculate depth distance with outlier rejection
        
        Args:
            depth_image (np.array): Depth frame data
            x_min, y_min, x_max, y_max (int): Bounding box coordinates
            
        Returns:
            float: Filtered depth distance in centimeters
        """
        # Create sampling grid within bounding box
        y_coords, x_coords = np.mgrid[y_min:y_max:10j, x_min:x_max:10j]
        x_coords = x_coords.astype(int).clip(0, depth_image.shape[1]-1)
        y_coords = y_coords.astype(int).clip(0, depth_image.shape[0]-1)

        # Convert depth values to centimeters
        depth_values = depth_image[y_coords, x_coords] * self.depth_scale * 100

        # Remove outliers using IQR method
        q25, q75 = np.percentile(depth_values, [25, 75])
        iqr = q75 - q25
        valid_mask = (depth_values >= q25 - 1.5*iqr) & (depth_values <= q75 + 1.5*iqr)
        filtered = depth_values[valid_mask]

        return np.median(filtered) if filtered.size > 0 else 0.0

    def _draw_annotations(self, image, bbox, class_name, confidence, distance):
        """Draw bounding box and text annotations
        
        Args:
            image (np.array): Frame to draw on
            bbox (tuple): Bounding box coordinates
            class_name (str): Detected class name
            confidence (float): Detection confidence
            distance (float): Calculated distance
        """
        color = (255, 0, 0) if distance < 40 else (0, 0, 255)
        x_min, y_min, x_max, y_max = bbox
        
        # Draw bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Draw text label
        label = f"{class_name}: {confidence:.2f}, Dist: {distance:.2f}cm"
        cv2.putText(image, label, (x_min, y_min - 15),
                   cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

# =============================================
# Depth Sensor Initialization
# =============================================
depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
DEPTH_SCALE = depth_sensor.get_depth_scale()
processor = ObjectDetectionProcessor(YOLO_EXPORT_PATH, DEPTH_SCALE)

# =============================================
# Frame Generation Functions
# =============================================
def depth_to_colormap(depth_frame):
    """Convert depth frame to color-mapped image
    
    Args:
        depth_frame (rs.frame): Input depth frame
        
    Returns:
        np.array: Color-mapped depth image
    """
    depth_image = np.asanyarray(depth_frame.get_data())
    return cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

def generate_rgb_frames():
    """Generator for RGB video stream"""
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if color_frame:
            frame = np.asanyarray(color_frame.get_data())
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def generate_depth_frames():
    """Generator for depth video stream"""
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if depth_frame:
            depth_colormap = depth_to_colormap(depth_frame)
            _, buffer = cv2.imencode('.jpg', depth_colormap)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def generate_yolo_frames():
    """Generator for processed YOLO video stream"""
    align = rs.align(rs.stream.color)
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if color_frame and depth_frame:
            color_img = np.asanyarray(color_frame.get_data())
            depth_img = np.asanyarray(depth_frame.get_data())
            
            processed_frame = processor.process_image(color_img, depth_img)
            _, buffer = cv2.imencode('.jpg', processed_frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# =============================================
# Flask Routes
# =============================================
@app.route('/video_feed_rgb')
def video_feed_rgb():
    """Endpoint for RGB video feed"""
    return Response(generate_rgb_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_depth')
def video_feed_depth():
    """Endpoint for depth video feed"""
    return Response(generate_depth_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_yolo')
def video_feed_yolo():
    """Endpoint for processed YOLO video feed"""
    return Response(generate_yolo_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections')
def get_detections():
    """Endpoint for detection data"""
    with processor.lock:
        return jsonify({
            'detections': processor.detections,
            'timestamp': processor.last_update_time
        })

@app.route('/')
def index():
    """Main page serving HTML interface"""
    return '''
    <html>
        <head>
            <title>Intel RealSense Streaming</title>
            <style>
                body { 
                    display: flex; 
                    flex-direction: column; 
                    justify-content: center; 
                    align-items: center; 
                    background-color: black; 
                    color: white; 
                }
                .container { 
                    display: flex; 
                    flex-direction: row; 
                }
                .stream { 
                    margin: 10px; 
                    text-align: center; 
                }
                #detections-table {
                    margin-top: 20px;
                    border-collapse: collapse;
                    table-layout: fixed;
                    width: auto;
                }
                #detections-table th, #detections-table td {
                    border: 1px solid white;
                    padding: 8px;
                    text-align: left;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                }
                #detections-table th:nth-child(1),
                #detections-table td:nth-child(1) {
                    width: 150px;
                }
                #detections-table th:nth-child(2),
                #detections-table td:nth-child(2),
                #detections-table th:nth-child(3),
                #detections-table td:nth-child(3) {
                    width: 100px;
                }
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
            <div id="detections-container">
                <h2>Detected Objects</h2>
                <table id="detections-table">
                    <thead>
                        <tr>
                            <th>Object</th>
                            <th>Confidence</th>
                            <th>Calculated (cm)</th>
                            <th>Mapped (cm)</th>  <!-- Nueva columna -->
                        </tr>
                    </thead>
                    <tbody id="detections-body">
                    </tbody>
                </table>
            </div>
            <script>
                let lastTimestamp = 0;

                function updateDetections() {
                    fetch('/detections')
                        .then(response => response.json())
                        .then(data => {
                            if (data.timestamp > lastTimestamp) {
                                lastTimestamp = data.timestamp;
                                const tbody = document.getElementById('detections-body');
                                tbody.innerHTML = '';
                                data.detections.forEach(obj => {
                                    const row = document.createElement('tr');
                                    row.innerHTML = `
                                        <td>${obj.class_name}</td>
                                        <td>${obj.confidence.toFixed(2)}</td>
                                        <td>${obj.distance_calculated.toFixed(2)}</td> 
                                        <td>${obj.distance_mapped.toFixed(2)}</td>
                                    `;
                                    tbody.appendChild(row);
                                });
                            }
                        });
                }
                setInterval(updateDetections, 100);  // Intervalo más corto
            </script>
        </body>
    </html>
    '''

# =============================================
# Main Execution
# =============================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)