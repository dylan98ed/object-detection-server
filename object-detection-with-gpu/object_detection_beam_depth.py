import cv2
from ultralytics import YOLO
import numpy as np
import os

class ObjectDetectionProcessor:

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.engine_path = "yolo_3rt_model.engine"
        self.model = YOLO(self.engine_path)
        self.class_names = self.model.names


    def calculate_distance(self, depth_frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)

        depth_sensor = self.pipeline.get_active_profile() \
                                .get_device() \
                                .first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()  # en metros :contentReference[oaicite:0]{index=0}
    
        # escala a metros
        depth_m = raw * depth_scale

        # coordenadas vigentes
        h, w = depth_m.shape
        xs = np.clip(np.arange(cx-3, cx+4), 0, w-1)
        ys = np.clip(np.arange(cy-3, cy+4), 0, h-1)
        patch = depth_m[np.ix_(ys, xs)]
        valid = patch[np.isfinite(patch) & (patch > 0.01)]  # > 1 cm
        if valid.size == 0:
            return None
        return np.median(valid) * 100  # cm

    def process_image(self, cv_image, depth_frame):
        # Perform inference with YOLO
        results = self.model(cv_image)

        # Extract detections and draw bounding boxes
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()             # Bounding box coordinates, ".cpu()" asegura que el tensor esté en la CPU antes de convertirlo a NumPy.
            confidences = result.boxes.conf.cpu().numpy()       # Confidence scores
            labels = result.boxes.cls.cpu().numpy().astype(int) # Class labels

            valid_distance = True  # Default flag for distance validation

            for i, bbox in enumerate(boxes):
                if confidences[i] < 0.5:  # Ignore detections with low confidence
                    continue

                class_name = self.class_names[labels[i]]  # Get the detected class name
                #distance_cm = self.calculate_distance(bbox_width)  # Compute estimated distance
                distance_cm = self.calculate_distance(depth_frame, bbox)

                # Si no hay distancia válida, saltamos esta detección
                if distance_cm is None:
                    print(f"Distancia no valida para: {bbox}")
                    continue

                # Set bounding box color based on distance
                # Blue if the object is closer than 40 cm, red otherwise
                bbox_color = (255, 0, 0) if distance_cm < 40 else (0, 0, 255)  
                valid_distance = distance_cm >= 40  # Mark distance as valid if >= 40 cm

                # Draw bounding box on the image
                cv2.rectangle(cv_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox_color, 2)

                # Overlay text with class name, confidence score, and estimated distance
                cv2.putText(cv_image,
                            f"{class_name}: {round(confidences[i], 4)}, Dist: {round(distance_cm, 2)}cm",
                            (bbox[0], bbox[1] - 15),
                            cv2.FONT_HERSHEY_PLAIN,
                            1,
                            (255, 255, 255),  # White text
                            2)

        return cv_image  # Return the processed image with annotations