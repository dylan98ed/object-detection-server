import cv2
import torch
from ultralytics import YOLO

class ObjectDetectionProcessor:

    def __init__(self, yolo_path):
        # Load the YOLO model
        self.model_path = yolo_path
        self.model = YOLO(self.model_path)
        self.class_names = self.model.names  # Get class names

        # Average real-world dimensions of a stop sign (in meters)
        self.real_width = 0.06  # Stop sign: 6 cm actual width
        self.focal_length = 480  # Webcam focal length in pixels

    def calculate_distance(self, bbox_width):
        """ Calculates the distance to the stop sign based on bounding box width. """
        distance_m = (self.real_width * self.focal_length) / bbox_width
        return distance_m * 100  # Convert to centimeters

    def process_image(self, cv_image):
        """
        Processes the input image to detect stop signs and display their distance.
        """
        # Perform inference with YOLO
        results = self.model(cv_image)

        # Extract detections and draw bounding boxes
        for result in results:
            boxes = result.boxes.xyxy.numpy()  # Bounding box coordinates
            confidences = result.boxes.conf.numpy()  # Confidence scores
            labels = result.boxes.cls.numpy().astype(int)  # Class labels

            valid_distance = True  # Default flag for distance validation

            for i, bbox in enumerate(boxes):
                if confidences[i] < 0.5:  # Ignore detections with low confidence
                    continue

                class_name = self.class_names[labels[i]]  # Get the detected class name

                bbox = bbox.astype(int)  # Convert bbox coordinates to integers
                bbox_width = bbox[2] - bbox[0]  # Calculate bounding box width
                distance_cm = self.calculate_distance(bbox_width)  # Compute estimated distance

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
