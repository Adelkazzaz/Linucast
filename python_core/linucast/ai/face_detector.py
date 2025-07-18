"""Face detection using MediaPipe."""

import logging
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import mediapipe as mp

from dataclasses import dataclass


@dataclass
class DetectedFace:
    """Represents a detected face."""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    landmarks: List[Tuple[float, float]]
    confidence: float
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get the center point of the face."""
        x, y, w, h = self.bbox
        return (x + w / 2, y + h / 2)


class FaceDetector:
    """Face detector using MediaPipe Face Detection and Face Mesh."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("face_detection", {})
        self.logger = logging.getLogger(__name__)
        
        # MediaPipe components
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize detectors
        self.face_detection = None
        self.face_mesh = None
        
        # Configuration
        self.confidence_threshold = self.config.get("confidence_threshold", 0.5)
        self.max_num_faces = self.config.get("max_faces", 5)
        
    def initialize(self) -> bool:
        """Initialize the face detector."""
        try:
            self.logger.info("Initializing MediaPipe face detector...")
            
            # Initialize face detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0,  # 0 for short-range (< 2 meters), 1 for full-range (> 2 meters)
                min_detection_confidence=self.confidence_threshold
            )
            
            # Initialize face mesh for landmarks
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=self.max_num_faces,
                refine_landmarks=True,
                min_detection_confidence=self.confidence_threshold,
                min_tracking_confidence=0.5
            )
            
            self.logger.info("Face detector initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing face detector: {e}")
            return False
    
    def detect_faces(self, image: np.ndarray) -> List[DetectedFace]:
        """Detect faces in the given image."""
        if self.face_detection is None or self.face_mesh is None:
            return []
        
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            
            # Detect faces
            detection_results = self.face_detection.process(rgb_image)
            mesh_results = self.face_mesh.process(rgb_image)
            
            faces = []
            
            if detection_results.detections:
                for i, detection in enumerate(detection_results.detections):
                    # Get bounding box
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * width)
                    y = int(bbox.ymin * height)
                    w = int(bbox.width * width)
                    h = int(bbox.height * height)
                    
                    # Ensure bbox is within image bounds
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, width - x)
                    h = min(h, height - y)
                    
                    # Get landmarks from face mesh if available
                    landmarks = []
                    if (mesh_results.multi_face_landmarks and 
                        i < len(mesh_results.multi_face_landmarks)):
                        
                        face_landmarks = mesh_results.multi_face_landmarks[i]
                        # Convert to pixel coordinates
                        landmarks = [
                            (landmark.x * width, landmark.y * height)
                            for landmark in face_landmarks.landmark
                        ]
                    
                    # Get confidence score
                    confidence = detection.score[0] if detection.score else 0.0
                    
                    face = DetectedFace(
                        bbox=(x, y, w, h),
                        landmarks=landmarks,
                        confidence=confidence
                    )
                    faces.append(face)
            
            return faces
            
        except Exception as e:
            self.logger.error(f"Error detecting faces: {e}")
            return []
    
    def draw_faces(self, image: np.ndarray, faces: List[DetectedFace]) -> np.ndarray:
        """Draw detected faces on the image."""
        annotated_image = image.copy()
        
        for face in faces:
            x, y, w, h = face.bbox
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw confidence
            confidence_text = f"{face.confidence:.2f}"
            cv2.putText(annotated_image, confidence_text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw some key landmarks if available
            if face.landmarks and len(face.landmarks) > 0:
                # Draw a few key points (nose tip, eye centers, mouth corners)
                key_indices = [1, 33, 263, 61, 291]  # MediaPipe landmark indices
                for idx in key_indices:
                    if idx < len(face.landmarks):
                        x_lm, y_lm = face.landmarks[idx]
                        cv2.circle(annotated_image, (int(x_lm), int(y_lm)), 2, (255, 0, 0), -1)
        
        return annotated_image
    
    def get_largest_face(self, faces: List[DetectedFace]) -> Optional[DetectedFace]:
        """Get the largest face from the detected faces."""
        if not faces:
            return None
        
        largest_face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])  # Area = width * height
        return largest_face
    
    def filter_faces_by_confidence(self, faces: List[DetectedFace], 
                                  min_confidence: float = None) -> List[DetectedFace]:
        """Filter faces by confidence threshold."""
        if min_confidence is None:
            min_confidence = self.confidence_threshold
        
        return [face for face in faces if face.confidence >= min_confidence]
    
    def shutdown(self) -> None:
        """Shutdown the face detector."""
        if self.face_detection:
            self.face_detection.close()
            self.face_detection = None
        
        if self.face_mesh:
            self.face_mesh.close()
            self.face_mesh = None
        
        self.logger.info("Face detector shutdown")


# Example usage
if __name__ == "__main__":
    # Test the face detector
    logging.basicConfig(level=logging.INFO)
    
    config = {
        "face_detection": {
            "confidence_threshold": 0.5,
            "max_faces": 5
        }
    }
    
    detector = FaceDetector(config)
    if detector.initialize():
        # Test with webcam
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            faces = detector.detect_faces(frame)
            annotated_frame = detector.draw_faces(frame, faces)
            
            cv2.imshow("Face Detection", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        detector.shutdown()
