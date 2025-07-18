"""Face identification and recognition using ArcFace."""

import logging
import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pickle
import os

from .face_detector import DetectedFace


@dataclass
class FaceIdentity:
    """Represents a face identity."""
    id: int
    name: str
    embedding: np.ndarray
    confidence: float = 0.0


class FaceIdentifier:
    """Face identifier using ArcFace model."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("face_recognition", {})
        self.logger = logging.getLogger(__name__)
        
        # ONNX Runtime session
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: str = ""
        self.output_name: str = ""
        
        # Model configuration
        self.model_path = self.config.get("model_path", "models/arcface.onnx")
        self.input_size = (112, 112)  # ArcFace standard input size
        
        # Face database
        self.known_faces: Dict[int, FaceIdentity] = {}
        self.next_face_id = 0
        self.similarity_threshold = 0.65
        
        # Face database file
        self.db_file = "face_database.pkl"
        
    def initialize(self) -> bool:
        """Initialize the face identifier."""
        try:
            self.logger.info("Initializing ArcFace face identifier...")
            
            # Check if model file exists
            if not os.path.exists(self.model_path):
                self.logger.warning(f"ArcFace model not found at {self.model_path}")
                self.logger.info("Face identification will be disabled")
                return True  # Not a critical error
            
            # Initialize ONNX Runtime session
            providers = ['CPUExecutionProvider']
            if ort.get_available_providers():
                # Use GPU if available
                if 'CUDAExecutionProvider' in ort.get_available_providers():
                    providers.insert(0, 'CUDAExecutionProvider')
                elif 'ROCMExecutionProvider' in ort.get_available_providers():
                    providers.insert(0, 'ROCMExecutionProvider')
            
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            
            # Get input and output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            # Load existing face database
            self.load_face_database()
            
            self.logger.info("Face identifier initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing face identifier: {e}")
            return False
    
    def preprocess_face(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Preprocess face image for ArcFace model."""
        x, y, w, h = bbox
        
        # Extract face region with some padding
        padding = int(max(w, h) * 0.1)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        face_image = image[y1:y2, x1:x2]
        
        if face_image.size == 0:
            return np.zeros((1, 3, self.input_size[0], self.input_size[1]), dtype=np.float32)
        
        # Resize to model input size
        face_resized = cv2.resize(face_image, self.input_size)
        
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [-1, 1]
        face_normalized = (face_rgb.astype(np.float32) - 127.5) / 127.5
        
        # Transpose to CHW format and add batch dimension
        face_input = np.transpose(face_normalized, (2, 0, 1))
        face_input = np.expand_dims(face_input, axis=0)
        
        return face_input
    
    def extract_embedding(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Extract face embedding using ArcFace model."""
        if self.session is None:
            return None
        
        try:
            # Preprocess face
            face_input = self.preprocess_face(image, bbox)
            
            # Run inference
            outputs = self.session.run([self.output_name], {self.input_name: face_input})
            embedding = outputs[0][0]  # Remove batch dimension
            
            # L2 normalize the embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error extracting face embedding: {e}")
            return None
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            # Cosine similarity
            similarity = np.dot(embedding1, embedding2)
            return float(similarity)
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def identify_face(self, image: np.ndarray, face: DetectedFace) -> Optional[FaceIdentity]:
        """Identify a face by comparing with known faces."""
        embedding = self.extract_embedding(image, face.bbox)
        if embedding is None:
            return None
        
        best_match = None
        best_similarity = 0.0
        
        for face_id, identity in self.known_faces.items():
            similarity = self.calculate_similarity(embedding, identity.embedding)
            if similarity > best_similarity and similarity > self.similarity_threshold:
                best_similarity = similarity
                best_match = FaceIdentity(
                    id=identity.id,
                    name=identity.name,
                    embedding=embedding,
                    confidence=similarity
                )
        
        return best_match
    
    def add_face(self, image: np.ndarray, face: DetectedFace, name: str = "") -> Optional[FaceIdentity]:
        """Add a new face to the database."""
        embedding = self.extract_embedding(image, face.bbox)
        if embedding is None:
            return None
        
        # Check if face already exists
        for identity in self.known_faces.values():
            similarity = self.calculate_similarity(embedding, identity.embedding)
            if similarity > self.similarity_threshold:
                self.logger.info(f"Face already exists with similarity {similarity:.3f}")
                return identity
        
        # Create new identity
        face_id = self.next_face_id
        self.next_face_id += 1
        
        if not name:
            name = f"Person_{face_id}"
        
        identity = FaceIdentity(
            id=face_id,
            name=name,
            embedding=embedding,
            confidence=1.0
        )
        
        self.known_faces[face_id] = identity
        self.save_face_database()
        
        self.logger.info(f"Added new face: {name} (ID: {face_id})")
        return identity
    
    def remove_face(self, face_id: int) -> bool:
        """Remove a face from the database."""
        if face_id in self.known_faces:
            name = self.known_faces[face_id].name
            del self.known_faces[face_id]
            self.save_face_database()
            self.logger.info(f"Removed face: {name} (ID: {face_id})")
            return True
        return False
    
    def get_all_faces(self) -> List[FaceIdentity]:
        """Get all known faces."""
        return list(self.known_faces.values())
    
    def save_face_database(self) -> None:
        """Save face database to file."""
        try:
            with open(self.db_file, 'wb') as f:
                pickle.dump({
                    'faces': self.known_faces,
                    'next_id': self.next_face_id
                }, f)
            self.logger.debug("Face database saved")
        except Exception as e:
            self.logger.error(f"Error saving face database: {e}")
    
    def load_face_database(self) -> None:
        """Load face database from file."""
        try:
            if os.path.exists(self.db_file):
                with open(self.db_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_faces = data.get('faces', {})
                    self.next_face_id = data.get('next_id', 0)
                self.logger.info(f"Loaded {len(self.known_faces)} known faces")
            else:
                self.logger.info("No existing face database found")
        except Exception as e:
            self.logger.error(f"Error loading face database: {e}")
            self.known_faces = {}
            self.next_face_id = 0
    
    def set_similarity_threshold(self, threshold: float) -> None:
        """Set the similarity threshold for face matching."""
        self.similarity_threshold = max(0.0, min(1.0, threshold))
        self.logger.info(f"Similarity threshold set to {self.similarity_threshold}")
    
    def shutdown(self) -> None:
        """Shutdown the face identifier."""
        self.save_face_database()
        self.session = None
        self.logger.info("Face identifier shutdown")


# Example usage
if __name__ == "__main__":
    # Test the face identifier
    logging.basicConfig(level=logging.INFO)
    
    config = {
        "face_recognition": {
            "model_path": "models/arcface.onnx"
        }
    }
    
    identifier = FaceIdentifier(config)
    if identifier.initialize():
        print("Face identifier initialized successfully")
        print(f"Known faces: {len(identifier.get_all_faces())}")
        identifier.shutdown()
