"""Background segmentation using MODNet and MediaPipe."""

import logging
import cv2
import numpy as np
import onnxruntime as ort
from typing import Dict, Any, Optional
import os

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None


class BackgroundSegmenter:
    """Background segmenter supporting multiple models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("background_segmentation", {})
        self.logger = logging.getLogger(__name__)
        
        # Model selection
        self.model_type = self.config.get("model", "modnet")
        self.model_path = self.config.get("model_path", "models/modnet.onnx")
        
        # ONNX Runtime session for MODNet
        self.onnx_session: Optional[ort.InferenceSession] = None
        
        # MediaPipe selfie segmentation
        self.mp_selfie_segmentation = None
        
        # Model configuration
        self.input_size = (512, 512)  # MODNet input size
        
    def initialize(self) -> bool:
        """Initialize the background segmenter."""
        try:
            self.logger.info(f"Initializing background segmenter with model: {self.model_type}")
            
            if self.model_type == "modnet":
                return self._initialize_modnet()
            elif self.model_type == "mediapipe":
                return self._initialize_mediapipe()
            else:
                self.logger.error(f"Unknown model type: {self.model_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error initializing background segmenter: {e}")
            return False
    
    def _initialize_modnet(self) -> bool:
        """Initialize MODNet ONNX model."""
        try:
            if not os.path.exists(self.model_path):
                self.logger.warning(f"MODNet model not found at {self.model_path}")
                # Fallback to MediaPipe if available
                if MEDIAPIPE_AVAILABLE:
                    self.logger.info("Falling back to MediaPipe selfie segmentation")
                    self.model_type = "mediapipe"
                    return self._initialize_mediapipe()
                else:
                    self.logger.error("No background segmentation models available")
                    return False
            
            # Initialize ONNX Runtime session
            providers = ['CPUExecutionProvider']
            if ort.get_available_providers():
                if 'CUDAExecutionProvider' in ort.get_available_providers():
                    providers.insert(0, 'CUDAExecutionProvider')
                elif 'ROCMExecutionProvider' in ort.get_available_providers():
                    providers.insert(0, 'ROCMExecutionProvider')
            
            self.onnx_session = ort.InferenceSession(self.model_path, providers=providers)
            
            self.logger.info("MODNet model initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing MODNet: {e}")
            return False
    
    def _initialize_mediapipe(self) -> bool:
        """Initialize MediaPipe selfie segmentation."""
        try:
            if not MEDIAPIPE_AVAILABLE:
                self.logger.error("MediaPipe not available")
                return False
            
            self.mp_selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(
                model_selection=1  # 0 for general, 1 for landscape
            )
            
            self.logger.info("MediaPipe selfie segmentation initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing MediaPipe: {e}")
            return False
    
    def segment_background(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Segment background and return mask."""
        if self.model_type == "modnet":
            return self._segment_modnet(image)
        elif self.model_type == "mediapipe":
            return self._segment_mediapipe(image)
        else:
            return None
    
    def _segment_modnet(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Segment using MODNet model."""
        if self.onnx_session is None:
            return None
        
        try:
            # Preprocess image
            input_tensor = self._preprocess_modnet(image)
            
            # Run inference
            input_name = self.onnx_session.get_inputs()[0].name
            output_name = self.onnx_session.get_outputs()[0].name
            
            outputs = self.onnx_session.run([output_name], {input_name: input_tensor})
            mask = outputs[0][0][0]  # Remove batch and channel dimensions
            
            # Resize mask to original image size
            mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
            
            # Convert to 0-255 range
            mask_uint8 = (mask_resized * 255).astype(np.uint8)
            
            return mask_uint8
            
        except Exception as e:
            self.logger.error(f"Error in MODNet segmentation: {e}")
            return None
    
    def _segment_mediapipe(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Segment using MediaPipe selfie segmentation."""
        if self.mp_selfie_segmentation is None:
            return None
        
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process image
            results = self.mp_selfie_segmentation.process(rgb_image)
            
            if results.segmentation_mask is not None:
                # Convert mask to uint8
                mask = (results.segmentation_mask * 255).astype(np.uint8)
                return mask
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error in MediaPipe segmentation: {e}")
            return None
    
    def _preprocess_modnet(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for MODNet model."""
        # Resize to model input size
        resized = cv2.resize(image, self.input_size)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Transpose to CHW format and add batch dimension
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor
    
    def apply_background_effect(self, image: np.ndarray, mask: np.ndarray, 
                               effect: str = "blur", background_image: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply background effect using the segmentation mask."""
        try:
            # Ensure mask is single channel
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
            # Ensure mask is the same size as image
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            
            # Normalize mask to [0, 1]
            mask_norm = mask.astype(np.float32) / 255.0
            
            # Apply effect
            if effect == "blur":
                return self._apply_blur_effect(image, mask_norm)
            elif effect == "replace" and background_image is not None:
                return self._apply_replace_effect(image, mask_norm, background_image)
            elif effect == "none":
                return image
            else:
                self.logger.warning(f"Unknown effect: {effect}")
                return image
                
        except Exception as e:
            self.logger.error(f"Error applying background effect: {e}")
            return image
    
    def _apply_blur_effect(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply blur effect to background."""
        # Create blurred version of the image
        blurred = cv2.GaussianBlur(image, (51, 51), 0)
        
        # Blend original and blurred based on mask
        # mask = 1 for foreground (person), 0 for background
        mask_3ch = np.stack([mask, mask, mask], axis=2)
        result = (image * mask_3ch + blurred * (1 - mask_3ch)).astype(np.uint8)
        
        return result
    
    def _apply_replace_effect(self, image: np.ndarray, mask: np.ndarray, 
                             background_image: np.ndarray) -> np.ndarray:
        """Replace background with another image."""
        # Resize background to match image size
        bg_resized = cv2.resize(background_image, (image.shape[1], image.shape[0]))
        
        # Blend original and background based on mask
        mask_3ch = np.stack([mask, mask, mask], axis=2)
        result = (image * mask_3ch + bg_resized * (1 - mask_3ch)).astype(np.uint8)
        
        return result
    
    def get_mask_quality_score(self, mask: np.ndarray) -> float:
        """Calculate a quality score for the segmentation mask."""
        try:
            # Simple quality metric based on edge smoothness and coverage
            edges = cv2.Canny(mask, 50, 150)
            edge_ratio = np.sum(edges > 0) / mask.size
            
            # Foreground coverage
            fg_ratio = np.sum(mask > 127) / mask.size
            
            # Combined score (lower edge ratio and reasonable fg coverage = better)
            score = max(0, 1 - edge_ratio * 2) * min(1, fg_ratio * 2)
            return float(score)
            
        except Exception as e:
            self.logger.error(f"Error calculating mask quality: {e}")
            return 0.0
    
    def shutdown(self) -> None:
        """Shutdown the background segmenter."""
        if self.onnx_session:
            self.onnx_session = None
        
        if self.mp_selfie_segmentation:
            self.mp_selfie_segmentation.close()
            self.mp_selfie_segmentation = None
        
        self.logger.info("Background segmenter shutdown")


# Example usage
if __name__ == "__main__":
    # Test the background segmenter
    logging.basicConfig(level=logging.INFO)
    
    config = {
        "background_segmentation": {
            "model": "mediapipe",  # Use MediaPipe for testing
            "model_path": "models/modnet.onnx"
        }
    }
    
    segmenter = BackgroundSegmenter(config)
    if segmenter.initialize():
        # Test with webcam
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            mask = segmenter.segment_background(frame)
            if mask is not None:
                # Apply blur effect
                result = segmenter.apply_background_effect(frame, mask, "blur")
                
                # Show results
                cv2.imshow("Original", frame)
                cv2.imshow("Mask", mask)
                cv2.imshow("Result", result)
            else:
                cv2.imshow("Original", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        segmenter.shutdown()
