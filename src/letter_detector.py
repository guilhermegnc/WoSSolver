# -*- coding: utf-8 -*-
import logging
import json
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple

# Imports for CNN
try:
    import onnxruntime as ort
    CNN_AVAILABLE = True
except Exception:
    CNN_AVAILABLE = False

logger = logging.getLogger(__name__)

class LetterDetectorCNN:
    """Wrapper to load and use the ONNX letter detection model."""
    
    def __init__(self, model_path=Path(__file__).parent / '../model/letter_detector_model.onnx', label_map_path=Path(__file__).parent / '../model/label_map.json'):
        self.session = None
        self.label_map = {}
        self.img_size = 64
        self.input_name: Optional[str] = None
        
        if not CNN_AVAILABLE:
            logger.error("ONNX Runtime is not available. Please install it.")
            return

        try:
            # Load the ONNX model
            if Path(model_path).exists():
                self.session = ort.InferenceSession(str(model_path))
                # Get the input name from the model
                self.input_name = self.session.get_inputs()[0].name
                logger.info(f"ONNX model loaded: {model_path}")
                logger.info(f"Model input name: {self.input_name}")
            else:
                logger.warning(f"Model not found at: {model_path}")
                
            # Load label map
            if Path(label_map_path).exists():
                with open(str(label_map_path), 'r') as f:
                    self.label_map = json.load(f)
                logger.info(f"Label map loaded: {len(self.label_map)} classes")
            else:
                logger.warning(f"Label map not found at: {label_map_path}")
        except Exception as e:
            logger.error(f"Error loading ONNX model: {e}")
    
    def is_ready(self) -> bool:
        """Checks if the model has been loaded correctly."""
        return self.session is not None and self.input_name is not None and len(self.label_map) > 0
    
    def predict_letter(self, tile_image: np.ndarray, confidence_threshold: float = 0.3) -> Optional[Tuple[str, float]]:
        """Predicts the letter of an individual tile.
        
        Args:
            tile_image: Tile image (BGR or grayscale)
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Tuple (letter, confidence) or None if below threshold
        """
        if not self.is_ready():
            return None
        
        try:
            # Convert to grayscale if necessary
            if len(tile_image.shape) == 3:
                gray = cv2.cvtColor(tile_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = tile_image
            
            # Resize to model's expected size
            resized = cv2.resize(gray, (self.img_size, self.img_size))
            
            # Normalize (0-1)
            normalized = resized.astype('float32') / 255.0
            
            # Add batch and channel dimensions
            input_data = np.expand_dims(normalized, axis=(0, -1))  # (1, 64, 64, 1)
            
            # Prediction with ONNX Runtime
            # The input must be a dictionary where keys are input names
            result = self.session.run(None, {self.input_name: input_data})
            
            # The output is a list of numpy arrays
            prediction = result[0][0]
            confidence = float(np.max(prediction))
            class_idx = int(np.argmax(prediction))
            
            # Check confidence threshold
            if confidence < confidence_threshold:
                logger.debug(f"Prediction below threshold: {confidence:.2%}")
                return None
            
            # Get corresponding letter
            reverse_map = {v: k for k, v in self.label_map.items()}
            if class_idx in reverse_map:
                letter = reverse_map[class_idx]
                logger.debug(f"Predicted letter: {letter} ({confidence:.2%})")
                return (letter, confidence)
            else:
                logger.warning(f"Class {class_idx} not found in label map")
                return None
        except Exception as e:
            logger.error(f"Error in ONNX prediction: {e}")
            return None