import tensorflow as tf
import numpy as np
from PIL import Image
import io
import logging
import time
from typing import Dict, List, Optional
from pathlib import Path

from app.config import settings
from app.services.image_processor import ImageProcessor

logger = logging.getLogger(__name__)


class AcneClassifierService:
    """Service for acne classification using TensorFlow model"""

    def __init__(self):
        self.model: Optional[tf.keras.Model] = None
        self.image_processor = ImageProcessor()
        self.classes = settings.ACNE_CLASSES

    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load the TensorFlow model

        Args:
            model_path: Path to the model file (optional, uses config default)
        """
        try:
            path = model_path or settings.MODEL_PATH

            if not Path(path).exists():
                raise FileNotFoundError(f"Model file not found at {path}")

            logger.info(f"Loading model from {path}")
            self.model = tf.keras.models.load_model(path)
            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None

    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        if not self.is_model_loaded():
            raise RuntimeError("Model not loaded")

        return {
            "model_name": "Acne Classifier",
            "version": "1.0.0",
            "classes": self.classes,
            "input_size": list(settings.MODEL_INPUT_SIZE),
            "total_classes": len(self.classes)
        }

    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        """
        Preprocess image for model input

        Args:
            image_data: Raw image bytes

        Returns:
            Preprocessed image array
        """
        try:
            # Load image
            image = Image.open(io.BytesIO(image_data))

            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Resize to model input size
            image = image.resize(settings.MODEL_INPUT_SIZE)

            # Convert to array and normalize
            img_array = np.array(image, dtype=np.float32)

            # Normalize pixel values to [0, 1]
            img_array = img_array / 255.0

            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)

            return img_array

        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise ValueError(f"Failed to preprocess image: {str(e)}")

    def postprocess_predictions(
            self,
            predictions: np.ndarray,
            top_k: int = 3
    ) -> List[Dict]:
        """
        Post-process model predictions

        Args:
            predictions: Raw model predictions
            top_k: Number of top predictions to return

        Returns:
            List of prediction results
        """
        # Get top k predictions
        top_k = min(top_k, len(self.classes))
        top_indices = np.argsort(predictions[0])[::-1][:top_k]

        results = []
        for idx in top_indices:
            confidence = float(predictions[0][idx])

            # Only include predictions above threshold
            if confidence >= settings.CONFIDENCE_THRESHOLD or len(results) == 0:
                results.append({
                    "class_name": self.classes[idx],
                    "confidence": confidence,
                    "confidence_percentage": round(confidence * 100, 2)
                })

        return results

    def get_recommendations(self, class_name: str) -> List[str]:
        """
        Get treatment recommendations for a given acne class

        Args:
            class_name: Predicted acne class

        Returns:
            List of recommendations
        """
        return settings.RECOMMENDATIONS.get(class_name, [
            "Consult a dermatologist for personalized advice",
            "Maintain a consistent skincare routine",
            "Avoid touching or picking at affected areas"
        ])

    def classify(
            self,
            image_data: bytes,
            top_k: int = 3
    ) -> Dict:
        """
        Classify acne type from image

        Args:
            image_data: Raw image bytes
            top_k: Number of top predictions to return

        Returns:
            Classification results
        """
        if not self.is_model_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            start_time = time.time()

            # Preprocess image
            processed_image = self.preprocess_image(image_data)

            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)

            # Post-process results
            all_predictions = self.postprocess_predictions(predictions, top_k)

            # Get top prediction
            top_prediction = all_predictions[0]
            predicted_class = top_prediction["class_name"]
            confidence = top_prediction["confidence"]

            # Get recommendations
            recommendations = self.get_recommendations(predicted_class)

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to ms

            return {
                "success": True,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "all_predictions": all_predictions,
                "recommendations": recommendations,
                "processing_time_ms": round(processing_time, 2)
            }

        except Exception as e:
            logger.error(f"Classification error: {str(e)}")
            raise

    def classify_batch(
            self,
            images_data: List[bytes],
            top_k: int = 3
    ) -> List[Dict]:
        """
        Classify multiple images in batch

        Args:
            images_data: List of raw image bytes
            top_k: Number of top predictions per image

        Returns:
            List of classification results
        """
        results = []

        for image_data in images_data:
            try:
                result = self.classify(image_data, top_k)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch classification error: {str(e)}")
                results.append({
                    "success": False,
                    "error": str(e)
                })

        return results