import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Utility class for image processing operations"""

    @staticmethod
    def validate_image(image_data: bytes) -> bool:
        """
        Validate if the data is a valid image

        Args:
            image_data: Raw image bytes

        Returns:
            True if valid image, False otherwise
        """
        try:
            image = Image.open(io.BytesIO(image_data))
            image.verify()
            return True
        except Exception as e:
            logger.error(f"Image validation failed: {str(e)}")
            return False

    @staticmethod
    def get_image_dimensions(image_data: bytes) -> Tuple[int, int]:
        """
        Get image dimensions

        Args:
            image_data: Raw image bytes

        Returns:
            Tuple of (width, height)
        """
        try:
            image = Image.open(io.BytesIO(image_data))
            return image.size
        except Exception as e:
            logger.error(f"Error getting image dimensions: {str(e)}")
            raise

    @staticmethod
    def resize_image(
            image_data: bytes,
            target_size: Tuple[int, int],
            maintain_aspect_ratio: bool = False
    ) -> bytes:
        """
        Resize image to target size

        Args:
            image_data: Raw image bytes
            target_size: Target (width, height)
            maintain_aspect_ratio: Whether to maintain aspect ratio

        Returns:
            Resized image bytes
        """
        try:
            image = Image.open(io.BytesIO(image_data))

            if maintain_aspect_ratio:
                image.thumbnail(target_size, Image.Resampling.LANCZOS)
            else:
                image = image.resize(target_size, Image.Resampling.LANCZOS)

            # Convert back to bytes
            output = io.BytesIO()
            image.save(output, format='PNG')
            return output.getvalue()

        except Exception as e:
            logger.error(f"Error resizing image: {str(e)}")
            raise

    @staticmethod
    def enhance_image(
            image_data: bytes,
            brightness: float = 1.0,
            contrast: float = 1.0,
            sharpness: float = 1.0
    ) -> bytes:
        """
        Enhance image quality

        Args:
            image_data: Raw image bytes
            brightness: Brightness factor (1.0 = no change)
            contrast: Contrast factor (1.0 = no change)
            sharpness: Sharpness factor (1.0 = no change)

        Returns:
            Enhanced image bytes
        """
        try:
            image = Image.open(io.BytesIO(image_data))

            if brightness != 1.0:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(brightness)

            if contrast != 1.0:
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(contrast)

            if sharpness != 1.0:
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(sharpness)

            # Convert back to bytes
            output = io.BytesIO()
            image.save(output, format='PNG')
            return output.getvalue()

        except Exception as e:
            logger.error(f"Error enhancing image: {str(e)}")
            raise

    @staticmethod
    def normalize_image(image_array: np.ndarray) -> np.ndarray:
        """
        Normalize image array to [0, 1] range

        Args:
            image_array: Image as numpy array

        Returns:
            Normalized image array
        """
        return image_array.astype(np.float32) / 255.0

    @staticmethod
    def remove_noise(image_data: bytes) -> bytes:
        """
        Remove noise from image using Gaussian blur

        Args:
            image_data: Raw image bytes

        Returns:
            Denoised image bytes
        """
        try:
            # Convert to numpy array
            image = Image.open(io.BytesIO(image_data))
            img_array = np.array(image)

            # Apply Gaussian blur
            denoised = cv2.GaussianBlur(img_array, (5, 5), 0)

            # Convert back to PIL Image
            result_image = Image.fromarray(denoised)

            # Convert to bytes
            output = io.BytesIO()
            result_image.save(output, format='PNG')
            return output.getvalue()

        except Exception as e:
            logger.error(f"Error removing noise: {str(e)}")
            raise

    @staticmethod
    def crop_center(
            image_data: bytes,
            crop_size: Tuple[int, int]
    ) -> bytes:
        """
        Crop image from center

        Args:
            image_data: Raw image bytes
            crop_size: Size of crop (width, height)

        Returns:
            Cropped image bytes
        """
        try:
            image = Image.open(io.BytesIO(image_data))
            width, height = image.size
            crop_width, crop_height = crop_size

            left = (width - crop_width) // 2
            top = (height - crop_height) // 2
            right = left + crop_width
            bottom = top + crop_height

            cropped = image.crop((left, top, right, bottom))

            # Convert to bytes
            output = io.BytesIO()
            cropped.save(output, format='PNG')
            return output.getvalue()

        except Exception as e:
            logger.error(f"Error cropping image: {str(e)}")
            raise

    @staticmethod
    def convert_to_rgb(image_data: bytes) -> bytes:
        """
        Convert image to RGB format

        Args:
            image_data: Raw image bytes

        Returns:
            RGB image bytes
        """
        try:
            image = Image.open(io.BytesIO(image_data))

            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Convert to bytes
            output = io.BytesIO()
            image.save(output, format='PNG')
            return output.getvalue()

        except Exception as e:
            logger.error(f"Error converting to RGB: {str(e)}")
            raise