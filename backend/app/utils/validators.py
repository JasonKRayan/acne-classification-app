from typing import Optional
from pathlib import Path
import magic
import logging

from app.config import settings

logger = logging.getLogger(__name__)


class FileValidator:
    """Validator for file uploads"""

    @staticmethod
    def validate_file_size(file_size: int) -> tuple[bool, Optional[str]]:
        """
        Validate file size

        Args:
            file_size: Size of file in bytes

        Returns:
            Tuple of (is_valid, error_message)
        """
        if file_size > settings.MAX_FILE_SIZE:
            max_mb = settings.MAX_FILE_SIZE / (1024 * 1024)
            return False, f"File size exceeds {max_mb}MB limit"

        if file_size == 0:
            return False, "File is empty"

        return True, None

    @staticmethod
    def validate_file_extension(filename: str) -> tuple[bool, Optional[str]]:
        """
        Validate file extension

        Args:
            filename: Name of the file

        Returns:
            Tuple of (is_valid, error_message)
        """
        file_ext = Path(filename).suffix.lower()

        if not file_ext:
            return False, "File has no extension"

        if file_ext not in settings.ALLOWED_EXTENSIONS:
            allowed = ", ".join(settings.ALLOWED_EXTENSIONS)
            return False, f"File extension not allowed. Allowed: {allowed}"

        return True, None

    @staticmethod
    def validate_mime_type(file_data: bytes) -> tuple[bool, Optional[str]]:
        """
        Validate file MIME type using python-magic

        Args:
            file_data: Raw file bytes

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            mime = magic.from_buffer(file_data, mime=True)

            allowed_mimes = ['image/jpeg', 'image/png', 'image/jpg']

            if mime not in allowed_mimes:
                return False, f"Invalid file type. Must be JPEG or PNG image"

            return True, None

        except Exception as e:
            logger.error(f"Error validating MIME type: {str(e)}")
            return False, "Could not validate file type"

    @staticmethod
    def validate_image_file(
            filename: str,
            file_data: bytes,
            check_mime: bool = True
    ) -> tuple[bool, Optional[str]]:
        """
        Comprehensive validation for image files

        Args:
            filename: Name of the file
            file_data: Raw file bytes
            check_mime: Whether to check MIME type (requires python-magic)

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Validate file size
        is_valid, error = FileValidator.validate_file_size(len(file_data))
        if not is_valid:
            return False, error

        # Validate file extension
        is_valid, error = FileValidator.validate_file_extension(filename)
        if not is_valid:
            return False, error

        # Validate MIME type if requested
        if check_mime:
            try:
                is_valid, error = FileValidator.validate_mime_type(file_data)
                if not is_valid:
                    return False, error
            except Exception as e:
                logger.warning(f"Could not check MIME type: {str(e)}")
                # Continue without MIME check if python-magic is not available

        return True, None


class InputValidator:
    """Validator for input parameters"""

    @staticmethod
    def validate_top_k(top_k: Optional[int]) -> tuple[bool, Optional[str]]:
        """
        Validate top_k parameter

        Args:
            top_k: Number of top predictions to return

        Returns:
            Tuple of (is_valid, error_message)
        """
        if top_k is None:
            return True, None

        if not isinstance(top_k, int):
            return False, "top_k must be an integer"

        if top_k < 1:
            return False, "top_k must be at least 1"

        if top_k > len(settings.ACNE_CLASSES):
            return False, f"top_k cannot exceed {len(settings.ACNE_CLASSES)}"

        return True, None

    @staticmethod
    def validate_confidence_threshold(threshold: Optional[float]) -> tuple[bool, Optional[str]]:
        """
        Validate confidence threshold parameter

        Args:
            threshold: Confidence threshold value

        Returns:
            Tuple of (is_valid, error_message)
        """
        if threshold is None:
            return True, None

        if not isinstance(threshold, (int, float)):
            return False, "Confidence threshold must be a number"

        if threshold < 0.0 or threshold > 1.0:
            return False, "Confidence threshold must be between 0.0 and 1.0"

        return True, None