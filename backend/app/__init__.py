"""
Acne Classification API

A FastAPI-based backend for classifying acne types from images.
"""

__version__ = "1.0.0"
__author__ = "Juan"

from app.main import app

__all__ = ["app"]