from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime


class PredictionResult(BaseModel):
    """Single prediction result"""
    class_name: str = Field(..., description="Predicted acne class")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    confidence_percentage: float = Field(..., ge=0.0, le=100.0, description="Confidence as percentage")


class ClassificationResponse(BaseModel):
    """Response model for classification endpoint"""
    success: bool = Field(default=True, description="Whether classification was successful")
    predicted_class: str = Field(..., description="Top predicted acne class")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for top prediction")
    all_predictions: List[PredictionResult] = Field(..., description="All top-k predictions")
    recommendations: Optional[List[str]] = Field(None, description="Treatment recommendations")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Classification timestamp")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "predicted_class": "Pustules",
                "confidence": 0.87,
                "all_predictions": [
                    {
                        "class_name": "Pustules",
                        "confidence": 0.87,
                        "confidence_percentage": 87.0
                    },
                    {
                        "class_name": "Papules",
                        "confidence": 0.08,
                        "confidence_percentage": 8.0
                    },
                    {
                        "class_name": "Nodules",
                        "confidence": 0.03,
                        "confidence_percentage": 3.0
                    }
                ],
                "recommendations": [
                    "Use benzoyl peroxide or salicylic acid",
                    "Keep area clean and dry",
                    "Avoid touching or popping"
                ],
                "timestamp": "2024-01-15T10:30:00",
                "processing_time_ms": 145.5
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service health status")
    model_loaded: bool = Field(..., description="Whether ML model is loaded")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "timestamp": "2024-01-15T10:30:00"
            }
        }


class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_name: str = Field(..., description="Name of the model")
    version: str = Field(..., description="Model version")
    classes: List[str] = Field(..., description="List of acne classes")
    input_size: tuple = Field(..., description="Expected input image size")
    total_classes: int = Field(..., description="Total number of classes")

    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "Acne Classifier",
                "version": "1.0.0",
                "classes": ["Blackheads", "Dark Spot", "Nodules", "Papules", "Pustules", "Whiteheads"],
                "input_size": [224, 224],
                "total_classes": 6
            }
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "error": "Classification failed",
                "detail": "Invalid image format",
                "timestamp": "2024-01-15T10:30:00"
            }
        }


class BatchClassificationRequest(BaseModel):
    """Request model for batch classification"""
    image_urls: List[str] = Field(..., min_length=1, max_length=10, description="List of image URLs")
    top_k: Optional[int] = Field(3, ge=1, le=10, description="Number of top predictions")


class BatchClassificationResponse(BaseModel):
    """Response model for batch classification"""
    success: bool = Field(default=True)
    results: List[ClassificationResponse] = Field(..., description="Classification results for each image")
    total_processed: int = Field(..., description="Total number of images processed")
    failed_count: int = Field(0, description="Number of failed classifications")
    timestamp: datetime = Field(default_factory=datetime.utcnow)