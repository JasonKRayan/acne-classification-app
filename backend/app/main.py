from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import Optional
import logging

from app.models.schemas import (
    ClassificationResponse,
    HealthResponse,
    ModelInfoResponse,
    ErrorResponse
)
from app.services.classifier import AcneClassifierService
from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Acne Classification API",
    description="API for classifying acne types from images",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize classifier service
classifier_service = AcneClassifierService()


@app.on_event("startup")
async def startup_event():
    """Load ML model on startup"""
    try:
        logger.info("Loading ML model...")
        classifier_service.load_model()
        logger.info("ML model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Acne Classification API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    model_loaded = classifier_service.is_model_loaded()
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded
    )


@app.get("/api/v1/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get model information"""
    try:
        info = classifier_service.get_model_info()
        return ModelInfoResponse(**info)
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model info")


@app.post("/api/v1/classify", response_model=ClassificationResponse)
async def classify_acne(
        file: UploadFile = File(...),
        top_k: Optional[int] = 3
):
    """
    Classify acne type from an uploaded image

    Args:
        file: Image file (JPEG, PNG)
        top_k: Number of top predictions to return (default: 3)

    Returns:
        Classification results with confidence scores
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="File must be an image (JPEG, PNG, etc.)"
            )

        # Read image data
        image_data = await file.read()

        # Validate file size (10MB limit)
        if len(image_data) > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds {settings.MAX_FILE_SIZE / (1024 * 1024)}MB limit"
            )

        # Perform classification
        result = classifier_service.classify(image_data, top_k=top_k)

        return ClassificationResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Classification failed: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )