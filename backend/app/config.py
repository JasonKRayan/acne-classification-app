import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # API Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    # CORS Settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000"
    ]

    # File Upload Settings
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png"]

    # Model Settings
    MODEL_PATH: str = os.getenv("MODEL_PATH", "ml/models/acne_classifier.h5")
    MODEL_INPUT_SIZE: tuple = (224, 224)

    # Classification Settings
    CONFIDENCE_THRESHOLD: float = 0.3
    DEFAULT_TOP_K: int = 3

    # Acne Classes (update based on your model)
    ACNE_CLASSES: List[str] = [
        "Blackheads",
        "Dark Spot",
        "Nodules",
        "Papules",
        "Pustules",
        "Whiteheads"
    ]

    # Recommendations (optional - can be enhanced)
    RECOMMENDATIONS: dict = {
        "Blackheads": [
            "Use salicylic acid cleanser",
            "Consider regular exfoliation",
            "Avoid comedogenic products"
        ],
        "Dark Spot": [
            "Use vitamin C serum",
            "Apply sunscreen daily",
            "Consider niacinamide products"
        ],
        "Nodules": [
            "Consult a dermatologist",
            "Avoid picking or squeezing",
            "May require prescription treatment"
        ],
        "Papules": [
            "Use benzoyl peroxide products",
            "Maintain gentle skincare routine",
            "Avoid harsh scrubs"
        ],
        "Pustules": [
            "Use benzoyl peroxide or salicylic acid",
            "Keep area clean and dry",
            "Avoid touching or popping"
        ],
        "Whiteheads": [
            "Use retinoid products",
            "Regular gentle cleansing",
            "Consider non-comedogenic moisturizer"
        ]
    }

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()