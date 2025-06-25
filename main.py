from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, validator
import uvicorn
import logging
from typing import Optional
import json
from face_matcher import FaceMatcher
from preprocess import ImagePreprocessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Face Verification Service",
    description="Production-ready microservice for facial verification with adaptive image processing",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize components
face_matcher = None
preprocessor = None

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global face_matcher, preprocessor
    try:
        logger.info("Initializing Face Verification Service...")
        preprocessor = ImagePreprocessor()
        face_matcher = FaceMatcher()
        logger.info("✅ Face Verification Service initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize service: {e}")
        raise

# Pydantic models
class VerificationRequest(BaseModel):
    user_id: str
    image_type: str
    image_1_base64: str
    image_2_base64: str

    @validator('image_type')
    def validate_image_type(cls, v):
        if v not in ['id_to_selfie', 'selfie_to_selfie']:
            raise ValueError('image_type must be either "id_to_selfie" or "selfie_to_selfie"')
        return v

    @validator('user_id')
    def validate_user_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('user_id cannot be empty')
        return v.strip()

# Updated response model - removed extra validation fields
class VerificationResponse(BaseModel):
    user_id: str
    match_score: float
    match: bool
    confidence_level: str
    image_type: str
    threshold: float
    status: str

class HealthResponse(BaseModel):
    status: str
    adaptive_processing: bool = True

class VersionResponse(BaseModel):
    model_version: str
    adaptive_features: bool = True

# Dependency to get face matcher
def get_face_matcher() -> FaceMatcher:
    if face_matcher is None:
        raise HTTPException(status_code=503, detail="Face matcher not initialized")
    return face_matcher

def get_preprocessor() -> ImagePreprocessor:
    if preprocessor is None:
        raise HTTPException(status_code=503, detail="Preprocessor not initialized")
    return preprocessor

# API Endpoints
@app.post("/verify-face", response_model=VerificationResponse)
async def verify_face(
    request: VerificationRequest,
    matcher: FaceMatcher = Depends(get_face_matcher),
    prep: ImagePreprocessor = Depends(get_preprocessor)
):
    """
    Verify if two facial images belong to the same person with adaptive processing
    Supports outdoor selfies, indoor selfies, and ID document images
    """
    try:
        logger.info(f"Processing verification request for user: {request.user_id}")

        # Convert base64 images to OpenCV format
        img1 = prep.base64_to_cv2(request.image_1_base64)
        img2 = prep.base64_to_cv2(request.image_2_base64)

        if img1 is None:
            raise HTTPException(
                status_code=400,
                detail="Failed to decode image_1_base64. Please check the base64 format."
            )

        if img2 is None:
            raise HTTPException(
                status_code=400,
                detail="Failed to decode image_2_base64. Please check the base64 format."
            )

        # Validate image quality (keep validation logic but don't include in response)
        img1_validation = prep.validate_image(img1)
        if not img1_validation['valid']:
            raise HTTPException(
                status_code=400,
                detail=f"Image 1 validation failed: {img1_validation['reason']}. "
                       f"Blur score: {img1_validation.get('blur_score', 'N/A')}, "
                       f"Threshold used: {img1_validation.get('threshold_used', 'N/A')}, "
                       f"Image type: {img1_validation.get('image_characteristics', 'unknown')}"
            )

        img2_validation = prep.validate_image(img2)
        if not img2_validation['valid']:
            raise HTTPException(
                status_code=400,
                detail=f"Image 2 validation failed: {img2_validation['reason']}. "
                       f"Blur score: {img2_validation.get('blur_score', 'N/A')}, "
                       f"Threshold used: {img2_validation.get('threshold_used', 'N/A')}, "
                       f"Image type: {img2_validation.get('image_characteristics', 'unknown')}"
            )

        # Perform face verification (same logic)
        result = matcher.verify_faces(img1, img2, request.image_type)
        ensemble_result = result['ensemble']

        # Check if face detection failed
        if ensemble_result['status'] == 'failed_detection':
            raise HTTPException(
                status_code=422,
                detail="No face detected in one or both images"
            )

        # Prepare clean response (removed validation details)
        response = VerificationResponse(
            user_id=request.user_id,
            match_score=ensemble_result['match_score'],
            match=ensemble_result['match'],
            confidence_level=ensemble_result['confidence_level'],
            image_type=request.image_type,
            threshold=ensemble_result['threshold'],
            status=ensemble_result['status']
        )

        logger.info(f"Verification completed for user {request.user_id}: {response.status}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing verification request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during face verification")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint with adaptive processing status
    """
    return HealthResponse(status="ok", adaptive_processing=True)

@app.get("/version", response_model=VersionResponse)
async def get_version():
    """
    Get model version information with adaptive features
    """
    return VersionResponse(model_version="2.0.0", adaptive_features=True)

@app.post("/test-validation")
async def test_image_validation(
    image_base64: str,
    prep: ImagePreprocessor = Depends(get_preprocessor)
):
    """
    Test endpoint for image validation with detailed diagnostics
    """
    try:
        img = prep.base64_to_cv2(image_base64)
        if img is None:
            return {"error": "Failed to decode image"}
        
        validation_result = prep.validate_image(img)
        return {
            "validation_result": validation_result,
            "adaptive_processing": True
        }
    except Exception as e:
        return {"error": str(e)}

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found"}

@app.exception_handler(422)
async def validation_exception_handler(request, exc):
    return {"error": "Validation error", "details": str(exc)}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
