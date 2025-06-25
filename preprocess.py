import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import json
import logging
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    def __init__(self, config_path: str = "config.json"):
        """Initialize preprocessor with configuration"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.preprocessing_config = self.config['preprocessing']

    def base64_to_cv2(self, base64_str: str) -> Optional[np.ndarray]:
        """Convert base64 string to OpenCV image with enhanced preprocessing"""
        try:
            if ',' in base64_str:
                base64_str = base64_str.split(',')[-1]
            img_data = base64.b64decode(base64_str)
            img_array = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                logger.error("Failed to decode image")
                return None
            
            return self.enhance_image_quality(img)
        except Exception as e:
            logger.error(f"Error converting base64 to image: {e}")
            return None

    def _analyze_image_type(self, brightness: float, width: int, height: int) -> str:
        """Analyze image characteristics to determine likely image type"""
        aspect_ratio = width / height if height > 0 else 1
        pixel_count = width * height
        
        # ID document characteristics
        if (1.4 < aspect_ratio < 1.8 and  # ID card aspect ratio
            pixel_count > 300000 and      # Reasonable resolution
            80 < brightness < 180):       # Controlled lighting
            return "id_document"
        
        # Outdoor selfie characteristics
        elif (brightness > 150 or         # Bright lighting
              brightness < 40):           # Very dark/night
            return "outdoor_selfie"
        
        # Indoor selfie (default for portraits)
        elif (0.7 < aspect_ratio < 1.4):  # Portrait/square aspect ratio
            return "indoor_selfie"
        
        return "unknown"

    def enhance_image_quality(self, img: np.ndarray) -> np.ndarray:
        """Adaptive preprocessing based on image characteristics"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            height, width = img.shape[:2]
            image_type = self._analyze_image_type(brightness, width, height)
            
            # Convert to PIL for enhancement
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            # Adaptive enhancement based on image type
            if image_type == "outdoor_selfie":
                # Handle bright outdoor conditions
                if brightness > 150:
                    enhancer = ImageEnhance.Brightness(pil_img)
                    pil_img = enhancer.enhance(0.9)
                enhancer = ImageEnhance.Contrast(pil_img)
                pil_img = enhancer.enhance(1.3)
                
            elif image_type == "indoor_selfie":
                # Enhance indoor lighting
                if brightness < 100:
                    enhancer = ImageEnhance.Brightness(pil_img)
                    pil_img = enhancer.enhance(1.2)
                enhancer = ImageEnhance.Contrast(pil_img)
                pil_img = enhancer.enhance(1.15)
                
            elif image_type == "id_document":
                # Minimal processing for ID documents
                enhancer = ImageEnhance.Contrast(pil_img)
                pil_img = enhancer.enhance(1.1)
            
            # Universal sharpness enhancement
            enhancer = ImageEnhance.Sharpness(pil_img)
            pil_img = enhancer.enhance(1.2)
            
            # Convert back to OpenCV
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            # Adaptive CLAHE
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            clip_limit = 1.5 if image_type == "id_document" else 3.0
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            l = clahe.apply(l)
            img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
            
            # Noise reduction for compressed images
            if image_type in ["outdoor_selfie", "indoor_selfie"]:
                img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
            
            return img
            
        except Exception as e:
            logger.error(f"Error enhancing image: {e}")
            return img

    def validate_image(self, img: np.ndarray) -> Dict[str, Any]:
        """Enhanced validation for diverse image types"""
        if img is None:
            return {"valid": False, "reason": "Image is None"}

        height, width = img.shape[:2]
        
        # More lenient size requirements
        if height < 32 or width < 32:
            return {"valid": False, "reason": "Image too small"}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        # Adaptive brightness validation
        if mean_brightness < 10:  # Much more lenient
            return {"valid": False, "reason": "Image too dark"}
        if mean_brightness > 250:
            return {"valid": False, "reason": "Image too bright"}

        # Adaptive blur detection based on image characteristics
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Determine image type and adjust threshold
        image_type = self._analyze_image_type(mean_brightness, width, height)
        
        if image_type == "id_document":
            blur_threshold = 25  # Lower for ID documents
        elif image_type == "outdoor_selfie":
            blur_threshold = 15  # Much lower for outdoor photos
        else:
            blur_threshold = 20  # Lower for indoor selfies
        
        if blur_score < blur_threshold:
            return {
                "valid": False,
                "reason": "Image too blurry",
                "blur_score": blur_score,
                "threshold_used": blur_threshold,
                "image_type": image_type
            }

        return {
            "valid": True,
            "width": width,
            "height": height,
            "brightness": mean_brightness,
            "blur_score": blur_score,
            "image_type": image_type
        }

    def extract_face_region(self, img: np.ndarray, bbox, padding_ratio: float = None) -> np.ndarray:
        """Extract face region with intelligent padding"""
        try:
            if padding_ratio is None:
                padding_ratio = self.preprocessing_config['padding_ratio']
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Apply padding if enabled
            if self.preprocessing_config.get('padding', True):
                width = x2 - x1
                height = y2 - y1
                pad_x = int(width * padding_ratio)
                pad_y = int(height * padding_ratio)
                
                x1_pad = max(0, x1 - pad_x)
                y1_pad = max(0, y1 - pad_y)
                x2_pad = min(img.shape[1], x2 + pad_x)
                y2_pad = min(img.shape[0], y2 + pad_y)
                
                face_region = img[y1_pad:y2_pad, x1_pad:x2_pad]
            else:
                face_region = img[y1:y2, x1:x2]
            
            # Resize to target size
            target_size = tuple(self.preprocessing_config['target_size'])
            face_region = cv2.resize(
                face_region,
                target_size,
                interpolation=cv2.INTER_CUBIC
            )
            
            return face_region
            
        except Exception as e:
            logger.error(f"Error extracting face region: {e}")
            target_size = tuple(self.preprocessing_config['target_size'])
            return cv2.resize(img, target_size)

