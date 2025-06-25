import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import dlib
import insightface
from insightface.app import FaceAnalysis
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from scipy.spatial.distance import cosine
from typing import Dict, Optional, Tuple, List, Any
import json
import logging

logger = logging.getLogger(__name__)

class FaceMatcher:
    def __init__(self, config_path: str = "config.json"):
        """Initialize face matcher with configuration"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.thresholds = self.config['thresholds']
        self.model_config = self.config['models']
        self.ensemble_config = self.config['ensemble']
        self.models = {}
        self._init_models()

    def _init_models(self) -> None:
        """Initialize all face recognition models"""
        try:
            # Initialize InsightFace
            logger.info("Initializing InsightFace...")
            try:
                self.models['insightface'] = FaceAnalysis(
                    name=self.model_config['insightface']['model_name'],
                    providers=['CPUExecutionProvider'],
                    allowed_modules=['detection', 'recognition']
                )
                self.models['insightface'].prepare(
                    ctx_id=-1,
                    det_size=tuple(self.model_config['insightface']['det_size'])
                )
                logger.info("✅ InsightFace initialized successfully")
            except Exception as e:
                logger.error(f"❌ InsightFace initialization failed: {e}")

            # Initialize FaceNet
            logger.info("Initializing FaceNet...")
            try:
                self.models['facenet'] = InceptionResnetV1(
                    pretrained=self.model_config['facenet']['pretrained']
                ).eval()
                self.models['mtcnn'] = MTCNN(
                    image_size=160,
                    margin=20,
                    min_face_size=40,
                    thresholds=[0.6, 0.7, 0.9],
                    factor=0.709,
                    post_process=True,
                    device='cpu'
                )
                logger.info("✅ FaceNet initialized successfully")
            except Exception as e:
                logger.error(f"❌ FaceNet initialization failed: {e}")

            # Initialize Dlib
            logger.info("Initializing Dlib...")
            try:
                self.models['dlib_detector'] = dlib.get_frontal_face_detector()
                if os.path.exists('shape_predictor_68_face_landmarks.dat'):
                    self.models['dlib_sp'] = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
                if os.path.exists('dlib_face_recognition_resnet_model_v1.dat'):
                    self.models['dlib_model'] = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
                logger.info("✅ Dlib initialized successfully")
            except Exception as e:
                logger.error(f"❌ Dlib initialization failed: {e}")

        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise

    def get_face_embeddings(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract embeddings from all available models with enhanced error handling"""
        embeddings = {}

        # InsightFace embedding
        try:
            if 'insightface' in self.models:
                faces = self.models['insightface'].get(img)
                if faces and len(faces) > 0:
                    # Select the largest face
                    face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
                    embeddings['insightface'] = face.normed_embedding
                    logger.debug("✅ InsightFace embedding extracted")
        except Exception as e:
            logger.error(f"❌ InsightFace embedding error: {e}")

        # FaceNet embedding
        try:
            if 'facenet' in self.models and 'mtcnn' in self.models:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                face_tensor = self.models['mtcnn'](img_pil)
                
                if face_tensor is not None:
                    face_tensor = face_tensor.unsqueeze(0)
                    with torch.no_grad():
                        embedding = self.models['facenet'](face_tensor)
                    embeddings['facenet'] = F.normalize(embedding, p=2, dim=1).detach().numpy().flatten()
                    logger.debug("✅ FaceNet embedding extracted")
        except Exception as e:
            logger.error(f"❌ FaceNet embedding error: {e}")

        # Dlib embedding
        try:
            if all(k in self.models for k in ['dlib_detector', 'dlib_sp', 'dlib_model']):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                dets = self.models['dlib_detector'](gray, 1)
                
                if len(dets) > 0:
                    # Select the largest detected face
                    best_det = max(dets, key=lambda det: det.width() * det.height())
                    shape = self.models['dlib_sp'](gray, best_det)
                    descriptor = self.models['dlib_model'].compute_face_descriptor(
                        img, shape, num_jitters=self.model_config['dlib']['num_jitters']
                    )
                    embeddings['dlib'] = np.array(descriptor)
                    logger.debug("✅ Dlib embedding extracted")
        except Exception as e:
            logger.error(f"❌ Dlib embedding error: {e}")

        return embeddings

    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate enhanced cosine similarity between embeddings"""
        try:
            # Normalize embeddings
            emb1_norm = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
            emb2_norm = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
            
            # Calculate cosine similarity
            similarity = 1 - cosine(emb1_norm, emb2_norm)
            return np.clip(similarity, 0.0, 1.0)
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0

    def verify_faces(self, img1: np.ndarray, img2: np.ndarray, image_type: str) -> Dict[str, Any]:
        """Verify if two face images belong to the same person with enhanced ensemble"""
        try:
            # Get embeddings for both images
            embeddings1 = self.get_face_embeddings(img1)
            embeddings2 = self.get_face_embeddings(img2)

            # Calculate individual model scores
            model_results = {}
            for model_name in ['insightface', 'facenet', 'dlib']:
                if model_name in embeddings1 and model_name in embeddings2:
                    score = self.calculate_similarity(embeddings1[model_name], embeddings2[model_name])
                    threshold = self.thresholds[image_type][model_name]
                    model_results[model_name] = {
                        'score': float(score),
                        'threshold': threshold,
                        'match': score >= threshold,
                        'status': 'success'
                    }
                else:
                    model_results[model_name] = {
                        'score': None,
                        'threshold': self.thresholds[image_type][model_name],
                        'match': False,
                        'status': 'failed_detection'
                    }

            # Calculate ensemble result
            ensemble_result = self._ensemble_decision(model_results, image_type)

            return {
                'model_results': model_results,
                'ensemble': ensemble_result,
                'image_type': image_type
            }

        except Exception as e:
            logger.error(f"Error in face verification: {e}")
            return {
                'model_results': {},
                'ensemble': {
                    'match_score': 0.0,
                    'match': False,
                    'confidence_level': 'low',
                    'status': 'error',
                    'threshold': 0.35
                },
                'image_type': image_type
            }

    def _ensemble_decision(self, model_results: Dict, image_type: str) -> Dict[str, Any]:
        """Improved ensemble decision with adaptive thresholds"""
        try:
            valid_results = {k: v for k, v in model_results.items()
                            if v['status'] == 'success' and v['score'] is not None}
            
            if len(valid_results) < 1:  # Allow single model decisions
                return {
                    'match_score': 0.0,
                    'match': False,
                    'confidence_level': 'low',
                    'status': 'failed_detection',
                    'threshold': 0.35
                }
            
            # Weighted ensemble with adaptive weights
            weighted_scores = []
            total_weight = 0
            
            for model, result in valid_results.items():
                score = result['score']
                base_weight = self.model_config[model]['weight']
                
                # Boost weight for higher-performing models
                if score > 0.7:
                    weight = base_weight * 1.2
                elif score > 0.5:
                    weight = base_weight * 1.0
                else:
                    weight = base_weight * 0.8
                    
                weighted_scores.append(score * weight)
                total_weight += weight
            
            ensemble_score = sum(weighted_scores) / total_weight if total_weight > 0 else 0.0
            
            # Adaptive threshold based on image type and model count
            base_threshold = 0.35
            
            if image_type == "id_to_selfie":
                type_adjustment = 0.9  # More lenient for cross-domain
            else:
                type_adjustment = 1.0
                
            # Confidence boost for multiple models
            model_confidence = min(len(valid_results) / 3.0, 1.0)
            adjusted_threshold = base_threshold * type_adjustment * (1 - 0.1 * model_confidence)
            
            match = ensemble_score >= adjusted_threshold
            confidence_level = self._get_confidence_level(ensemble_score)
            
            return {
                'match_score': float(ensemble_score),
                'match': match,
                'confidence_level': confidence_level,
                'status': 'verified' if match else 'mismatch',
                'threshold': float(adjusted_threshold),
                'contributing_models': len(valid_results)
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble decision: {e}")
            return {
                'match_score': 0.0,
                'match': False,
                'confidence_level': 'low',
                'status': 'error',
                'threshold': 0.35
            }

    def _get_confidence_level(self, score: float) -> str:
        """Determine confidence level based on score"""
        confidence_levels = self.ensemble_config['confidence_levels']
        if confidence_levels['low'][0] <= score < confidence_levels['low'][1]:
            return 'low'
        elif confidence_levels['medium'][0] <= score < confidence_levels['medium'][1]:
            return 'medium'
        elif confidence_levels['high'][0] <= score <= confidence_levels['high'][1]:
            return 'high'
        else:
            return 'low'
