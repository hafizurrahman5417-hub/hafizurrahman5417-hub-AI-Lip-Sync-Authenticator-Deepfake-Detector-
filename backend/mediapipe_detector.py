import cv2
import mediapipe as mp
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger("MediaPipeDetector")

# Define standard MediaPipe Face Mesh landmark indices for lips
LIP_OUTER_TOP = 13
LIP_OUTER_BOTTOM = 14
LIP_OUTER_LEFT = 61
LIP_OUTER_RIGHT = 291
LIP_INNER_TOP = 0
LIP_INNER_BOTTOM = 17

class MediaPipeDetector:
    def __init__(self):
        try:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6
            )
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe FaceMesh: {e}")
            self.face_mesh = None

    def _euclidean_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        return float(np.linalg.norm(p1 - p2))

    def extract_lip_features(self, video_path: str) -> Tuple[np.ndarray, np.ndarray]:
        if self.face_mesh is None:
            logger.error("Face mesh not loaded. Skipping video.")
            return np.array([]), np.array([])
            
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps):
            fps = 30.0
            
        raw_features = []
        timestamps = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            t_sec = frame_idx / fps
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                coords = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in landmarks])
                
                outer_height = self._euclidean_distance(coords[LIP_OUTER_TOP], coords[LIP_OUTER_BOTTOM])
                outer_width = self._euclidean_distance(coords[LIP_OUTER_LEFT], coords[LIP_OUTER_RIGHT])
                inner_height = self._euclidean_distance(coords[LIP_INNER_TOP], coords[LIP_INNER_BOTTOM])
                
                lip_aspect_ratio = outer_height / max(outer_width, 1e-6)
                lip_open_ratio = inner_height / max(outer_height, 1e-6)
                approx_mouth_area = (np.pi * (outer_width / 2.0) * (outer_height / 2.0))
                
                feature_vector = [outer_height, outer_width, inner_height, lip_aspect_ratio, lip_open_ratio, approx_mouth_area]
                raw_features.append(feature_vector)
                timestamps.append(t_sec)
            else:
                if len(raw_features) > 0:
                    raw_features.append(raw_features[-1])
                else:
                    raw_features.append([0.0] * 6)
                timestamps.append(t_sec)
                
            frame_idx += 1
            
        cap.release()
        
        feat_matrix = np.array(raw_features, dtype=np.float32)
        time_array = np.array(timestamps, dtype=np.float32)
        
        if len(feat_matrix) > 1:
            velocity = np.diff(feat_matrix, axis=0, prepend=feat_matrix[0:1])
            acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])
            feat_matrix = np.hstack((feat_matrix, velocity, acceleration))
        else:
            padding = np.zeros((feat_matrix.shape[0], 12), dtype=np.float32)
            feat_matrix = np.hstack((feat_matrix, padding))
            
        return feat_matrix, time_array

    def __del__(self):
        if hasattr(self, 'face_mesh') and self.face_mesh is not None:
            self.face_mesh.close()