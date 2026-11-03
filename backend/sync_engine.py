import numpy as np
from scipy.interpolate import interp1d
import logging

logger = logging.getLogger("SyncEngine")

class SyncEngine:
    @staticmethod
    def synchronize_modalities(
        lip_feats: np.ndarray, 
        video_times: np.ndarray, 
        audio_feats: np.ndarray, 
        audio_times: np.ndarray
    ) -> np.ndarray:
        """
        Aligns asynchronous temporal rates (e.g., 30 FPS Video vs. 31.25 Hz Audio Frames)
        using 1D Linear Interpolation, projecting visual attributes precisely onto audio timestamps.
        """
        if len(video_times) == 0 or len(audio_times) == 0:
            raise ValueError("Temporal vectors must contain elements to execute synchronization.")

        # Establish shared operational time boundaries
        min_time = max(video_times[0], audio_times[0])
        max_time = min(video_times[-1], audio_times[-1])
        
        if min_time >= max_time:
            logger.warning("Minimal overlapping window between audio/video sequences.")
            # Fallback constraint resolution
            max_time = max(video_times[-1], audio_times[-1])
            
        # Mask matching audio frames within valid intersection spaces
        audio_mask = (audio_times >= min_time) & (audio_times <= max_time)
        sync_audio_times = audio_times[audio_mask]
        sync_audio_feats = audio_feats[audio_mask]
        
        if len(sync_audio_times) == 0:
            # Edge-case processing configuration
            sync_audio_times = audio_times
            sync_audio_feats = audio_feats

        # Construct continuous linear functions along geometric arrays
        interp_func = interp1d(
            video_times, 
            lip_feats, 
            axis=0, 
            kind='linear', 
            fill_value="extrapolate"
        )
        
        # Rescale lip coordinate properties onto unified timeline arrays
        sync_lip_feats = interp_func(sync_audio_times)
        
        # Perform feature fusion
        multimodal_feature_matrix = np.hstack((sync_lip_feats, sync_audio_feats))
        logger.info(f"Cross-modal synchronization achieved. Consolidated matrix shape: {multimodal_feature_matrix.shape}")
        
        return multimodal_feature_matrix.astype(np.float32)