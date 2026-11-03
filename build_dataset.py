import os
import sys
import pandas as pd
import numpy as np
import logging

# --- PATH FIX ---
# This tells Python to look inside the "backend" folder for your modules!
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# Import your existing backend extraction tools (no 'backend.' prefix needed now)
from mediapipe_detector import MediaPipeDetector
from audio_processor import AudioProcessor
from sync_engine import SyncEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DatasetBuilder")

def extract_video_features(video_path: str, label: int):
    """Runs a single video through your extraction pipeline and flattens it."""
    visual_detector = MediaPipeDetector()
    acoustic_processor = AudioProcessor()
    
    try:
        # 1. Extract raw data
        lip_features, video_timestamps = visual_detector.extract_lip_features(video_path)
        audio_features, audio_timestamps = acoustic_processor.extract_acoustic_features(video_path)
        
        # 2. Sync the timelines
        fused_matrix = SyncEngine.synchronize_modalities(
            lip_features, video_timestamps, audio_features, audio_timestamps
        )
        
        # 3. Flatten into a single row of statistics for XGBoost (Mean & Std Dev)
        mean_features = np.mean(fused_matrix, axis=0)
        std_features = np.std(fused_matrix, axis=0)
        final_vector = np.hstack((mean_features, std_features))
        
        # 4. Add the label (1 for Real, 0 for Fake) at the end
        final_vector = np.append(final_vector, label)
        return final_vector
        
    except Exception as e:
        logger.error(f"Failed to process {video_path}: {e}")
        return None

def process_folder(folder_path: str, label: int):
    dataset_rows = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.mp4', '.avi', '.mov')):
            logger.info(f"Processing: {filename}")
            filepath = os.path.join(folder_path, filename)
            
            features = extract_video_features(filepath, label)
            if features is not None:
                dataset_rows.append(features)
    return dataset_rows

if __name__ == "__main__":
    print("🚀 Starting Data Extraction...")
    
    # Define paths to your video folders
    REAL_VIDEOS_PATH = "dataset/real"
    FAKE_VIDEOS_PATH = "dataset/fake"
    
    # Process both folders
    real_data = process_folder(REAL_VIDEOS_PATH, label=1)
    fake_data = process_folder(FAKE_VIDEOS_PATH, label=0)
    
    # Combine and save to CSV
    all_data = real_data + fake_data
    
    # SAFETY CHECK: Prevent Pandas from crashing if extraction fails
    if len(all_data) == 0:
        print("\n❌ FATAL ERROR: No data was extracted! The dataset is empty.")
        print("This means MediaPipe or AudioProcessor failed on every video.")
        sys.exit(1)
        
    df = pd.DataFrame(all_data)
    
    # The last column is the target label
    cols = [f"feature_{i}" for i in range(df.shape[1] - 1)] + ["label"]
    df.columns = cols
    
    df.to_csv("dataset/final_training_data.csv", index=False)
    print("✅ Extraction Complete! Saved to final_training_data.csv")