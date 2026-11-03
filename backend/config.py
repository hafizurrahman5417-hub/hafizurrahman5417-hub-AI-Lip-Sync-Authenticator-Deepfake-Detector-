import os

# Directory Routing
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.normpath(os.path.join(BASE_DIR, "../models"))
DATASET_DIR = os.path.normpath(os.path.join(BASE_DIR, "../dataset"))

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# Audio Hyperparameters
AUDIO_SAMPLE_RATE = 16000  # Standardized sampling rate for speech phonemes
N_MFCC = 13               # Number of Mel-Frequency Cepstral Coefficients
HOP_LENGTH = 512          # Step size for STFT windows (~32ms frames)

# Video Processing Configurations
TARGET_FPS = 30.0

# MediaPipe Landmark Index Mapping (Face Mesh Topology)
# Specific indices tracking upper/lower and left/right internal and external lip margins
LIP_OUTER_TOP = 0
LIP_OUTER_BOTTOM = 17
LIP_OUTER_LEFT = 61
LIP_OUTER_RIGHT = 291

LIP_INNER_TOP = 13
LIP_INNER_BOTTOM = 14

MODEL_PATH = os.path.join(MODEL_DIR, "lipsync_detector.pkl")