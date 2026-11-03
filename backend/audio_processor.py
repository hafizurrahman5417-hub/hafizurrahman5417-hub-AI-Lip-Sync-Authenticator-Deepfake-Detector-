import librosa
import numpy as np
import logging
import os
import tempfile

logger = logging.getLogger("AudioProcessor")

class AudioProcessor:
    def __init__(self, sample_rate=16000):
        self.sr = sample_rate

    def extract_acoustic_features(self, video_path: str):
        try:
            # 1. MoviePy v2.x fix: The '.editor' submodule was removed.
            # We must import directly from moviepy now!
            from moviepy import VideoFileClip
            
            # Create a safe temporary path for the audio
            temp_audio_path = os.path.join(tempfile.gettempdir(), f"temp_audio_{os.path.basename(video_path)}.wav")
            
            # 2. Extract audio from the video
            video = VideoFileClip(video_path)
            if video.audio is None:
                logger.error("No audio track found in video.")
                return np.array([]), np.array([])
                
            # Write to a temporary file (removed the deprecated logger=None parameter)
            video.audio.write_audiofile(temp_audio_path, fps=self.sr)
            video.close()
            
            # 3. Now librosa can read the clean WAV file without crashing
            y, sr = librosa.load(temp_audio_path, sr=self.sr)
            
            # Clean up the temp file from your hard drive
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                
            # 4. Extract MFCCs (Acoustic Features)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfccs = mfccs.T # Transpose to match (frames, features) shape
            
            # Generate timestamps
            hop_length = 512
            timestamps = librosa.frames_to_time(np.arange(len(mfccs)), sr=sr, hop_length=hop_length)
            
            return mfccs, timestamps
            
        except Exception as e:
            logger.error(f"Failed to process audio stream context: {e}")
            return np.array([]), np.array([])