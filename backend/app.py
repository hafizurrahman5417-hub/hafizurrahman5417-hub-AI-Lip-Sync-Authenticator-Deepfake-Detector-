import os
import shutil
import tempfile
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Pipeline imports
from  mediapipe_detector import MediaPipeDetector
from  audio_processor import AudioProcessor
from  sync_engine import SyncEngine
from  classifier import LipSyncClassifier

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("FastAPICore")

app = FastAPI(
    title="AI Lip-Sync Authenticator & Deepfake Detection Engine",
    description="Production-grade multi-modal analytical API parsing acoustic/visual signals.",
    version="1.0.0"
)

# Cross-Origin Resource Sharing protocols
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate core architectural blocks
visual_detector = MediaPipeDetector()
acoustic_processor = AudioProcessor()
classifier = LipSyncClassifier()

class DetectionOutputSchema(BaseModel):
    match_probability_score: float
    decision: str
    processed_frames: int
    status: str

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "model_loaded": classifier.model is not None,
        "runtime_environment": "Python 2026 Sandbox"
    }

@app.post("/analyze", response_model=DetectionOutputSchema)
async def analyze_media_feed(file: UploadFile = File(...)):
    logger.info(f"Received processing request for file asset: {file.filename}")
    
    # Establish a safe, isolated directory context to process raw file writes
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename)
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Pipeline Execution Step A: Parallel Visual / Acoustic Landmark Extraction
        lip_features, video_timestamps = visual_detector.extract_lip_features(temp_file_path)
        audio_features, audio_timestamps = acoustic_processor.extract_acoustic_features(temp_file_path)
        
        if len(lip_features) == 0 or len(audio_features) == 0:
            raise HTTPException(
                status_code=400, 
                detail="Multi-modal matrix validation failed. Incomplete feature collection loops."
            )
            
        # Pipeline Execution Step B: Cross-Modal Synchronization & Temporal Interpolation
        fused_feature_matrix = SyncEngine.synchronize_modalities(
            lip_features, video_timestamps,
            audio_features, audio_timestamps
        )
        
        # Pipeline Execution Step C: Machine Learning Classification Inference
        probability_score = classifier.predict_probability(fused_feature_matrix)
        
        # System decision configuration based on threshold metrics
        decision_verdict = "Authentic" if probability_score >= 0.68 else "Suspicious"
        
        return DetectionOutputSchema(
            match_probability_score=round(probability_score, 4),
            decision=decision_verdict,
            processed_frames=int(fused_feature_matrix.shape[0]),
            status="Processing Complete"
        )
        
    except Exception as runtime_error:
        logger.error(f"Fatal error within active pipeline: {str(runtime_error)}", exc_info=True)
        return DetectionOutputSchema(
            match_probability_score=0.0,
            decision="Suspicious",
            processed_frames=0,
            status=f"Pipeline Failure: {str(runtime_error)}"
        )
        
    finally:
        # Clean up temporary disk files to prevent memory leakages
        try:
            shutil.rmtree(temp_dir)
            logger.info("Temporary storage volumes successfully purged.")
        except Exception as cleanup_err:
            logger.error(f"Error wiping system temporary files: {str(cleanup_err)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)