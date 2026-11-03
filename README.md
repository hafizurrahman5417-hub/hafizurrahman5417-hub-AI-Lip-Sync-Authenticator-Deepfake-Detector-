# AI-Lip-Sync Authenticator & Deepfake Detector

Real-time multi-modal deepfake detection system leveraging Google MediaPipe for lip-geometry kinetics, Librosa for acoustic analysis, and an XGBoost decision engine. 

Traditional image-only deepfake detectors are easily bypassed by high-definition, generative AI video clones. This solution acts like a human lip-reader, cross-referencing audio phonemes with precise physical lip movements to catch millisecond mismatches and deliver edge-optimized authenticity verdicts.

## 🗂 Project Structure
Below is the core directory structure for this project. Note that the raw `dataset/` folder and generated CSVs are excluded from version control to maintain a lightweight repository.

```text
Project_Name/
├── backend/
│   ├── app.py                 # FastAPI backend server
│   ├── audio_processor.py     # Librosa audio feature extraction
│   ├── classifier.py          # XGBoost inference logic
│   ├── config.py              # Application configurations
│   ├── mediapipe_detector.py  # 3D Lip-geometry extraction
│   └── sync_engine.py         # Audio-visual alignment logic
├── frontend/
│   └── dashboard.py           # Streamlit interactive user interface
├── models/
│   └── xgboost_v1.pkl         # Trained XGBoost model binary
├── build_dataset.py           # Script to extract multi-modal features into CSV
├── train_xgboost.py           # Script to train the XGBoost classifier
├── .gitignore                 # Untracked files (dataset/, environments, pycache)
└── README.md                  # Project documentation
```
##📊 Data Access & Dataset Setup
To keep this source repository clean and fast, the raw video datasets and large CSV feature files are hosted externally.

Data Access Link: https://www.kaggle.com/datasets/hafizur19/multi-modal-deepfake-video-dataset

How to set up the data locally:

Download the dataset zip file from the link above.

Extract the downloaded folder.

Place the extracted dataset/ folder directly into the root directory of this project ("Folder Name or Project Name"/dataset/).

(Optional) If you wish to extract features from scratch and retrain the model on your own hardware, run python build_dataset.py followed by python train_xgboost.py.

🚀 How to Run the Application
To run the full-stack system, open two separate terminal windows in your IDE.

Terminal 1: Start the FastAPI Backend

PowerShell
cd backend
python -m uvicorn app:app --reload
(Ensure it is running on http://127.0.0.1:8000)

Terminal 2: Start the Streamlit Frontend Dashboard

PowerShell
cd frontend
python -m streamlit run dashboard.py
Once both servers are running, open the local Streamlit URL in your browser to access the control panel and upload videos for real-time deepfake analysis.
