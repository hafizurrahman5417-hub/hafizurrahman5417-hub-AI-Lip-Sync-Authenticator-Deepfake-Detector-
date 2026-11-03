import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import time
import requests

# -------------------------------------------------------------
# 1. PAGE CONFIGURATION & THEME SETUP
# -------------------------------------------------------------
st.set_page_config(
    page_title="AI Lip-Sync Authenticator",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Flashing Banners & Dark Mode Aesthetics
st.markdown("""
    <style>
    .authentic-banner {
        background-color: #0b3d0b; color: #4af54a; padding: 20px; 
        border-radius: 10px; text-align: center; font-size: 24px; font-weight: bold;
        border: 2px solid #4af54a;
    }
    .deepfake-banner {
        background-color: #3d0b0b; color: #f54a4a; padding: 20px; 
        border-radius: 10px; text-align: center; font-size: 24px; font-weight: bold;
        border: 2px solid #f54a4a;
        animation: blinker 1s linear infinite;
    }
    @keyframes blinker {
        50% { opacity: 0.5; }
    }
    .stream-chip {
        background-color: #1e1e1e; padding: 8px 15px; border-radius: 20px;
        font-family: monospace; font-size: 14px; border: 1px solid #444;
        display: inline-block; margin-right: 10px;
    }
    .chip-active { color: #00ff00; }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. SESSION STATE MANAGEMENT
# -------------------------------------------------------------
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'history_df' not in st.session_state:
    st.session_state.history_df = pd.DataFrame(columns=['Time', 'Match_Score', 'Audio_Energy', 'Mouth_Open', 'Verdict'])

# -------------------------------------------------------------
# 3. SIDEBAR: INPUT CONTROLS & SETUP
# -------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.markdown("---")
    
    source_option = st.radio(
        "Source Selector Toggle",
        ("Live Webcam", "Video Upload"),
        help="Choose the media feed for the AI engine."
    )
    
    uploaded_file = None
    if source_option == "Video Upload":
        uploaded_file = st.file_uploader("Upload Media (.mp4, .avi, .mov)", type=['mp4', 'avi', 'mov'])
        
    st.markdown("---")
    sensitivity_threshold = st.slider(
        "Mismatch Sensitivity Threshold (%)", 
        min_value=0, max_value=100, value=70, step=1,
        help="If the match probability drops below this, the system flags it as a Deepfake."
    )
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    if col1.button("▶ Start Analysis", use_container_width=True):
        st.session_state.is_running = True
    if col2.button("⏹ Stop Analysis", use_container_width=True):
        st.session_state.is_running = False
        
    if st.button("🔄 Reset Dashboard", use_container_width=True):
        st.session_state.is_running = False
        st.session_state.history_df = pd.DataFrame(columns=['Time', 'Match_Score', 'Audio_Energy', 'Mouth_Open', 'Verdict'])
        st.rerun()

# -------------------------------------------------------------
# 4. MAIN DASHBOARD UI
# -------------------------------------------------------------
st.title("🛡️ AI Lip-Sync Authenticator & Deepfake Detector")

# Stream Indicators
st.markdown(f"""
    <div>
        <span class="stream-chip">🎥 Video Stream: 30 FPS <span class="chip-active">[ACTIVE]</span></span>
        <span class="stream-chip">🎙️ Audio Stream: 16 kHz <span class="chip-active">[ACTIVE]</span></span>
    </div>
    <br>
""", unsafe_allow_html=True)

# Layout Columns
col_video, col_analytics = st.columns([1.2, 1])

with col_video:
    st.subheader("Visual Media Feed & Overlay")
    video_placeholder = st.empty()
    
with col_analytics:
    st.subheader("Real-Time Analytics")
    verdict_placeholder = st.empty()
    gauge_placeholder = st.empty()

st.markdown("---")
st.subheader("Synchronized Time-Series Analytics")
graph_placeholder = st.empty()

# Export Section
st.markdown("---")
export_placeholder = st.empty()

# -------------------------------------------------------------
# 5. UI COMPONENTS FUNCTIONS (Gauges & Graphs)
# -------------------------------------------------------------
def get_gauge_chart(score, threshold):
    color = "#4af54a" if score >= threshold else "#f54a4a"
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Match Probability", 'font': {'size': 20, 'color': 'white'}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0)",
            'steps': [
                {'range': [0, threshold], 'color': "rgba(245, 74, 74, 0.3)"},
                {'range': [threshold, 100], 'color': "rgba(74, 245, 74, 0.3)"}],
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig

def get_timeseries_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Time'], y=df['Audio_Energy'], mode='lines', name='Audio Pitch/Energy', line=dict(color='#00d4ff', width=2)))
    fig.add_trace(go.Scatter(x=df['Time'], y=df['Mouth_Open'], mode='lines', name='Mouth Opening Dist', line=dict(color='#ff00ff', width=2)))
    
    suspicious_df = df[df['Verdict'] == 'Suspicious']
    if not suspicious_df.empty:
        fig.add_trace(go.Scatter(
            x=suspicious_df['Time'], y=suspicious_df['Mouth_Open'], 
            mode='markers', name='Mismatch Detected',
            marker=dict(color='red', size=10, symbol='x')
        ))
        
    fig.update_layout(
        height=300, 
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Time (Seconds)", yaxis_title="Normalized Signal",
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# -------------------------------------------------------------
# 6. PROCESSING LOOP & API INTEGRATION
# -------------------------------------------------------------

if st.session_state.is_running:
    
    # === SCENARIO A: REAL BACKEND PROCESSING (VIDEO UPLOAD) ===
    if source_option == "Video Upload":
        if uploaded_file is None:
            st.warning("Please upload a video file first, or switch to Live Webcam.")
            st.session_state.is_running = False
            st.rerun()
        else:
            with st.spinner("⏳ Sending video to FastAPI Backend for multi-modal analysis..."):
                try:
                    # Prepare the file for the POST request
                    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    
                    # Call your FastAPI backend
                    response = requests.post("http://localhost:8000/analyze", files=files)
                    
                    if response.status_code == 200:
                        results = response.json()
                        match_score = results['match_probability_score'] * 100 # Convert to percentage
                        verdict = results['decision']
                        
                        st.success(f"✅ Analysis Complete! Processed {results['processed_frames']} frames.")
                        
                        # Update the Gauge Chart with REAL Data
                        gauge_placeholder.plotly_chart(get_gauge_chart(match_score, sensitivity_threshold), use_container_width=True)
                        
                        # Update Banners
                        if verdict == "Authentic":
                            verdict_placeholder.markdown('<div class="authentic-banner">✅ VERIFIED AUTHENTIC</div>', unsafe_allow_html=True)
                        else:
                            verdict_placeholder.markdown('<div class="deepfake-banner">⚠️ WARNING: AUDIO-VISUAL MISMATCH DETECTED</div>', unsafe_allow_html=True)
                        
                        # Display the video
                        video_placeholder.video(uploaded_file)
                        
                    else:
                        st.error(f"Backend Error: {response.text}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("🚨 Could not connect to backend! Is your FastAPI server running on port 8000?")
            
            # Stop running after the file is processed
            st.session_state.is_running = False

    # === SCENARIO B: LIVE BOOTH DEMO (WEBCAM SIMULATION) ===
    elif source_option == "Live Webcam":
        cap = cv2.VideoCapture(0) # Open Webcam
        start_time = time.time()
        
        while st.session_state.is_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            current_time = time.time() - start_time
            
            # --- SIMULATING BACKEND AI PIPELINE DATA FOR LIVE DEMO ---
            base_signal = np.abs(np.sin(current_time * 3))
            mouth_open = base_signal + np.random.normal(0, 0.05)
            
            is_simulated_fake = np.random.rand() > 0.85
            if is_simulated_fake:
                audio_energy = base_signal * 0.2
                match_score = np.random.uniform(20, sensitivity_threshold - 1)
                verdict = "Suspicious"
                mesh_color = (0, 0, 255) # Red
            else:
                audio_energy = base_signal + np.random.normal(0, 0.05)
                match_score = np.random.uniform(sensitivity_threshold + 5, 99)
                verdict = "Authentic"
                mesh_color = (0, 255, 0) # Green
            # -------------------------------------------------------------

            h, w, _ = frame.shape
            center_x, center_y = int(w/2), int(h/2) + 50
            lip_w, lip_h = int(60 * (1 + mouth_open)), int(20 * (1 + mouth_open))
            cv2.ellipse(frame, (center_x, center_y), (lip_w, lip_h), 0, 0, 360, mesh_color, 2)
            cv2.putText(frame, f"Viseme Sync: {verdict}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mesh_color, 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame, channels="RGB", use_container_width=True)
            
            gauge_placeholder.plotly_chart(get_gauge_chart(match_score, sensitivity_threshold), use_container_width=True)
            
            if verdict == "Authentic":
                verdict_placeholder.markdown('<div class="authentic-banner">✅ VERIFIED AUTHENTIC</div>', unsafe_allow_html=True)
            else:
                verdict_placeholder.markdown('<div class="deepfake-banner">⚠️ WARNING: AUDIO-VISUAL MISMATCH DETECTED</div>', unsafe_allow_html=True)
                
            new_row = pd.DataFrame([{
                'Time': round(current_time, 2), 
                'Match_Score': match_score, 
                'Audio_Energy': audio_energy, 
                'Mouth_Open': mouth_open, 
                'Verdict': verdict
            }])
            st.session_state.history_df = pd.concat([st.session_state.history_df, new_row], ignore_index=True)
            
            if len(st.session_state.history_df) > 100:
                st.session_state.history_df = st.session_state.history_df.tail(100)
                
            graph_placeholder.plotly_chart(get_timeseries_chart(st.session_state.history_df), use_container_width=True)
            time.sleep(0.05) 

        cap.release()

# -------------------------------------------------------------
# 7. EXPORT & REPORTING
# -------------------------------------------------------------
if not st.session_state.history_df.empty:
    csv = st.session_state.history_df.to_csv(index=False).encode('utf-8')
    export_placeholder.download_button(
        label="📄 Download Analysis Summary (CSV Report)",
        data=csv,
        file_name='deepfake_analysis_report.csv',
        mime='text/csv',
    )