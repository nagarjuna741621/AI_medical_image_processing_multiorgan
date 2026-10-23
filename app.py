import os
import sys

# Ensure repository root is on sys.path
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Import prediction and XAI modules from modular src package
from src.pipeline.predict import (
    load_and_prep_image,
    process_prediction,
    get_xai_explanations,
    validate_image_modality,
)
from src.config import CHEST_MODEL_PATH, BRAIN_MODEL_PATH

# ---------------------------------------------------------
# UI CONFIGURATION & STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="Medical AI Diagnostic System",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Medical Theme Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        color: #0d47a1;
        font-weight: 700;
    }
    
    div.stButton > button:first-child {
        background-color: #1976d2; 
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
    }
    
    div.stButton > button:first-child:hover {
        background-color: #1565c0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.12);
    }

    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #1976d2;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# APP HEADER
# ---------------------------------------------------------
st.title("⚕️ Medical AI Multi-Organ Diagnostic Suite")
st.markdown("*Clinical-grade multi-organ disease classification with Tri-Modal Explainable AI (XAI)*")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("<div style='font-size: 3rem; text-align: center;'>⚕️</div>", unsafe_allow_html=True)
    st.header("Diagnostic Controls")
    model_selection = st.radio(
        "Select Target Anatomical Modality",
        ("Chest X-Ray (Multi-class: Normal / Pneumonia / TB)", "Brain MRI (Binary: Normal / Tumor)")
    )
    st.markdown("---")
    st.subheader("System Status")
    chest_ok = "✅ Ready" if os.path.exists(CHEST_MODEL_PATH) else "⚠️ Missing"
    brain_ok = "✅ Ready" if os.path.exists(BRAIN_MODEL_PATH) else "⚠️ Missing"
    st.markdown(f"**Chest Diagnostic Engine:** {chest_ok}")
    st.markdown(f"**Brain Diagnostic Engine:** {brain_ok}")
    st.markdown("---")
    st.info("Upload a patient radiograph or scan to compute automated triage classification alongside Grad-CAM heatmaps, LIME superpixels, and SHAP feature attribution.")

# Determine model type
model_type = 'chest' if 'Chest' in model_selection else 'brain'

# Main UI
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Patient Imaging Input")
    uploaded_file = st.file_uploader("Upload Radiograph or MRI Scan...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Patient Scan", use_container_width=True)

with col2:
    if uploaded_file is not None:
        st.subheader("Automated Clinical Analysis")
        
        with st.spinner("Processing image and executing neural inference..."):
            try:
                # 1. Prep image
                img_array = load_and_prep_image(uploaded_file)
                
                # Modality validation triage
                if not validate_image_modality(img_array, model_type):
                    st.error(f"🛑 Modality Mismatch Detected: The uploaded image does not appear to match the selected '{model_type.upper()}' engine. Please ensure the scan corresponds to the selected modality.")
                    st.stop()
                
                # 2. Predict
                class_name, confidence, pred_idx, model = process_prediction(img_array, model_type)
                
                # 3. Display Results
                color = "#16a34a" if class_name == "NORMAL" else "#dc2626"
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin-top: 0;">Predicted Diagnosis: <span style="color: {color};">{class_name}</span></h3>
                    <h4 style="margin-top: 0; color: #475569;">Confidence Score: {confidence*100:.2f}% <span style="font-size: 0.85rem; color: #94a3b8;">(Softmax Probability)</span></h4>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### Tri-Modal Explainable AI (XAI) Verification")
                st.write("Generating multi-granular morphological feature verification:")
                
                # 4. Generate Explanations
                with st.spinner("Generating Grad-CAM, LIME, and SHAP visual explanations..."):
                    gradcam_img, lime_img, shap_fig = get_xai_explanations(img_array, model, uploaded_file, model_type, pred_idx)
                
                # 5. Show Explanations in tabs
                tab1, tab2, tab3 = st.tabs(["1. Grad-CAM (Heatmap)", "2. LIME (Superpixels)", "3. SHAP (Attribution)"])
                
                with tab1:
                    st.write("**Gradient-weighted Class Activation Mapping (Grad-CAM)**: Highlights pathological activation focal points in the final convolutional layer.")
                    st.image(gradcam_img, caption="Grad-CAM Pathology Localization Heatmap", use_container_width=True)
                    
                with tab2:
                    st.write("**Local Interpretable Model-agnostic Explanations (LIME)**: Delineates local superpixel boundaries supporting and opposing the diagnosis.")
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.imshow(lime_img)
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close(fig)
                    
                with tab3:
                    st.write("**SHapley Additive exPlanations (SHAP)**: Computes pixel-level Shapley values attributing positive and negative marginal contributions relative to background reference.")
                    st.pyplot(shap_fig)
                    plt.close(shap_fig)
                    
            except FileNotFoundError as e:
                st.error(f"Error loading model resources: {str(e)}")
            except Exception as e:
                st.error(f"Diagnostic error: {str(e)}")
    else:
        st.info("Awaiting scan upload. Select an imaging modality and upload a medical image to run the diagnostic pipeline.")
