import streamlit as st
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# Import prediction modules
from predict import load_and_prep_image, process_prediction, get_xai_explanations, validate_image_modality

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
    /* Global Font and Body */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #0d47a1; /* Dark Blue */
        font-weight: 600;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #f0f4f8; /* Very light blue-grey */
    }
    
    /* Custom buttons */
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
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    /* Cards for metrics */
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
# APP LAYOUT
# ---------------------------------------------------------
st.title("⚕️ Medical Imaging Diagnostic System")
st.markdown("*Advanced Disease Detection powered by Explainable AI (XAI)*")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063206.png", width=100)
    st.header("Diagnostic Settings")
    model_selection = st.radio(
        "Select Imaging Modality",
        ("Chest X-Ray (Multi-class)", "Brain MRI (Binary)")
    )
    st.markdown("---")
    st.info("Upload a medical image to receive a diagnosis alongside Explainable AI heatmaps ensuring absolute clinical transparency.")

# Determine model type
model_type = 'chest' if 'Chest' in model_selection else 'brain'

# Main UI
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Patient Imaging Data")
    uploaded_file = st.file_uploader("Upload Image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

with col2:
    if uploaded_file is not None:
        st.subheader("AI Diagnosis Report")
        
        with st.spinner("Analyzing Imaging Data..."):
            try:
                # 1. Prep image
                img_array = load_and_prep_image(uploaded_file)
                
                if not validate_image_modality(img_array, model_type):
                    st.error(f"🛑 Modality Mismatch: The uploaded image does not appear to be a {model_type.upper()} scan. Please upload the correct type of medical image for this model.")
                    st.stop()
                
                # 2. Predict
                class_name, confidence, pred_idx, model = process_prediction(img_array, model_type)
                
                # 3. Display Results
                color = "green" if class_name == "NORMAL" else "red"
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin-top: 0;">Predicted Diagnosis: <span style="color: {color};">{class_name}</span></h3>
                    <h4 style="margin-top: 0; color: #555;">Confidence Score: {confidence*100:.2f}%</h4>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### Explainable AI (XAI) Analysis")
                st.write("Generating deep visualizations. This may take a moment...")
                
                # 4. Generate Explanations
                with st.spinner("Generating Grad-CAM, LIME, and SHAP visualizations..."):
                    gradcam_img, lime_img, shap_fig = get_xai_explanations(img_array, model, uploaded_file, model_type)
                
                # 5. Show Explanations in tabs
                tab1, tab2, tab3 = st.tabs(["Grad-CAM", "LIME", "SHAP"])
                
                with tab1:
                    st.write("**Gradient-weighted Class Activation Mapping** highlights the spatial regions in the image most responsible for the network's decision.")
                    st.image(gradcam_img, caption="Grad-CAM Heatmap", use_column_width=True)
                    
                with tab2:
                    st.write("**Local Interpretable Model-agnostic Explanations** highlights superpixels contributing positively to the model's prediction.")
                    # LIME returns values in [0, 1] as float
                    fig, ax = plt.subplots()
                    ax.imshow(lime_img)
                    ax.axis('off')
                    st.pyplot(fig)
                    
                with tab3:
                    st.write("**SHapley Additive exPlanations** provides feature importance maps at scale.")
                    st.pyplot(shap_fig)
                    
            except FileNotFoundError as e:
                 st.error(f"Error loading model resources. Please ensure the models are trained and saved. {str(e)}")
            except Exception as e:
                 st.error(f"An error occurred during diagnosis: {str(e)}")
    else:
        st.info("Awaiting image upload to proceed with analysis.")
