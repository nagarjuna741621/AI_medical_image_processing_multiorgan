import os
import tensorflow as tf
import numpy as np
from PIL import Image

from src.config import (
    CHEST_MODEL_PATH,
    BRAIN_MODEL_PATH,
    MODALITY_MODEL_PATH,
    CHEST_CLASSES,
    BRAIN_CLASSES,
    IMAGE_SIZE,
)
from src.models.load_model import load_medical_model
from src.xai.gradcam import make_gradcam_heatmap, overlay_gradcam
from src.xai.lime_explainer import explain_with_lime
from src.xai.shap_explainer import explain_with_shap
from src.logger import get_logger

log = get_logger("predict")

# Cache loaded models in memory
_MODELS = {}

def get_loaded_model(model_type):
    """Loads and caches the requested clinical model."""
    if model_type in _MODELS:
        return _MODELS[model_type]
    
    path = CHEST_MODEL_PATH if model_type == 'chest' else BRAIN_MODEL_PATH
    if not os.path.exists(path):
        log.error(f"Model {path} not found.")
        return None
        
    model = load_medical_model(path)
    _MODELS[model_type] = model
    return model

def load_and_prep_image(image_file):
    """Loads image from file upload or path and prepares it for prediction."""
    img = Image.open(image_file).convert('RGB')
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img)
    return img_array

def validate_image_modality(img_array, selected_model_type):
    """Checks if the uploaded image matches the requested model type (Brain vs Chest)."""
    if not os.path.exists(MODALITY_MODEL_PATH):
        return True
        
    try:
        if 'modality' not in _MODELS:
            _MODELS['modality'] = tf.keras.models.load_model(MODALITY_MODEL_PATH)
            
        modality_model = _MODELS['modality']
        
        # Resize to 128x128 as expected by modality triage classifier
        img_resized = tf.image.resize(img_array, (128, 128))
        img_batch = np.expand_dims(img_resized, axis=0)
        
        # 0 = brain, 1 = chest
        pred = float(modality_model.predict(img_batch, verbose=0)[0][0])
        
        if selected_model_type == 'chest' and pred < 0.2:
            return False  # Input appears to be Brain MRI
        if selected_model_type == 'brain' and pred > 0.8:
            return False  # Input appears to be Chest X-Ray
            
        return True
    except Exception as e:
        log.error(f"Modality validation check failed: {e}")
        return True

def process_prediction(img_array, model_type):
    """Runs inference and returns predicted class, confidence, and class index."""
    model = get_loaded_model(model_type)
    if model is None:
        raise ValueError(f"Model for '{model_type}' is not available.")
        
    img_batch = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_batch, verbose=0)
    
    if model_type == 'chest':
        pred_idx = int(np.argmax(preds[0]))
        confidence = float(preds[0][pred_idx])
        class_name = CHEST_CLASSES[pred_idx]
    else:  # brain
        confidence = float(preds[0][0])
        # Binary: if > 0.5 then TUMOR (index 1), else NORMAL (index 0)
        pred_idx = 1 if confidence > 0.5 else 0
        if pred_idx == 0:
            confidence = 1.0 - confidence
        class_name = BRAIN_CLASSES[pred_idx]
        
    return class_name, confidence, pred_idx, model

def get_xai_explanations(img_array, model, img_path_or_file, model_type, pred_idx=None):
    """Generates Tri-Modal Explainable AI outputs (Grad-CAM, LIME, SHAP)."""
    log.info("Generating Grad-CAM class activation heatmap...")
    heatmap = make_gradcam_heatmap(
        np.expand_dims(img_array, axis=0), 
        model, 
        pred_index=pred_idx
    )
    gradcam_img = overlay_gradcam(img_array, heatmap)
        
    log.info("Generating LIME superpixel segmentations...")
    lime_img = explain_with_lime(model, img_array, num_classes=3 if model_type == 'chest' else 2)
    
    log.info("Generating SHAP feature attributions...")
    return gradcam_img, lime_img, shap_fig

def run_cli_prediction(image_path, modality=None):
    """Runs command-line inference on a given image file."""
    import argparse
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' does not exist.")
        return
        
    print(f"\nProcessing medical scan: {image_path}")
    img_array = load_and_prep_image(image_path)
    
    # Auto-detect modality if not specified
    if modality is None:
        if os.path.exists(MODALITY_MODEL_PATH):
            try:
                if 'modality' not in _MODELS:
                    _MODELS['modality'] = tf.keras.models.load_model(MODALITY_MODEL_PATH)
                mod_model = _MODELS['modality']
                resized = tf.image.resize(img_array, (128, 128))
                score = float(mod_model.predict(np.expand_dims(resized, 0), verbose=0)[0][0])
                modality = 'chest' if score >= 0.5 else 'brain'
                print(f"Modality Router Auto-Detection: {modality.upper()} (score: {score:.3f})")
            except Exception as e:
                print(f"Modality triage router fallback to chest: {e}")
                modality = 'chest'
        else:
            modality = 'chest'
            
    class_name, confidence, pred_idx, _ = process_prediction(img_array, modality)
    print("\n" + "=" * 45)
    print("      CLINICAL DIAGNOSTIC REPORT")
    print("=" * 45)
    print(f" Target Modality:    {modality.upper()}")
    print(f" Predicted Finding:  {class_name}")
    print(f" Confidence Score:   {confidence:.2%}")
    print("=" * 45 + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Medical Image AI Diagnostic Predictor")
    parser.add_argument("--image", type=str, required=True, help="Path to input medical image scan")
    parser.add_argument("--modality", type=str, choices=["chest", "brain"], default=None, help="Force modality target (chest or brain)")
    args = parser.parse_args()
    run_cli_prediction(args.image, args.modality)

