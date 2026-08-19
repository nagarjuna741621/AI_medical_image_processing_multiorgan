import tensorflow as tf
import numpy as np
from PIL import Image
import os

from config import CHEST_MODEL_PATH, BRAIN_MODEL_PATH, CHEST_CLASSES, BRAIN_CLASSES, IMAGE_SIZE
from gradcam import make_gradcam_heatmap, overlay_gradcam
from lime_explainer import explain_with_lime
from shap_explainer import explain_with_shap
from logger import get_logger

log = get_logger("predict")

# Cache models to avoid reloading
_MODELS = {}

def load_model(model_type):
    if model_type in _MODELS:
        return _MODELS[model_type]
    
    path = CHEST_MODEL_PATH if model_type == 'chest' else BRAIN_MODEL_PATH
    if not os.path.exists(path):
        log.error(f"Model {path} not found.")
        return None
        
    model = tf.keras.models.load_model(path)
    _MODELS[model_type] = model
    return model

def load_and_prep_image(image_file):
    """Loads image from file upload or path and prepares it for prediction."""
    img = Image.open(image_file).convert('RGB')
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img)
    # The image is kept in [0, 255] for EfficientNet
    return img_array

def validate_image_modality(img_array, selected_model_type):
    """Checks if the uploaded image matches the requested model type (Brain vs Chest)."""
    modality_path = os.path.join(os.path.dirname(CHEST_MODEL_PATH), 'modality_model.keras')
    if not os.path.exists(modality_path):
        return True # Skip validation if model isn't built yet
        
    try:
        if 'modality' not in _MODELS:
            _MODELS['modality'] = tf.keras.models.load_model(modality_path)
            
        modality_model = _MODELS['modality']
        
        # Resize to 128x128 as expected by modality_model
        img_resized = tf.image.resize(img_array, (128, 128))
        img_batch = np.expand_dims(img_resized, axis=0)
        
        # 0 = brain, 1 = chest
        pred = modality_model.predict(img_batch, verbose=0)[0][0]
        
        if selected_model_type == 'chest' and pred < 0.2:
            return False # Looks like brain
        if selected_model_type == 'brain' and pred > 0.8:
            return False # Looks like chest
            
        return True
    except Exception as e:
        log.error(f"Modality validation failed: {e}")
        return True

def process_prediction(img_array, model_type):
    model = load_model(model_type)
    if model is None:
        raise ValueError("Model not available.")
        
    img_batch = np.expand_dims(img_array, axis=0)
    
    preds = model.predict(img_batch)
    
    if model_type == 'chest':
        pred_idx = np.argmax(preds[0])
        confidence = float(preds[0][pred_idx])
        class_name = CHEST_CLASSES[pred_idx]
    else:  # brain
        confidence = float(preds[0][0])
        # Binary: if > 0.5 then TUMOR (index 1), else NORMAL (index 0)
        pred_idx = 1 if confidence > 0.5 else 0
        # If class 0, confidence of class 0 is 1 - confidence
        if pred_idx == 0:
            confidence = 1.0 - confidence
        class_name = BRAIN_CLASSES[pred_idx]
        
    return class_name, confidence, pred_idx, model

def get_xai_explanations(img_array, model, img_path_or_file, model_type):
    log.info("Generating Grad-CAM...")
    # Finding last conv layer dynamically might fail if naming varies, we fallback if needed.
    # We will use the fact that last layer in EfficientNetB0 is 'top_activation'
    heatmap = make_gradcam_heatmap(img_batch:=np.expand_dims(img_array, axis=0), model, 'top_activation')
    
    # We need a temporary path if img_path_or_file is a file buffer (Streamlit)
    if not isinstance(img_path_or_file, str):
        # image is already loaded in img_array
        # Create temp file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp:
            Image.fromarray(img_array).save(temp.name)
            gradcam_img = overlay_gradcam(temp.name, heatmap)
            os.remove(temp.name)
    else:
        gradcam_img = overlay_gradcam(img_path_or_file, heatmap)
        
    log.info("Generating LIME...")
    lime_img = explain_with_lime(model, img_array, num_classes=3 if model_type=='chest' else 2)
    
    log.info("Generating SHAP...")
    shap_fig = explain_with_shap(model, img_array)
    
    return gradcam_img, lime_img, shap_fig
