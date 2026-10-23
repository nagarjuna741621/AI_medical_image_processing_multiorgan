import shap
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def explain_with_shap(model, image, background_images=None, output_path=None):
    """
    Generate SHAP GradientExplainer visualization for image inputs.
    
    Args:
        model: Trained Keras model
        image: Single image array to explain (shape matching model input, [0, 255])
        background_images: Reference background for GradientExplainer.
    """
    img_float = image.astype(np.float32)
    if img_float.max() > 1.0:
        img_norm = img_float / 255.0
    else:
        img_norm = img_float

    img_batch = np.expand_dims(img_norm * 255.0, axis=0)
    
    if background_images is None:
        background_images = np.zeros((1, *image.shape), dtype=np.float32)
        
    explainer = shap.GradientExplainer(model, background_images)
    res = explainer.shap_values(img_batch, ranked_outputs=1)
    if isinstance(res, tuple) and len(res) == 2:
        shap_values, indexes = res
    else:
        shap_values = res
    
    if isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 5 and shap_values.shape[-1] == 1:
            shap_values = np.squeeze(shap_values, axis=-1)
        shap_list = [shap_values]
    elif isinstance(shap_values, list):
        shap_list = [
            np.squeeze(s, axis=-1) if (isinstance(s, np.ndarray) and s.ndim == 5 and s.shape[-1] == 1) else s
            for s in shap_values
        ]
    else:
        shap_list = list(shap_values)

    pixel_values = np.expand_dims(img_norm, axis=0)
    
    plt.close('all')
    fig = plt.figure(figsize=(10, 4))
    shap.image_plot(shap_list, pixel_values, show=False)
    fig = plt.gcf()
    
    return fig
