import shap
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def explain_with_shap(model, image, background_images=None, output_path=None):
    """
    Generate SHAP explanation.
    
    Args:
        model: Trained Keras model
        image: Single image array to explain (shape MUST match model input)
        background_images: Images to use as background for GradientExplainer.
                           If None, we will use black images.
    """
    if background_images is None:
        # Create dummy background if not provided
        background_images = np.zeros((10, *image.shape))
        
    explainer = shap.GradientExplainer(model, background_images)
    
    # We must pass wrapped image in batch dimension
    img_batch = np.expand_dims(image, axis=0)
    
    # explainer.shap_values returns a list/tuple for each class or ranked outputs
    res = explainer.shap_values(img_batch, ranked_outputs=1)
    if isinstance(res, tuple) and len(res) == 2:
        shap_values, indexes = res
    else:
        shap_values = res
    
    # Depending on SHAP version, shap_values might be a list of arrays or array
    if isinstance(shap_values, list):
        # We take the top class
        shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
        test_numpy = np.swapaxes(np.swapaxes(img_batch, 1, -1), 1, 2)
    else:
        shap_numpy = shap_values
        test_numpy = img_batch
        
    # We can plot it using shap.image_plot
    shap.image_plot(shap_numpy, -test_numpy, show=False)
    
    # Get current figure and return it so Streamlit can render it
    fig = plt.gcf()
    
    return fig
