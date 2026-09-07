import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

def explain_with_lime(model, image, num_classes=2, top_labels=1, hide_color=0, num_samples=250):
    """
    Generate LIME explanation for an image classification.
    
    Args:
        model: Trained Keras model
        image: Original image as numpy array (normalized or not depending on model,
               but EfficientNet expects unnormalized if using standard applications)
        num_classes: Number of output classes (determines random state setup)
        
    Returns:
        image, mask for visualizing the LIME boundaries
    """
    explainer = lime_image.LimeImageExplainer()
    
    # Needs to process batched inputs so we wrap predicting function
    def predict_fn(images):
        preds = model.predict(images, verbose=0)
        # For binary sigmoid output (shape (N, 1)), expand to 2-class probabilities (N, 2)
        if preds.ndim == 2 and preds.shape[1] == 1:
            return np.hstack([1.0 - preds, preds])
        return preds

    explanation = explainer.explain_instance(
        image.astype('double'), 
        predict_fn, 
        top_labels=top_labels, 
        hide_color=hide_color, 
        num_samples=num_samples
    )
    
    # Get image and mask for the top predicted label
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], 
        positive_only=True, 
        num_features=5, 
        hide_rest=False
    )
    
    # Return boundaries applied to temp
    img_boundry = mark_boundaries(temp / 255.0, mask)
    return img_boundry
