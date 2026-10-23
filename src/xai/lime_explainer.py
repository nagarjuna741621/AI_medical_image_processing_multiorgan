import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries

def explain_with_lime(model, image, num_classes=2, top_labels=1, hide_color=0, num_samples=100):
    """
    Generate LIME explanation for image classification showing both supporting
    and opposing superpixel regions.
    
    Args:
        model: Trained Keras model
        image: Original image as numpy array in [0, 255]
        num_classes: Number of output classes
        top_labels: Number of top labels to explain
        hide_color: Color to replace perturbed superpixels with
        num_samples: Number of perturbation samples (100 balances precision and CPU latency)
        
    Returns:
        img_boundary: Image with superpixel boundaries marked
    """
    explainer = lime_image.LimeImageExplainer()
    
    # Efficient batched prediction wrapper
    def predict_fn(images):
        preds = model(images, training=False).numpy()
        # For binary sigmoid output (shape (N, 1)), expand to 2-class probabilities (N, 2)
        if preds.ndim == 2 and preds.shape[1] == 1:
            return np.hstack([1.0 - preds, preds])
        return preds

    explanation = explainer.explain_instance(
        image.astype('double'), 
        predict_fn, 
        top_labels=top_labels, 
        hide_color=hide_color, 
        num_samples=num_samples,
        random_seed=42
    )
    
    # Get image and mask showing positive (supporting) and negative (opposing) features
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], 
        positive_only=False, 
        num_features=8, 
        hide_rest=False
    )
    
    img_boundary = mark_boundaries(temp / 255.0, mask)
    return img_boundary
