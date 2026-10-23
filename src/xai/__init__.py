"""
Explainable AI (XAI) Suite
==========================
Tri-modal explainability combining:
1. Grad-CAM: Class activation heatmaps
2. LIME: Local interpretable superpixel boundaries
3. SHAP: Gradient feature attribution
"""

from src.xai.gradcam import make_gradcam_heatmap, overlay_gradcam
from src.xai.lime_explainer import explain_with_lime
from src.xai.shap_explainer import explain_with_shap

__all__ = [
    "make_gradcam_heatmap",
    "overlay_gradcam",
    "explain_with_lime",
    "explain_with_shap",
]
