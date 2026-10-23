#!/usr/bin/env python3
"""
Prediction and inference module for multi-organ medical AI.
Forwards to src.pipeline.predict for modularity and backward compatibility.
"""

import sys
import os

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from src.pipeline.predict import (
    load_and_prep_image,
    process_prediction,
    get_xai_explanations,
    validate_image_modality,
    get_loaded_model,
    run_cli_prediction,
)

__all__ = [
    "load_and_prep_image",
    "process_prediction",
    "get_xai_explanations",
    "validate_image_modality",
    "get_loaded_model",
    "run_cli_prediction",
]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Medical Image AI Diagnostic Predictor")
    parser.add_argument("--image", type=str, required=True, help="Path to input medical image scan")
    parser.add_argument("--modality", type=str, choices=["chest", "brain"], default=None, help="Force modality target (chest or brain)")
    args = parser.parse_args()
    run_cli_prediction(args.image, args.modality)
