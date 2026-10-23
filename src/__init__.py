"""
AI Medical Image Processing (Multi-Organ) Source Package
=========================================================
Modular architecture for Chest X-Ray and Brain MRI disease classification
with integrated triage routing and tri-modal Explainable AI (XAI).
"""

import os
import sys

# Ensure repository root is on sys.path for seamless relative and package imports
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_PACKAGE_DIR, ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

__version__ = "2.0.0"
