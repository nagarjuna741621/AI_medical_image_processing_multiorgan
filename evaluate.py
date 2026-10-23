#!/usr/bin/env python3
"""
CLI entry point for evaluating medical image models.
Delegates to the modular evaluation pipeline in src.pipeline.evaluate.
"""

import sys
import os

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from src.pipeline.evaluate import main

if __name__ == "__main__":
    main()