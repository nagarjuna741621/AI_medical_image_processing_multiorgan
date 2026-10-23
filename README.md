<div align="center">

# Multi-Organ Medical AI Diagnostic System ⚕️
### Dual-Modality Deep Learning, Foundation Model Benchmarking & Tri-Modal Explainable AI (XAI)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.24%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Research Paper](https://img.shields.io/badge/Manuscript-IEEE%20TMI-8A2BE2.svg)](paper/research_paper.tex)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*An end-to-end clinical diagnostic intelligence platform for multi-class Chest Radiograph classification (Normal, Pneumonia, Tuberculosis) and binary Brain Magnetic Resonance Imaging (MRI) tumor detection, backed by automated anatomical triage, multi-architecture benchmarked models, and rigorous visual transparency.*

---

</div>

## 📑 Table of Contents
- [System Architecture & Overview](#-system-architecture--overview)
- [Clean Modular Repository Structure](#-clean-modular-repository-structure)
- [Architectural Benchmarks & Results](#-architectural-benchmarks--results)
- [Tri-Modal Explainable AI (XAI) Framework](#-tri-modal-explainable-ai-xai-framework)
- [Installation & Quickstart](#-installation--quickstart)
- [Dataset Preparation](#-dataset-preparation)
- [Workflow & Execution Guide](#-workflow--execution-guide)
  - [1. Modality Auto-Triage & CLI Inference](#1-modality-auto-triage--cli-inference)
  - [2. Model Training](#2-model-training)
  - [3. Comprehensive Diagnostic Evaluation](#3-comprehensive-diagnostic-evaluation)
  - [4. Publication Figures & Executive Dashboard](#4-publication-figures--executive-dashboard)
  - [5. Clinical Streamlit Dashboard](#5-clinical-streamlit-dashboard)
- [Jupyter Research Lab (`notebooks/`)](#-jupyter-research-lab-notebooks)
- [Academic Manuscript & Citation (`paper/`)](#-academic-manuscript--citation-paper)

---

## 🔬 System Architecture & Overview

The platform tackles high-stakes radiological triage through an integrated, four-stage clinical pipeline:

```
                          [ Incoming Medical Scan ]
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │ Modality Triage Router (CNN)  │
                     └───────────────┬───────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
         [ Chest Radiograph (CXR) ]        [ Brain MRI (Axial) ]
                    │                                 │
     ┌──────────────┴──────────────┐   ┌──────────────┴──────────────┐
     │  Multi-Class Classification │   │    Binary Classification    │
     │  - Normal                   │   │  - Normal                   │
     │  - Pneumonia                │   │  - Brain Tumor (Glioma/M.)  │
     │  - Tuberculosis             │   └──────────────┬──────────────┘
     └──────────────┬──────────────┘                  │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │ Tri-Modal Explainable AI (XAI)│
                     │  1. Grad-CAM Pathology Heatmap│
                     │  2. LIME Superpixel Bounds    │
                     │  3. SHAP Feature Attribution  │
                     └───────────────┬───────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │  Hospital Diagnostic Summary  │
                     │ (Confidence, Report, Overlay) │
                     └───────────────────────────────┘
```

1. **Automated Modality Triage**: A lightweight binary router dynamically validates incoming DICOM/JPEG scans to prevent clinical cross-modality misclassification (e.g., passing a brain scan into a pulmonary classifier).
2. **Deep Diagnostic Networks**: Specialized neural backbones trained with cosine-decay learning schedules, focal loss handling, and balanced class distributions.
3. **Multi-Architecture Benchmarking**: Direct head-to-head empirical comparison across **EfficientNetB0**, **DenseNet-121** (CheXNet initializations), **CoAtNet** (hybrid depthwise conv + relative self-attention), and **MobileViT** (low-latency mobile vision transformer).
4. **Tri-Modal Interpretability**: Seamless visual validation providing radiologists with transparent heatmaps before clinical sign-off.

---

## 📂 Clean Modular Repository Structure

The repository is structured into a modular Python package (`src/`) alongside dedicated research artifacts:

```
AI_medical_image_processing_multiorgan/
├── src/                                  # Modular core Python package
│   ├── __init__.py                       # Package initialization & workspace path setup
│   ├── config.py                         # Centralized hyperparameters, paths & class labels
│   ├── logger.py                         # Dual console & timestamped file logger
│   ├── utils.py                          # Publication-grade plotting & metrics engine (300 DPI)
│   ├── data/                             # Dataset acquisition & processing
│   │   ├── __init__.py
│   │   ├── data_loader.py                # tf.data autotuned pipelines (80/10/10 train/val/test)
│   │   ├── preprocess.py                 # Clinical augmentation & normalization
│   │   ├── download_datasets.py          # Automated KaggleHub dataset fetcher
│   │   └── generate_mock_data.py         # Synthetic scan generator for offline testing
│   ├── models/                           # Architecture definitions & weight registry
│   │   ├── __init__.py
│   │   ├── chest_model.py                # EfficientNetB0 3-class pulmonary classifier
│   │   ├── brain_model.py                # EfficientNetB0 binary neuro-oncology classifier
│   │   ├── modality_detector.py          # Anatomical triage router (Chest vs Brain)
│   │   └── load_model.py                 # Custom layer registry (CoAtNet & MobileViT loaders)
│   ├── xai/                              # Explainable AI interpretation suite
│   │   ├── __init__.py
│   │   ├── gradcam.py                    # Dynamic Grad-CAM with automatic conv detection
│   │   ├── lime_explainer.py             # LIME superpixel segmenter
│   │   └── shap_explainer.py             # SHAP gradient & kernel feature attributions
│   └── pipeline/                         # End-to-end operational pipelines
│       ├── __init__.py
│       ├── train.py                      # Multi-model training engine with smart callbacks
│       ├── evaluate.py                   # Rigorous test suite & metric exporter
│       └── predict.py                    # Multi-organ prediction & XAI generation engine
│
├── notebooks/                            # Experimental Google Colab & GPU research notebooks
│   ├── README.md                         # Detailed notebook execution guide
│   ├── 01_Pipeline_Hybrid_CoAtNet.ipynb
│   ├── 02_Pipeline_Hybrid_MobileViT.ipynb
│   ├── 03_Pipeline_Foundation_DenseNet121_CheXNet.ipynb
│   ├── 04_Pipeline_Foundation_BiomedCLIP_MedCLIP.ipynb
│   ├── 05_Pipeline_Foundation_DINOv2_MAE.ipynb
│   ├── 06_Pipeline_ModernCNN_ConvNeXt.ipynb
│   └── AI_Medical_Image_Processing_Pipeline.ipynb
│
├── paper/                                # Academic dissemination
│   ├── README.md                         # Overleaf & pdflatex compilation instructions
│   └── research_paper.tex                # Complete IEEE TMI LaTeX manuscript
│
├── scripts/                              # Standalone publication tools
│   ├── generate_graphical_results.py     # Publication figures generator (Figures 1-9)
│   └── generate_all_models_metrics_dashboard.py # Multi-model comparative dashboard
│
├── models/                               # Serialized model weights (.keras)
│   ├── chest_model.keras                 # EfficientNetB0 Chest Model
│   ├── brain_model.keras                 # EfficientNetB0 Brain Model
│   ├── modality_model.keras              # Anatomical Triage Router
│   ├── coatnet_chest_model.keras         # CoAtNet Chest Model
│   ├── coatnet_brain_model.keras         # CoAtNet Brain Model
│   ├── densenet121_chest_model.keras     # DenseNet-121 Chest Model
│   ├── densenet121_brain_model.keras     # DenseNet-121 Brain Model
│   ├── mobilevit_chest_model.keras       # MobileViT Chest Model
│   └── mobilevit_brain_model.keras       # MobileViT Brain Model
│
├── outputs/                              # Benchmark figures, metrics & test predictions
│   ├── Executive_Summary_Results_Dashboard.png
│   ├── all_models_metrics_master_dashboard.png
│   ├── 08_modality_detector_evaluation.png
│   └── ... (30+ publication charts & JSON reports)
│
├── app.py                                # Hospital-themed Streamlit interactive web application
├── train.py                              # Root CLI entry point (forwards to src.pipeline.train)
├── evaluate.py                           # Root CLI entry point (forwards to src.pipeline.evaluate)
├── predict.py                            # Root CLI entry point (forwards to src.pipeline.predict)
├── config.py                             # Root backwards-compatible configuration shim
├── utils.py                              # Root backwards-compatible utility shim
├── requirements.txt                      # Python dependencies specification
├── environment.yml                       # Conda environment specification (Python 3.10)
├── Dockerfile                            # Production container configuration
└── .gitignore                            # Comprehensive clean repository ignore rules
```

> **Backward Compatibility Guarantee**: All root-level commands (`python train.py`, `python evaluate.py`, `streamlit run app.py`, and legacy imports like `from config import ...`) continue to execute seamlessly via zero-overhead forwarding shims.

---

## 📊 Architectural Benchmarks & Results

Extensive empirical evaluations were conducted on hold-out clinical test partitions:

### 1. Chest Radiograph Multi-Class Benchmark (Normal vs. Pneumonia vs. Tuberculosis)

| Architecture | Paradigm | Params | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | ROC-AUC | Inference (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DenseNet-121** | Dense Feature Reuse | 7.0M | **98.24%** | **0.981** | **0.983** | **0.982** | **0.998** | 38.4 |
| **CoAtNet** | Hybrid Conv-Transformer | 2.9M | 96.88% | 0.969 | 0.967 | 0.968 | 0.994 | 44.1 |
| **MobileViT** | Edge Vision Transformer | 2.7M | 95.12% | 0.952 | 0.950 | 0.951 | 0.989 | **22.6** |
| **EfficientNetB0** | Compound Scaled CNN | 4.1M | 96.47% | 0.963 | 0.965 | 0.964 | 0.991 | 29.8 |

### 2. Brain MRI Binary Benchmark (Normal vs. Glioblastoma/Tumor)

| Architecture | Paradigm | Params | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DenseNet-121** | Dense Feature Reuse | 7.0M | **98.00%** | **0.980** | **0.980** | **0.980** | **0.980** | **0.997** |
| **CoAtNet** | Hybrid Conv-Transformer | 2.9M | 96.00% | 0.962 | 0.960 | 0.960 | 0.961 | 0.992 |
| **MobileViT** | Edge Vision Transformer | 2.7M | 94.00% | 0.941 | 0.940 | 0.940 | 0.940 | 0.984 |
| **EfficientNetB0** | Compound Scaled CNN | 4.1M | 96.00% | 0.960 | 0.960 | 0.960 | 0.960 | 0.989 |

---

## 🔍 Tri-Modal Explainable AI (XAI) Framework

To eliminate deep learning "black box" risks in critical clinical settings, this platform generates three complementary visual explanations for every scan:

1. **Grad-CAM (Gradient-Weighted Class Activation Mapping)**:
   - Identifies high-gradient spatial feature maps in the final convolutional layers.
   - Highlights pulmonary consolidations in pneumonia, apical fibro-cavitary opacities in tuberculosis, and localized hyper-intense mass lesions in brain MRIs.
2. **LIME (Local Interpretable Model-agnostic Explanations)**:
   - Partitions scans into contiguous superpixels using the Quickshift algorithm.
   - Computes local perturbations to reveal the precise anatomical boundaries driving positive prediction confidence.
3. **SHAP (SHapley Additive exPlanations)**:
   - Uses game-theoretic Shapley values to assign additive importance scores to sub-segmental features, confirming absence of spurious background artifacts.

---

## 🚀 Installation & Quickstart

### Prerequisites
- Python 3.10+ (Recommended)
- CUDA-enabled GPU (Optional, recommended for training; CPU inference is supported)

### Option A: Conda Environment (Recommended)
```bash
# Clone the repository
git clone https://github.com/nagarjuna741621/AI_medical_image_processing_multiorgan.git
cd AI_medical_image_processing_multiorgan

# Create environment from specification
conda env create -f environment.yml
conda activate medical_ai_env
```

### Option B: Standard Python Virtualenv
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Option C: Docker Container
```bash
docker build -t medical-ai-platform .
docker run -p 8501:8501 medical-ai-platform
```

---

## 📦 Dataset Preparation

The expected directory structure under `dataset/` is:

```
dataset/
├── chest/
│   ├── NORMAL/
│   ├── PNEUMONIA/
│   └── TUBERCULOSIS/
└── brain/
    ├── NORMAL/
    └── TUMOR/
```

### Automatic KaggleHub Dataset Acquisition
Download real-world Kaggle medical benchmark datasets automatically:
```bash
python -c "from src.data.download_datasets import download_and_structure_all; download_and_structure_all()"
```

### Synthetic Mock Dataset (Immediate Offline Testing)
If working offline or performing a smoke test, generate synthetic scan structures:
```bash
python -c "from src.data.generate_mock_data import ensure_mock_data; ensure_mock_data()"
```

---

## 💻 Workflow & Execution Guide

### 1. Modality Auto-Triage & CLI Inference
Run automated triage and diagnostic inference on any single scan:

```bash
# Perform inference with automatic organ detection and Grad-CAM generation
python predict.py --image path/to/scan.jpg

# Force specific modality evaluation
python predict.py --image path/to/scan.jpg --modality chest
```

### 2. Model Training
Train networks from scratch with automatic checkpointing, early stopping, and learning-rate reduction:

```bash
# Train both Chest and Brain models sequentially
python train.py --model all

# Train specifically the chest radiograph model
python train.py --model chest

# Train specifically the brain MRI model
python train.py --model brain
```

### 3. Comprehensive Diagnostic Evaluation
Run complete validation against hold-out test sets to export full metric suites:

```bash
python evaluate.py --model all
```

**Artifacts Generated in `outputs/`:**
- 300-DPI Confusion Matrices (Raw and Normalized)
- Multi-Class One-vs-Rest ROC Curves with AUC Scores
- Precision-Recall Curves with Average Precision (AP)
- Per-Class Precision, Recall, and F1-Score Bar Charts
- JSON metric summaries (`*_metrics.json`) and per-scan CSV predictions (`*_predictions.csv`)

### 4. Publication Figures & Executive Dashboard
Generate the complete suite of figures matching the IEEE TMI manuscript:

```bash
# Master publication visualizer (Figures 1-9)
python scripts/generate_graphical_results.py

# Multi-model comparative metrics dashboard
python scripts/generate_all_models_metrics_dashboard.py
```

### 5. Clinical Streamlit Dashboard
Launch the interactive hospital-grade graphical interface:

```bash
streamlit run app.py
```

Features included in the web app:
- Real-time scan upload (PNG, JPG, JPEG)
- Automatic anatomical triage validation
- Confidence percentage breakdown with interactive pie visualizer
- Side-by-side comparative Explainable AI panel (Original Scan, Grad-CAM, LIME)
- Instantaneous triage decision recommendations for clinical decision support

---

## 📓 Jupyter Research Lab (`notebooks/`)

Dedicated standalone pipelines configured for one-click execution in Google Colab (with free GPU acceleration) or local workstations:

| Notebook | Focus Architecture | Modality | Key Concepts |
| :--- | :--- | :--- | :--- |
| [`01_Pipeline_Hybrid_CoAtNet.ipynb`](notebooks/01_Pipeline_Hybrid_CoAtNet.ipynb) | CoAtNet | Chest & Brain | Hybrid depthwise conv + Relative Multi-Head Attention |
| [`02_Pipeline_Hybrid_MobileViT.ipynb`](notebooks/02_Pipeline_Hybrid_MobileViT.ipynb) | MobileViT | Chest & Brain | Lightweight mobile transformer blocks for point-of-care edge devices |
| [`03_Pipeline_Foundation_DenseNet121_CheXNet.ipynb`](notebooks/03_Pipeline_Foundation_DenseNet121_CheXNet.ipynb) | DenseNet-121 | Chest & Brain | CheXNet pretraining with dense connectivity and feature reuse |
| [`04_Pipeline_Foundation_BiomedCLIP_MedCLIP.ipynb`](notebooks/04_Pipeline_Foundation_BiomedCLIP_MedCLIP.ipynb) | BiomedCLIP / MedCLIP | Multimodal | Vision-language foundation embeddings & zero-shot pathology prompts |
| [`05_Pipeline_Foundation_DINOv2_MAE.ipynb`](notebooks/05_Pipeline_Foundation_DINOv2_MAE.ipynb) | DINOv2 & MAE | Multi-Organ | Self-supervised visual representation pretraining |
| [`06_Pipeline_ModernCNN_ConvNeXt.ipynb`](notebooks/06_Pipeline_ModernCNN_ConvNeXt.ipynb) | ConvNeXt | Chest & Brain | 7x7 depthwise convolutions, inverted bottlenecks, and modern CNN design |
| [`AI_Medical_Image_Processing_Pipeline.ipynb`](notebooks/AI_Medical_Image_Processing_Pipeline.ipynb) | Full Pipeline | Multi-Organ | End-to-end dataset visualizer, triage routing, and XAI suite |

*See [`notebooks/README.md`](notebooks/README.md) for step-by-step Colab setup and GPU memory growth settings.*

---

## 📄 Academic Manuscript & Citation (`paper/`)

The full LaTeX research paper prepared for submission to *IEEE Transactions on Medical Imaging* is available in the [`paper/`](paper/) directory.

To compile the paper locally:
```bash
cd paper
pdflatex research_paper.tex
bibtex research_paper
pdflatex research_paper.tex
pdflatex research_paper.tex
```

### BibTeX Citation
```bibtex
@article{medical_ai_multiorgan_2026,
  title   = {Multi-Organ Deep Learning Diagnosis and Tri-Modal Explainable Artificial Intelligence for Radiographic and Neuroimaging Triage},
  author  = {Medical AI Research Collaborative},
  journal = {IEEE Transactions on Medical Imaging (Preprint)},
  year    = {2026},
  url     = {https://github.com/nagarjuna741621/AI_medical_image_processing_multiorgan}
}
```

---

## 🛡️ License & Ethics Notice

This project is licensed under the [MIT License](LICENSE). 

**Clinical Disclaimer**: This software is intended strictly for research, educational, and benchmarking purposes. It is not approved as a primary diagnostic device under FDA 510(k) or CE mark regulations. All AI-generated diagnostic suggestions should be reviewed by a certified radiologist or medical professional.
