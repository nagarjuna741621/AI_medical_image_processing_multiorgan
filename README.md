# Medical Imaging Disease Detection AI ⚕️

This project is a high-performance, production-ready AI system designed to detect diseases from medical images (Chest X-Rays and Brain MRIs) with the absolute transparency provided by Explainable AI (XAI).

## Features
- **Multi-Model Architecture**: Features specialized EfficientNetB0 models for multi-class classification of Chest X-Rays and binary classification of Brain MRIs.
- **Explainable AI (XAI)**: Understand why the model makes decisions through three state-of-the-art techniques:
    - **Grad-CAM**: Gradient-weighted class activation heatmaps overlaying pathological focal points.
    - **LIME**: Identifies the local superpixels driving the decision.
    - **SHAP**: Visualizes Shapley feature importances at a sub-segmental level.
- **Production Pipeline**: End-to-end TensorFlow/Keras pipeline utilizing mixed precision readiness, rigorous data augmentation, BatchNormalization, and smart Callbacks (Early Stop, ReduceLR).
- **Professional UI**: A sleek, hospital-style Streamlit web dashboard.

---

## 🛠️ Step-by-Step Setup Instructions

1. **Virtual Environment Setup (Optional but Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Preparation**
   You can either place your dataset images in the created `dataset/` directory according to the standard folder structure:
   ```
   dataset/
     chest/
       NORMAL/
       PNEUMONIA/
       TUBERCULOSIS/
     brain/
       NORMAL/
       TUMOR/
   ```
   *OR* run the mock generator to create dummy structural data for testing the pipeline:
   ```bash
   python generate_mock_data.py
   ```

---

## 🚀 Training Guide

To train the models from scratch, execute `train.py`. The script implements model checkpointing, so your best performing weights are automatically saved to the `models/` directory.

```bash
# Train both models sequentially
python train.py --model all

# Train specifically the chest model
python train.py --model chest

# Train specifically the brain model
python train.py --model brain
```

Training histories (accuracy/loss plots) will be generated and saved beautifully in the `outputs/` directory.

---

## 🧪 Testing Guide

To evaluate the models on your hold-out test set, use `evaluate.py`. This procedure generates essential metrics including Precision, Recall, and the F1-score.

```bash
python evaluate.py --model all
```

This will log the detailed Multi-Class Classification Reports and export a complete graphical evaluation package into the `outputs/` directory for each model:

- Training accuracy/loss curves
- Raw-count and normalized confusion matrices
- Per-class precision, recall, and F1-score bar chart
- One-vs-rest ROC curves with AUC values
- Precision-recall curves with average precision values
- Correct-versus-incorrect confidence histogram and test-set class distribution
- `*_metrics.json` with the numerical report and summary scores
- `*_predictions.csv` with true labels, predicted labels, confidence, and every class probability

The charts are generated automatically whenever `evaluate.py` runs, so they can be inserted directly into a report or presentation.

---

## 🌐 Launching the UI

Fire up the Streamlit interface for live prediction and visualizations:

```bash
streamlit run app.py
```

Upload a test image to see the diagnosis with XAI outputs instantaneously.
