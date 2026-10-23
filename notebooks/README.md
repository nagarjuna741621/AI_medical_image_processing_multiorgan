# Medical AI Research & Training Notebooks 🔬

This directory houses the standalone Jupyter notebooks for experimental model training, foundation model exploration, and multi-architecture evaluation across Google Colab and local GPU workstations.

## Available Pipelines

| Notebook | Focus Architecture | Target Modality | Key Highlights |
| :--- | :--- | :--- | :--- |
| **`01_Pipeline_Hybrid_CoAtNet.ipynb`** | CoAtNet (Hybrid CNN + Self-Attention) | Chest X-Ray & Brain MRI | Relative attention, MBConv blocks, automated artifact export |
| **`02_Pipeline_Hybrid_MobileViT.ipynb`** | MobileViT (Mobile Vision Transformer) | Chest X-Ray & Brain MRI | Edge-optimized lightweight transformer blocks, low latency |
| **`03_Pipeline_Foundation_DenseNet121_CheXNet.ipynb`** | DenseNet-121 (CheXNet weights) | Chest X-Ray & Brain MRI | Feature reuse via dense connectivity; highest diagnostic AUC |
| **`04_Pipeline_Foundation_BiomedCLIP_MedCLIP.ipynb`** | BiomedCLIP & MedCLIP | Multimodal / Multi-Organ | Vision-language foundation embeddings & zero-shot diagnostic prompts |
| **`05_Pipeline_Foundation_DINOv2_MAE.ipynb`** | DINOv2 & Masked Autoencoder (MAE) | Multi-Organ Medical | Self-supervised visual representation pretraining |
| **`06_Pipeline_ModernCNN_ConvNeXt.ipynb`** | ConvNeXt (Modernized Pure CNN) | Chest X-Ray & Brain MRI | 7x7 depthwise convolutions, inverted bottleneck, LayerNorm |
| **`AI_Medical_Image_Processing_Pipeline.ipynb`** | Full Multi-Organ Pipeline | Chest & Brain | Complete end-to-end exploration and dataset visualization |

## Execution Guidelines
- **Google Colab**: Each notebook includes auto-detection for Google Colab environments with automatic dependency installation and GPU hardware check (`torch.cuda.is_available()` / `tf.config.list_physical_devices('GPU')`).
- **Local Workstations**: Ensure your virtual environment is active (`source venv/bin/activate` or `conda activate medical_ai_env`).
