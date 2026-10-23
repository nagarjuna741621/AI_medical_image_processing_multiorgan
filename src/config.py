import os
import sys

# Base project paths (dynamically resolved to repository root)
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

# Ensure required project directories exist
for directory in [MODELS_DIR, LOGS_DIR, OUTPUTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Dataset folder paths
CHEST_DIR = os.path.join(DATASET_DIR, "chest")
BRAIN_DIR = os.path.join(DATASET_DIR, "brain")

# Class label taxonomies
CHEST_CLASSES = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]
BRAIN_CLASSES = ["NORMAL", "TUMOR"]

# Model hyperparameters and input tensors
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4

# Model artifact paths
DENSENET_CHEST_MODEL_PATH = os.path.join(MODELS_DIR, "densenet121_chest_model.keras")
DENSENET_BRAIN_MODEL_PATH = os.path.join(MODELS_DIR, "densenet121_brain_model.keras")
COATNET_CHEST_MODEL_PATH = os.path.join(MODELS_DIR, "coatnet_chest_model.keras")
COATNET_BRAIN_MODEL_PATH = os.path.join(MODELS_DIR, "coatnet_brain_model.keras")
MOBILEVIT_CHEST_MODEL_PATH = os.path.join(MODELS_DIR, "mobilevit_chest_model.keras")
MOBILEVIT_BRAIN_MODEL_PATH = os.path.join(MODELS_DIR, "mobilevit_brain_model.keras")
ORIGINAL_CHEST_MODEL_PATH = os.path.join(MODELS_DIR, "chest_model.keras")
ORIGINAL_BRAIN_MODEL_PATH = os.path.join(MODELS_DIR, "brain_model.keras")
MODALITY_MODEL_PATH = os.path.join(MODELS_DIR, "modality_model.keras")

# Select best performing model if available, fallback to baseline EfficientNet
CHEST_MODEL_PATH = DENSENET_CHEST_MODEL_PATH if os.path.exists(DENSENET_CHEST_MODEL_PATH) else ORIGINAL_CHEST_MODEL_PATH
BRAIN_MODEL_PATH = DENSENET_BRAIN_MODEL_PATH if os.path.exists(DENSENET_BRAIN_MODEL_PATH) else ORIGINAL_BRAIN_MODEL_PATH

CHEST_NUM_CLASSES = len(CHEST_CLASSES)
BRAIN_NUM_CLASSES = len(BRAIN_CLASSES)
