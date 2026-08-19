import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')

# Ensure directories exist
for directory in [MODELS_DIR, LOGS_DIR, OUTPUTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Dataset specifics
CHEST_DIR = os.path.join(DATASET_DIR, 'chest')
BRAIN_DIR = os.path.join(DATASET_DIR, 'brain')

CHEST_CLASSES = ['NORMAL', 'PNEUMONIA', 'TUBERCULOSIS']
BRAIN_CLASSES = ['NORMAL', 'TUMOR']

# Model configurations
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
EPOCHS = 20  # You can increase this for full training
LEARNING_RATE = 1e-4

CHEST_MODEL_PATH = os.path.join(MODELS_DIR, 'chest_model.keras')
BRAIN_MODEL_PATH = os.path.join(MODELS_DIR, 'brain_model.keras')

# Setup constraints / types
CHEST_NUM_CLASSES = len(CHEST_CLASSES)
BRAIN_NUM_CLASSES = len(BRAIN_CLASSES)
