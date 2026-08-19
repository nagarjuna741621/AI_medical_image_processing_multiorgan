import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import argparse
import os

from config import CHEST_DIR, BRAIN_DIR, CHEST_CLASSES, BRAIN_CLASSES, CHEST_MODEL_PATH, BRAIN_MODEL_PATH, OUTPUTS_DIR
from data_loader import get_datasets
from utils import plot_confusion_matrix
from logger import get_logger

log = get_logger("evaluate")

def evaluate_model(model_type='chest'):
    if model_type == 'chest':
        dataset_dir = CHEST_DIR
        classes = CHEST_CLASSES
        model_path = CHEST_MODEL_PATH
        label_mode = 'categorical'
        name = "Chest_Model"
    else:
        dataset_dir = BRAIN_DIR
        classes = BRAIN_CLASSES
        model_path = BRAIN_MODEL_PATH
        label_mode = 'binary'
        name = "Brain_Model"

    if not os.path.exists(model_path):
        log.error(f"Model path {model_path} not found. Please train the model first.")
        return

    log.info(f"Loading {name} from {model_path}")
    model = tf.keras.models.load_model(model_path)

    _, _, test_ds = get_datasets(dataset_dir, classes, label_mode=label_mode)
    
    log.info(f"Evaluating {name} on test dataset")
    results = model.evaluate(test_ds)
    log.info(f"Loss: {results[0]:.4f}, Accuracy: {results[1]:.4f}, Precision: {results[2]:.4f}, Recall: {results[3]:.4f}")

    # Generate predictions for confusion matrix and F1-score
    y_pred = []
    y_true = []
    
    # We must predict batch by batch to get aligned true labels
    for x_batch, y_batch in test_ds:
        preds = model.predict(x_batch, verbose=0)
        
        if model_type == 'chest':
            pred_classes = np.argmax(preds, axis=1)
            true_classes = np.argmax(y_batch.numpy(), axis=1)
        else:
            pred_classes = (preds > 0.5).astype(int).flatten()
            true_classes = y_batch.numpy().flatten()
            
        y_pred.extend(pred_classes)
        y_true.extend(true_classes)
        
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # Compute additional metrics
    f1 = f1_score(y_true, y_pred, average='weighted')
    log.info(f"F1-Score: {f1:.4f}")

    log.info(f"Classification Report:\n{classification_report(y_true, y_pred, target_names=classes)}")

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, classes=classes, model_name=name, output_dir=OUTPUTS_DIR)
    log.info("Confusion matrix plotted and saved to outputs folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Medical Imaging Classifiers")
    parser.add_argument('--model', type=str, choices=['chest', 'brain', 'all'], default='all')
    args = parser.parse_args()
    
    if args.model in ['chest', 'all']:
        evaluate_model('chest')
    if args.model in ['brain', 'all']:
        evaluate_model('brain')
