import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import tensorflow as tf

from src.config import (
    BRAIN_CLASSES,
    BRAIN_DIR,
    BRAIN_MODEL_PATH,
    CHEST_CLASSES,
    CHEST_DIR,
    CHEST_MODEL_PATH,
    DATASET_DIR,
    OUTPUTS_DIR,
)
from src.data.data_loader import get_datasets
from src.models.load_model import load_medical_model
from src.logger import get_logger
from src.utils import (
    plot_confusion_matrix,
    plot_dataset_pie_charts,
    plot_dataset_splits_breakdown,
    plot_evaluation_results,
    plot_executive_summary_dashboard,
    plot_model_comparison_radar,
    plot_multimodel_benchmark_bar,
)

log = get_logger("evaluate")


def evaluate_model(model_type='chest'):
    """Evaluates a single model type and exports comprehensive validation metrics."""
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
        return None

    log.info(f"Loading {name} from {model_path}")
    model = load_medical_model(model_path)

    _, _, test_ds = get_datasets(dataset_dir, classes, label_mode=label_mode)

    log.info(f"Evaluating {name} on test dataset")
    results = model.evaluate(test_ds, return_dict=True, verbose=1)
    metrics_str = ", ".join(
        f"{metric_name.capitalize()}: {metric_value:.4f}"
        for metric_name, metric_value in results.items()
    )
    log.info(metrics_str)

    # Generate predictions for confusion matrix and metrics
    y_pred = []
    y_true = []
    y_probabilities = []

    for x_batch, y_batch in test_ds:
        preds = model.predict(x_batch, verbose=0)

        if model_type == 'chest':
            pred_classes = np.argmax(preds, axis=1)
            true_classes = np.argmax(y_batch.numpy(), axis=1)
        else:
            pred_classes = (preds > 0.5).astype(int).flatten()
            true_classes = y_batch.numpy().astype(int).flatten()

        y_pred.extend(pred_classes)
        y_true.extend(true_classes)
        if model_type == 'chest':
            y_probabilities.extend(preds)
        else:
            positive_probability = preds.astype(float).flatten()
            y_probabilities.extend(np.column_stack([1.0 - positive_probability, positive_probability]))

    y_pred = np.array(y_pred, dtype=int)
    y_true = np.array(y_true, dtype=int)

    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    log.info(f"F1-Score: {f1:.4f}")

    report = classification_report(
        y_true, y_pred, target_names=classes, zero_division=0
    )
    log.info(f"Classification Report:\n{report}")

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    cmap = 'Blues' if model_type == 'chest' else 'Purples'
    plot_confusion_matrix(cm, classes=classes, model_name=name, output_dir=OUTPUTS_DIR, cmap=cmap)
    plot_confusion_matrix(
        cm / np.maximum(cm.sum(axis=1, keepdims=True), 1),
        classes=classes,
        model_name=name,
        output_dir=OUTPUTS_DIR,
        title='Normalized Confusion Matrix',
        cmap=cmap,
        filename_suffix='_normalized',
    )
    metrics_dict = plot_evaluation_results(
        y_true, y_pred, np.asarray(y_probabilities), classes, name, OUTPUTS_DIR
    )
    log.info("Evaluation charts, metrics JSON, and predictions CSV saved to outputs folder.")
    return metrics_dict


def run_full_evaluation():
    """Runs end-to-end evaluation for all organ modalities and compiles comparison figures."""
    plot_dataset_pie_charts(DATASET_DIR, OUTPUTS_DIR)
    plot_dataset_splits_breakdown(OUTPUTS_DIR)
    
    chest_metrics = evaluate_model('chest')
    brain_metrics = evaluate_model('brain')
    
    if chest_metrics and brain_metrics:
        log.info("Generating multi-model comparison charts...")
        plot_model_comparison_radar(chest_metrics, brain_metrics, OUTPUTS_DIR)
        plot_multimodel_benchmark_bar(chest_metrics, brain_metrics, OUTPUTS_DIR)
        plot_executive_summary_dashboard(chest_metrics, brain_metrics, OUTPUTS_DIR)
        log.info("Executive summary dashboard and comparison charts saved.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Medical Imaging Classifiers")
    parser.add_argument('--model', type=str, choices=['chest', 'brain', 'all'], default='all')
    args = parser.parse_args()

    if args.model == 'all':
        run_full_evaluation()
    elif args.model == 'chest':
        evaluate_model('chest')
    elif args.model == 'brain':
        evaluate_model('brain')


if __name__ == "__main__":
    main()
