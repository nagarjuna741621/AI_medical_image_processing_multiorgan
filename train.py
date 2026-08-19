import tensorflow as tf
import os
import argparse
from config import CHEST_DIR, BRAIN_DIR, CHEST_CLASSES, BRAIN_CLASSES, EPOCHS, CHEST_MODEL_PATH, BRAIN_MODEL_PATH, OUTPUTS_DIR
from data_loader import get_datasets
from chest_model import build_chest_model
from brain_model import build_brain_model
from utils import plot_training_history
from logger import get_logger

log = get_logger("train")

def get_callbacks(model_path):
    """Returns a list of callbacks for training."""
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    return [checkpoint, early_stop, reduce_lr]

def train_model(model_type='chest'):
    if model_type == 'chest':
        log.info("Starting training for Chest Model")
        train_ds, val_ds, test_ds = get_datasets(CHEST_DIR, CHEST_CLASSES, label_mode='categorical')
        model = build_chest_model()
        model_path = CHEST_MODEL_PATH
        name = "Chest_Model"
    elif model_type == 'brain':
        log.info("Starting training for Brain Model")
        train_ds, val_ds, test_ds = get_datasets(BRAIN_DIR, BRAIN_CLASSES, label_mode='binary')
        model = build_brain_model()
        model_path = BRAIN_MODEL_PATH
        name = "Brain_Model"
    else:
        raise ValueError("Invalid model type. Choose 'chest' or 'brain'.")
        
    callbacks = get_callbacks(model_path)
    
    log.info(f"Training parameters setup. Epochs: {EPOCHS}")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    plot_training_history(history, name, OUTPUTS_DIR)
    log.info(f"Training completed for {name}. Best model saved to {model_path}.")
    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Medical Imaging Classifiers")
    parser.add_argument('--model', type=str, choices=['chest', 'brain', 'all'], default='all', help="Which model to train")
    args = parser.parse_args()
    
    if args.model in ['chest', 'all']:
        try:
            train_model('chest')
        except Exception as e:
            log.error(f"Error training chest model: {e}")
            
    if args.model in ['brain', 'all']:
        try:
            train_model('brain')
        except Exception as e:
            log.error(f"Error training brain model: {e}")
