import matplotlib.pyplot as plt
import os
import numpy as np

def plot_training_history(history, model_name, output_dir):
    """Plots training and validation metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='train_acc')
    ax1.plot(history.history['val_accuracy'], label='val_acc')
    ax1.set_title(f'{model_name} Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    
    # Loss plot
    ax2.plot(history.history['loss'], label='train_loss')
    ax2.plot(history.history['val_loss'], label='val_loss')
    ax2.set_title(f'{model_name} Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{model_name}_history.png')
    plt.savefig(plot_path)
    plt.close()

def plot_confusion_matrix(cm, classes, model_name, output_dir, title='Confusion matrix', cmap=plt.cm.Blues):
    """Plots the confusion matrix."""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(f"{model_name} - {title}")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{model_name}_confusion_matrix.png')
    plt.savefig(plot_path)
    plt.close()
