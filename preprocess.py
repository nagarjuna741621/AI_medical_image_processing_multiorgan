import tensorflow as tf

def get_data_augmentation():
    """
    Returns a Keras Sequential model applying data augmentation.
    This will be incorporated directly into the model architecture.
    """
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ], name="data_augmentation")
    
    return data_augmentation

def preprocess_input():
    """
    Returns the specific preprocessing function needed for EfficientNetB0.
    EfficientNetB0 in tf.keras expects inputs in [0, 255] directly as it has a built-in rescaling layer.
    Thus, no external rescaling is needed for the model itself, 
    but we might need it depending on the framework version.
    Current best practice for EfficientNetB0 in TensorFlow 2.x is to pass the raw image directly.
    """
    pass # Managed internally by EfficientNetB0 applications
