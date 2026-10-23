import tensorflow as tf

def get_data_augmentation():
    """
    Returns a Keras Sequential model applying clinical data augmentation.
    Preserves pathological invariants (flips, slight rotations, zoom, contrast).
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
    EfficientNetB0 has a built-in rescaling layer expecting inputs in [0, 255].
    """
    pass
