import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
from preprocess import get_data_augmentation
from config import INPUT_SHAPE

def build_brain_model(input_shape=INPUT_SHAPE, freeze_base=True):
    """
    Builds the Brain MRI classification model using EfficientNetB0.
    Binary classification (Tumor, Normal).
    """
    # Base model
    base_model = EfficientNetB0(
        weights='imagenet', 
        include_top=False, 
        input_shape=input_shape
    )
    
    if freeze_base:
        base_model.trainable = False

    # Input layer
    inputs = tf.keras.Input(shape=input_shape, name="brain_input")
    
    # Data Augmentation layer
    x = get_data_augmentation()(inputs)
    
    # Pass through base model
    x = base_model(x, training=not freeze_base)
    
    # Global Average Pooling
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    
    # Batch Normalization
    x = layers.BatchNormalization(name="batch_norm_1")(x)
    
    # Dropout for regularization
    x = layers.Dropout(0.4, name="dropout_1")(x)
    
    # Output layer (Sigmoid for binary classification)
    # Using 1 unit with sigmoid for binary crossentropy
    outputs = layers.Dense(1, activation='sigmoid', name="brain_output")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="Brain_EfficientNetB0")
    
    # Compile model 
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    
    return model

if __name__ == "__main__":
    model = build_brain_model()
    model.summary()
