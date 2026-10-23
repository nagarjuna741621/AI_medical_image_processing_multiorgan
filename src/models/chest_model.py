import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
from src.data.preprocess import get_data_augmentation
from src.config import INPUT_SHAPE, CHEST_NUM_CLASSES

def build_chest_model(input_shape=INPUT_SHAPE, num_classes=CHEST_NUM_CLASSES, freeze_base=True):
    """
    Builds the Chest X-ray classification model using EfficientNetB0.
    Multi-class classification (Normal, Pneumonia, Tuberculosis).
    """
    base_model = EfficientNetB0(
        weights='imagenet', 
        include_top=False, 
        input_shape=input_shape
    )
    
    if freeze_base:
        base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape, name="chest_input")
    x = get_data_augmentation()(inputs)
    x = base_model(x, training=not freeze_base)
    
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.BatchNormalization(name="batch_norm_1")(x)
    x = layers.Dropout(0.4, name="dropout_1")(x)
    
    x = layers.Dense(128, activation='relu', name="dense_1")(x)
    x = layers.BatchNormalization(name="batch_norm_2")(x)
    x = layers.Dropout(0.3, name="dropout_2")(x)
    
    outputs = layers.Dense(num_classes, activation='softmax', name="chest_output")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="Chest_EfficientNetB0")
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    
    return model

if __name__ == "__main__":
    model = build_chest_model()
    model.summary()
