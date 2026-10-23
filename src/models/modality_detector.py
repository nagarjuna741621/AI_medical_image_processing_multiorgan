import os
import tensorflow as tf
from src.config import DATASET_DIR, MODELS_DIR

def train_modality_classifier():
    """
    Trains the lightweight triage router (Chest X-Ray vs Brain MRI).
    Uses MobileNetV2 at 128x128 for rapid and accurate routing.
    """
    print("Training Modality Classifier (Chest vs Brain)...")
    IMAGE_SIZE = (128, 128)
    
    dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        labels='inferred',
        label_mode='binary',
        class_names=['brain', 'chest'],  # 0 = brain, 1 = chest
        color_mode='rgb',
        batch_size=32,
        image_size=IMAGE_SIZE,
        shuffle=True
    )
    
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=(128, 128, 3))
    x = tf.keras.layers.Rescaling(1./127.5, offset=-1)(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs, name="Modality_Triage_Classifier")
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
                  
    print("Fitting model...")
    model.fit(dataset, epochs=2)
    
    model_path = os.path.join(MODELS_DIR, 'modality_model.keras')
    model.save(model_path)
    print(f"Modality classifier saved to {model_path}")
    return model

if __name__ == '__main__':
    train_modality_classifier()
