import tensorflow as tf
from config import IMAGE_SIZE, BATCH_SIZE

def get_datasets(data_dir, class_names, label_mode='categorical'):
    """
    Creates train, validation, and test datasets.
    Assuming the data is in data_dir/<class_name>/image.jpg
    Since we don't have explicit val/test folders, we will split the dataset:
    80% train, 10% val, 10% test
    """
    # Create the training dataset (80%)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_names=class_names,
        label_mode=label_mode
    )

    # Create the validation+test dataset (20%)
    val_test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_names=class_names,
        label_mode=label_mode
    )

    # Split the 20% holdout into validation and test batches. Keep a usable
    # dataset on both sides when the holdout fits in a single batch.
    val_test_batches = int(tf.data.experimental.cardinality(val_test_ds).numpy())
    if val_test_batches <= 1:
        val_ds = val_test_ds
        test_ds = val_test_ds
    else:
        test_batches = max(1, val_test_batches // 2)
        test_ds = val_test_ds.take(test_batches)
        val_ds = val_test_ds.skip(test_batches)

    # Autotune for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds
