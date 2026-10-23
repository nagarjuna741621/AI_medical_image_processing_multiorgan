import numpy as np
import tensorflow as tf
import cv2

def resolve_last_conv_layer(model, preferred_name=None):
    """Dynamically finds the best convolutional/activation layer for Grad-CAM."""
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break

    target_model = base_model if base_model is not None else model

    if preferred_name:
        try:
            target_model.get_layer(preferred_name)
            return preferred_name, (base_model is not None), base_model
        except ValueError:
            pass

    # Common candidates in DenseNet, EfficientNet, MobileNet
    candidates = ['relu', 'top_activation', 'conv5_block16_2_conv', 'conv5_block16_concat', 'top_conv']
    for c in candidates:
        try:
            target_model.get_layer(c)
            return c, (base_model is not None), base_model
        except ValueError:
            pass

    # Search backwards for Conv2D, DepthwiseConv2D, or Activation
    for layer in reversed(target_model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D, tf.keras.layers.Activation)):
            return layer.name, (base_model is not None), base_model

    return target_model.layers[-1].name, (base_model is not None), base_model


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """
    Generates a Grad-CAM heatmap for a given image and model.
    Dynamically adapts to DenseNet-121, EfficientNet, or flat architectures.
    """
    last_conv_layer_name, inner_model_found, base_model = resolve_last_conv_layer(model, last_conv_layer_name)

    if inner_model_found:  # If using a nested Functional model
        last_conv_layer = base_model.get_layer(last_conv_layer_name)
        
        last_conv_model = tf.keras.Model(base_model.inputs, last_conv_layer.output)
        
        classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input
        base_model_idx = [i for i, l in enumerate(model.layers) if l.name == base_model.name][0]
        for layer in model.layers[base_model_idx+1:]:
            x = layer(x)
        classifier_model = tf.keras.Model(classifier_input, x)
        
        with tf.GradientTape() as tape:
            x = img_array
            for layer in model.layers[:base_model_idx]:
                if isinstance(layer, tf.keras.layers.InputLayer):
                    continue
                try:
                    x = layer(x, training=False)
                except TypeError:
                    x = layer(x)
            
            last_conv_layer_output = last_conv_model(x)
            tape.watch(last_conv_layer_output)
            
            preds = classifier_model(last_conv_layer_output)
            if preds.shape[-1] == 1:
                if pred_index is None:
                    pred_index = 1 if float(preds[0, 0]) > 0.5 else 0
                class_channel = preds[0, 0] if pred_index == 1 else (1.0 - preds[0, 0])
            else:
                if pred_index is None:
                    pred_index = int(tf.argmax(preds[0]))
                class_channel = preds[:, pred_index]

    else:
        # Standard flat model Grad-CAM
        grad_model = tf.keras.models.Model(
            model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if preds.shape[-1] == 1:
                if pred_index is None:
                    pred_index = 1 if float(preds[0, 0]) > 0.5 else 0
                class_channel = preds[0, 0] if pred_index == 1 else (1.0 - preds[0, 0])
            else:
                if pred_index is None:
                    pred_index = int(tf.argmax(preds[0]))
                class_channel = preds[:, pred_index]

    gradients = tape.gradient(class_channel, last_conv_layer_output)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_gradients[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / (max_val + 1e-10)
    
    return heatmap.numpy()


def overlay_gradcam(img_input, heatmap, alpha=0.4):
    """
    Overlays the heatmap on the original image (supports file path or numpy array),
    ensuring correct RGB color space rendering.
    """
    if isinstance(img_input, str):
        img = cv2.imread(img_input)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = np.array(img_input, copy=True)
    
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
    
    superimposed_img = (colored_heatmap * alpha) + (img * (1.0 - alpha))
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return superimposed_img
