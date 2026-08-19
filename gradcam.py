import numpy as np
import tensorflow as tf
import cv2

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generates a Grad-CAM heatmap for a given image and model.
    """
    # Create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    
    # In our architecture, the base model is a layer.
    # We must find the inner base model, or if the model itself is flat.
    # We'll just define the grad model dynamically.
    inner_model_found = False
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            try:
                layer.get_layer(last_conv_layer_name)
                base_model = layer
                inner_model_found = True
                break
            except ValueError:
                continue
            
    if inner_model_found: # If using a nested Functional model
        last_conv_layer = base_model.get_layer(last_conv_layer_name)
        
        # Sub-model from base model input to last conv layer
        last_conv_model = tf.keras.Model(base_model.inputs, last_conv_layer.output)
        
        # Sub-model from last conv layer to final predictions
        # We need to chain the layers after base_model
        classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input
        # Find index of base_model in the main model
        base_model_idx = [i for i, l in enumerate(model.layers) if l.name == base_model.name][0]
        for layer in model.layers[base_model_idx+1:]:
            x = layer(x)
        classifier_model = tf.keras.Model(classifier_input, x)
        
        # Now pass the image through the nested architecture
        with tf.GradientTape() as tape:
            # We want gradients of the output wrt the last conv layer of the base model
            # But the first layer of our main model is DataAugmentation, so we should 
            # pass the image through layers up to base_model.
            x = img_array
            for layer in model.layers[:base_model_idx]:
                x = layer(x, training=False)
            
            # Now pass x through the nested base model up to last conv layer
            last_conv_layer_output = last_conv_model(x)
            tape.watch(last_conv_layer_output)
            
            # Now compute predictions from the last conv layer output
            preds = classifier_model(last_conv_layer_output)
            if pred_index is None:
                # If binary, it's 0. If categorical, argmax.
                if preds.shape[-1] == 1:
                    pred_index = 0
                else:
                    pred_index = tf.argmax(preds[0])
                    
            if preds.shape[-1] == 1:
                class_channel = preds[0, 0]
            else:
                class_channel = preds[:, pred_index]

    else:
        # Standard flat model Grad-CAM
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                if preds.shape[-1] == 1:
                    pred_index = 0
                else:
                    pred_index = tf.argmax(preds[0])
                    
            if preds.shape[-1] == 1:
                class_channel = preds[0, 0]
            else:
                class_channel = preds[:, pred_index]

    gradients = tape.gradient(class_channel, last_conv_layer_output)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_gradients[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def overlay_gradcam(img_path, heatmap, alpha=0.4):
    """
    Overlays the heatmap on the original image.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize heatmap to match image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return superimposed_img
