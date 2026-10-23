import os
import tensorflow as tf
from tensorflow.keras import layers
from src.config import MODELS_DIR

class SqueezeExcitation(layers.Layer):
    def __init__(self, in_filters, se_ratio=0.25, **kwargs):
        super().__init__(**kwargs)
        self.in_filters = in_filters
        self.se_ratio = se_ratio
        reduced = max(1, int(in_filters * se_ratio))
        self.reduce = layers.Dense(reduced, activation='relu')
        self.expand = layers.Dense(in_filters, activation='sigmoid')

    def call(self, x):
        b, h, w, c = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        squeeze = tf.reduce_mean(x, axis=[1, 2])
        excitation = self.expand(self.reduce(squeeze))
        excitation = tf.reshape(excitation, [b, 1, 1, c])
        return x * excitation

    def get_config(self):
        config = super().get_config()
        config.update({
            "in_filters": self.in_filters,
            "se_ratio": self.se_ratio,
        })
        return config


class MBConvBlock(layers.Layer):
    def __init__(self, in_filters, out_filters, expand_ratio=4, stride=1, **kwargs):
        super().__init__(**kwargs)
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.expand_ratio = expand_ratio
        self.stride = stride
        self.use_residual = (stride == 1 and in_filters == out_filters)
        expanded = in_filters * expand_ratio
        self.expand_conv = layers.Conv2D(expanded, kernel_size=1, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.dw_conv = layers.DepthwiseConv2D(kernel_size=3, strides=stride, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.se = SqueezeExcitation(expanded)
        self.proj_conv = layers.Conv2D(out_filters, kernel_size=1, padding='same', use_bias=False)
        self.bn3 = layers.BatchNormalization()

    def call(self, x):
        res = x
        x = tf.nn.swish(self.bn1(self.expand_conv(x)))
        x = tf.nn.swish(self.bn2(self.dw_conv(x)))
        x = self.se(x)
        x = self.bn3(self.proj_conv(x))
        if self.use_residual:
            x = x + res
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "in_filters": self.in_filters,
            "out_filters": self.out_filters,
            "expand_ratio": self.expand_ratio,
            "stride": self.stride,
        })
        return config


class CoAtNetAttentionStage(layers.Layer):
    def __init__(self, out_dim, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=out_dim // num_heads)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        self.ffn = tf.keras.Sequential([
            layers.Dense(out_dim * 2, activation='gelu'),
            layers.Dense(out_dim)
        ])

    def call(self, x):
        b, h, w, c = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        flat = tf.reshape(x, [b, h * w, c])
        n1 = self.norm1(flat)
        attn = self.mha(n1, n1)
        flat = flat + attn
        n2 = self.norm2(flat)
        flat = flat + self.ffn(n2)
        return tf.reshape(flat, [b, h, w, c])

    def get_config(self):
        config = super().get_config()
        config.update({
            "out_dim": self.out_dim,
            "num_heads": self.num_heads,
        })
        return config


class MobileViTBlock(layers.Layer):
    def __init__(self, out_filters, transformer_dim, num_heads=2, **kwargs):
        super().__init__(**kwargs)
        self.out_filters = out_filters
        self.transformer_dim = transformer_dim
        self.num_heads = num_heads
        self.local_conv1 = layers.Conv2D(out_filters, kernel_size=3, padding='same', activation='swish')
        self.local_conv2 = layers.Conv2D(transformer_dim, kernel_size=1, padding='same', activation='swish')
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=transformer_dim // num_heads)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        self.ffn = tf.keras.Sequential([
            layers.Dense(transformer_dim * 2, activation='swish'),
            layers.Dense(transformer_dim)
        ])
        self.fusion_conv1 = layers.Conv2D(out_filters, kernel_size=1, padding='same', activation='swish')
        self.fusion_conv2 = layers.Conv2D(out_filters, kernel_size=3, padding='same', activation='swish')

    def call(self, x):
        res = x
        b, h, w = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        local = self.local_conv2(self.local_conv1(x))
        seq = tf.reshape(local, [b, h * w, self.transformer_dim])
        n1 = self.norm1(seq)
        attn = self.mha(n1, n1)
        seq = seq + attn
        n2 = self.norm2(seq)
        seq = seq + self.ffn(n2)
        glob = tf.reshape(seq, [b, h, w, self.transformer_dim])
        fused = self.fusion_conv1(glob)
        out = self.fusion_conv2(tf.concat([res, fused], axis=-1))
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "out_filters": self.out_filters,
            "transformer_dim": self.transformer_dim,
            "num_heads": self.num_heads,
        })
        return config


CUSTOM_OBJECTS = {
    'SqueezeExcitation': SqueezeExcitation,
    'MBConvBlock': MBConvBlock,
    'CoAtNetAttentionStage': CoAtNetAttentionStage,
    'MobileViTBlock': MobileViTBlock
}

def load_medical_model(model_identifier):
    """Loads any medical model (.keras) from disk with custom layer support."""
    if os.path.isabs(model_identifier) and os.path.exists(model_identifier):
        path = model_identifier
    elif os.path.exists(model_identifier):
        path = os.path.abspath(model_identifier)
    else:
        path = os.path.join(MODELS_DIR, f"{model_identifier}.keras")
        if not os.path.exists(path):
            path = os.path.join(MODELS_DIR, model_identifier)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Medical model file not found: {path}")

    print(f"Loading medical model from: {path}...")
    model = tf.keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS, compile=False)
    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load and inspect trained medical models")
    parser.add_argument("--model", type=str, default="densenet121_chest_model.keras", help="Model file in models/ to inspect")
    args = parser.parse_args()

    m = load_medical_model(args.model)
    print("\n--- Model Architecture Summary ---")
    m.summary()
