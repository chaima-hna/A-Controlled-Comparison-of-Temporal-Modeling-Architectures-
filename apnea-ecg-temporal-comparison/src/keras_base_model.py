# [split] import line(s) added so this file runs on its own; everything else is your original code, unchanged.
from keras_config import *

# ── Building blocks ───────────────────────────────────────────────────────────
@register_keras_serializable()

class ECABlock(keras.layers.Layer):

    def __init__(self, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = Conv1D(1, kernel_size, padding="same", use_bias=False)

    def call(self, x):
        gap     = keras.ops.mean(x, axis=1)
        gap     = keras.ops.expand_dims(gap, axis=-1)
        weights = keras.ops.sigmoid(self.conv(gap))
        weights = keras.ops.expand_dims(keras.ops.squeeze(weights, axis=-1), axis=1)
        return x * weights

    def get_config(self):
        cfg = super().get_config(); cfg.update({"kernel_size": self.kernel_size})
        return cfg

    @classmethod
    def from_config(cls, config): return cls(**config)


class TransformerBlock(keras.layers.Layer):

    def __init__(self, embed_dim, num_heads=4, ff_dim=128, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim; self.num_heads = num_heads
        self.ff_dim    = ff_dim;    self.dropout   = dropout
        self.attn  = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout)
        self.ffn   = keras.Sequential([Dense(ff_dim, activation="relu"), Dense(embed_dim)])
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.drop1 = Dropout(dropout)
        self.drop2 = Dropout(dropout)

    def call(self, x, training=False):
        a = self.drop1(self.attn(x, x, training=training), training=training)
        x = self.norm1(x + a)
        f = self.drop2(self.ffn(x), training=training)
        return self.norm2(x + f)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads,
                    "ff_dim": self.ff_dim, "dropout": self.dropout})
        return cfg

    @classmethod
    def from_config(cls, config): return cls(**config)


class PositionalEncoding(keras.layers.Layer):

    def __init__(self, seq_len, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len; self.embed_dim = embed_dim

    def build(self, input_shape):
        self.pos_emb = self.add_weight(
            name="pos_emb", shape=(1, self.seq_len, self.embed_dim),
            initializer="random_normal", trainable=True)
        super().build(input_shape)

    def call(self, x): return x + self.pos_emb

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"seq_len": self.seq_len, "embed_dim": self.embed_dim})
        return cfg

    @classmethod
    def from_config(cls, config): return cls(**config)


CUSTOM_OBJ = {
    "ECABlock": ECABlock,
    "TransformerBlock": TransformerBlock,
    "PositionalEncoding": PositionalEncoding,
}
