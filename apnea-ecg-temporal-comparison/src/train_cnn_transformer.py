"""
M0 - Full model (CNN + ECA + Transformer).
Same training as the other models: cosine LR, MixUp, TTA x10, threshold search.
Run from inside src/:  python train_cnn_transformer.py
Saves the best model to weights/cnn_transformer.weights.h5
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling modules importable from any folder

# [split] replaces the original 'from base import (...)'
from keras_config import SEED
from keras_data_loading import load_data
from keras_training import run_training
from keras_base_model import ECABlock, TransformerBlock, PositionalEncoding
import numpy as np
import tensorflow as tf
from keras.layers import (
    Conv1D, Dense, Reshape, GlobalAveragePooling1D, GlobalMaxPooling1D,
    Input, Dropout, BatchNormalization, Activation, Concatenate, MaxPooling1D,
)
from keras.models import Model
from keras.utils import to_categorical

tf.random.set_seed(SEED)
np.random.seed(SEED)

def create_model(input_shape, dropout_rate=0.1):
    inp     = Input(shape=input_shape)

    rri_in  = Reshape((900, 1))(inp[:, :, 0])
    ampl_in = Reshape((900, 1))(inp[:, :, 1])

    def branch(sig):
        x = Conv1D(16, 11, padding="same")(sig)
        x = BatchNormalization()(x); x = Activation("relu")(x)
        x = Conv1D(24, 11, strides=2, padding="same")(x)
        x = BatchNormalization()(x); x = Activation("relu")(x)
        x = MaxPooling1D(3, padding="same")(x)
        x = Conv1D(32, 11, padding="same")(x)
        x = BatchNormalization()(x); x = Activation("relu")(x)
        x = MaxPooling1D(5, padding="same")(x)
        return x

    fused = Concatenate(axis=-1)([branch(rri_in), branch(ampl_in)])  # (B,30,64)

    fused = ECABlock(kernel_size=3)(fused)

    fused = PositionalEncoding(seq_len=30, embed_dim=64)(fused)

    fused = TransformerBlock(embed_dim=64, num_heads=4, ff_dim=128, dropout=dropout_rate)(fused)
    fused = TransformerBlock(embed_dim=64, num_heads=4, ff_dim=128, dropout=dropout_rate)(fused)

    gap = GlobalAveragePooling1D()(fused)
    gmp    = GlobalMaxPooling1D()(fused)
    pooled = Concatenate(axis=-1)([gap, gmp])  # (B,128)
    pooled  = Dense(64, activation="relu")(pooled)
    pooled  = Dropout(dropout_rate)(pooled)
    outputs = Dense(2, activation="softmax", name="outputs")(pooled)
    return Model(inputs=inp, outputs=outputs)


if __name__ == "__main__":
    # Load data
    x_train, y_train, x_val, y_val, x_test, y_test = load_data()
    y_train_cat = to_categorical(y_train, 2)
    y_val_cat   = to_categorical(y_val,   2)
    y_test_cat  = to_categorical(y_test,  2)
    print(f"Train:{x_train.shape}  Val:{x_val.shape}  Test:{x_test.shape}")

    model  = create_model(x_train.shape[1:])
    result = run_training(model, "cnn_transformer",
                          x_train, y_train_cat,
                          x_val, y_val, y_val_cat,
                          x_test, y_test, y_test_cat)
    print(result)
