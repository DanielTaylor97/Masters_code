from __future__ import division, absolute_import, print_function, unicode_literals
from tensorflow.keras import layers
import tensorflow.keras as keras
import numpy as np


class ResidueLayer2D(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            layers.Conv2D(filters, 3, strides=strides, padding="same", use_bias=False),
            layers.BatchNormalization(),
            self.activation,
            layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False),
            layers.BatchNormalization()
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                layers.Conv2D(filters, 1, strides=strides, padding="same", use_bias=False),
                layers.BatchNormalization()
            ]

    # No build method, since the parameters are all in the convolution and BatchNorm layers anyway
    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)


class ResidueLayer3D(keras.layers.Layer):
    def __init__(self, filters, adjust_kernel=False, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            layers.Conv3D(filters, kernel_size(filters) if adjust_kernel else 3,
                          strides=strides, padding="same", use_bias=False),
            layers.BatchNormalization(),
            self.activation,
            layers.Conv3D(filters, kernel_size(filters) if adjust_kernel else 3,
                          strides=1, padding="same", use_bias=False),
            layers.BatchNormalization()
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                layers.Conv3D(filters, 1, strides=strides, padding="same", use_bias=False),
                layers.BatchNormalization()
            ]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)


# If we can or want to, we start with a larger convolution window and contract it as we increase the number of filters
def kernel_size(filters):

    k = 2*(8-np.log2(filters)) + 1
    k = int(k)

    return k


# If we want to use the relevance decomposition, we need each layer to be retrievable from the network, and so we cannot
# use the residue block as a single layer. We also dispense with the strides >1 and instead use ave pooling layers to
# downsample, so as to avoid grid patterns in the saliency maps
def residue_components_2d(model, filters, adjust_kernel=False, activation="relu"):
    model_skip = model

    model = layers.Conv2D(filters, kernel_size(filters) if adjust_kernel else 3,
                          strides=1, padding="same", use_bias=False)(model)
    model = layers.BatchNormalization()(model)
    model = keras.activations.get(activation)(model)
    model = layers.Conv2D(filters, kernel_size(filters) if adjust_kernel else 3,
                          strides=1, padding="same", use_bias=False)(model)
    model = layers.BatchNormalization()(model)

    model_skip = layers.Conv2D(filters, 1, strides=1, padding="same", use_bias=False)(model_skip)
    model_skip = layers.BatchNormalization()(model_skip)

    model = layers.Add()([model, model_skip])
    model = keras.activations.get(activation)(model)

    return model


def residue_components_3d(model, filters, adjust_kernel=False, activation="relu"):
    model_skip = model

    model = layers.Conv3D(filters, kernel_size(filters) if adjust_kernel else 3,
                          strides=1, padding="same", use_bias=False)(model)
    model = layers.BatchNormalization()(model)
    model = keras.activations.get(activation)(model)
    model = layers.Conv3D(filters, kernel_size(filters) if adjust_kernel else 3,
                          strides=1, padding="same", use_bias=False)(model)
    model = layers.BatchNormalization()(model)

    model_skip = layers.Conv3D(filters, 1, strides=1, padding="same", use_bias=False)(model_skip)
    model_skip = layers.BatchNormalization()(model_skip)

    model = layers.Add()([model, model_skip])
    model = keras.activations.get(activation)(model)

    return model
