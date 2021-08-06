from __future__ import division, absolute_import, print_function, unicode_literals
from tensorflow.keras import layers, models, preprocessing, callbacks
import numpy as np
import ResNet
from ResNet import ResidueLayer3D


# Set some network hyperparameters
activation = "softplus"
input_shape = [233, 189, 197, 1]
n_epochs = 15
batch_size = 4


def create_resnet(init_filters, filters_split):
    """
    Returns a ResNet model with residue blocks as a single layer

    int :param init_filters:    The number of filters to use in the first convolution layer

    list of ints :param filters_split:  The number of filters to use in the convolution layers of each residue block

    tf.keras.model :return: A ResNet Keras Model object
    """

    # Sequentially define the model
    rn_model = models.Sequential()
    rn_model.add(layers.Input(shape=(s for s in input_shape[0:-1])))
    rn_model.add(layers.Reshape((s for s in input_shape)))
    rn_model.add(layers.Conv3D(filters=init_filters,
                               kernel_size=7,
                               strides=2,
                               input_shape=input_shape,
                               padding="same",
                               use_bias=False))
    rn_model.add(layers.BatchNormalization())
    rn_model.add(layers.Activation(activation))
    rn_model.add(layers.AveragePooling3D())

    # Create the residue blocks as layers
    prev_filters = init_filters
    for filters in filters_split:
        # When we increase the number of filters, we downsample using strides >1
        strides = 1 if filters == prev_filters else 2
        rn_model.add(ResidueLayer3D(filters, strides=strides, activation=activation))
        prev_filters = filters

    rn_model.add(layers.GlobalAveragePooling3D())
    rn_model.add(layers.Flatten())
    # Use a linear activation for the regression model
    rn_model.add(layers.Dense(1, activation="linear"))

    return rn_model


def create_relprop_resnet(init_filters, filters_split):
    """
    Create a ResNet compatible with relevance decomposition. residue blocks are not considered a single layer, and we do
    not downsample with convolution strides >1, but rather with average pooling

    int :param init_filters:    The number of filters in the first convolutional layer

    list of ints :param filters_split:  The number of filters to use in the convolution layers of each residue block

    tf.keras.model :return: A ResNet Keras Model object
    """

    # Define the model by passing one layer to the next
    inputs = layers.Input(shape=(s for s in input_shape[0:-1]))
    rn_model = inputs
    rn_model = layers.Reshape((s for s in input_shape))(rn_model)
    rn_model = layers.Conv3D(filters=init_filters,
                             kernel_size=7,
                             strides=1,
                             input_shape=input_shape,
                             padding="same",
                             use_bias=False)(rn_model)
    rn_model = layers.BatchNormalization()(rn_model)
    rn_model = layers.Activation(activation)(rn_model)
    rn_model = layers.AveragePooling3D()(rn_model)

    # Create residue blocks with separate layers
    prev_filters = init_filters
    for filters in filters_split:
        rn_model = ResNet.residue_components_3d(rn_model, filters, activation=activation)
        # When we increase the number of filters, we downsample at the end of the residue block using average pooling
        if filters != prev_filters:
            rn_model = layers.AveragePooling3D(2)(rn_model)
        prev_filters = filters

    rn_model = layers.GlobalAveragePooling3D()(rn_model)
    rn_model = layers.Flatten()(rn_model)
    # Use a linear activation for the regression model
    rn_model = layers.Dense(1, activation="linear")(rn_model)

    model = models.Model(inputs=inputs, outputs=rn_model)

    return model


# Create the relevant type of network
# regression_model = create_resnet(32, [32, 32, 64, 64, 128])
regression_model = create_relprop_resnet(32, [32, 32, 64, 64, 128])
# 32, [32]*2 + [64]*2 + [128]

# Let the user know when the model is initialised
print()
print("Model created")

# Display the model structure
regression_model.summary()

# Compile the model with MAE and MAPE as metrics and MAPE as loss
regression_model.compile(optimizer='rmsprop',
                         loss='MAPE',
                         metrics=['MAPE', 'MAE'])

# Load the training data (scans are compressed)
train_labels = np.load('train_ages_GCN.npy', allow_pickle=True)
train_scans = np.load('train_scans_GCN.npz', allow_pickle=True)
train_images = np.array([])
for item in train_scans.files:
    train_images = train_scans[item]

print()
# Show the size of the training data
print("Training data loaded at {} bytes".format(train_images.nbytes))
# Reshape the training data
train_images = np.reshape(train_images, newshape=(-1, 233, 189, 197))
print()
print("Training data reshaped")

# Initialise the training batch generator
training_gen = preprocessing.image.ImageDataGenerator()

# Define the checkpoint callback
checkpoint_callback = callbacks.ModelCheckpoint('MRI_GCN_ResNet_model_{}'.format(activation))
# Train model using batch generator and callbacks
regression_model.fit(training_gen.flow(train_images, train_labels, batch_size=batch_size),
                     batch_size=batch_size,
                     epochs=n_epochs,
                     callbacks=[checkpoint_callback])
