from __future__ import division
import numpy as np
from tensorflow.keras import models
from matplotlib import pyplot as plt
from celluloid import Camera


def fetch_model_and_data(norm, arch, act, give_info=False):
    """
    Get the model, and the necessary training data to test the model

    String :param norm: The norm used on the data on which the model was trained

    String :param arch: The type of model used

    String :param act:  The non-linearity used in the network

    bool :param give_info:  Whether or not to display the network and data information (Default: False)

    tf.keras.model, np.ndarray, np.ndarray :return: The trained model and associated test scans and data
    """

    # Load the training ages, the model and the training scans (compressed)
    one_ages = np.load('test_ages_{}.npy'.format(norm), allow_pickle=True)
    one_model = models.load_model('MRI_{}_{}_model_{}'.format(norm, arch, act))
    data_file = np.load('test_scans_{}.npz'.format(norm), allow_pickle=True)
    one_data = np.array([])
    for item in data_file.files:
        one_data = data_file[item]

    # Show the size of the test data in memory
    print("Test data loaded at {} bytes".format(one_data.nbytes))
    # Reshape the test data
    one_data = np.reshape(one_data, newshape=(-1, 233, 189, 197, 1))

    # Display information about the data and model if necessary
    if give_info:
        print('Data shape: {}'.format(np.shape(one_data)))
        print()
        one_model.summary()
        print()

        for i in range(12):
            layer = one_model.get_layer(index=i)
            config = layer.get_config()
            print(config)

    return one_model, one_data, one_ages


def make_single_prediction(mod, dat, age, give_stats=False):
    """
    Make a prediction on a single individual in the test set and show the result alongside an animation of the patient's
    brain volume

    tf.keras.model :param mod:  The model to be used in predicting

    np.ndarray :param dat:  The array containing all the test scans

    np.ndarray :param age:  The array containing all the test ages

    bool :param give_stats: Whether or not to show some basic statistics about the voxel activations in a chosen scan
    """

    # Choose a random position in the data array and extract the corresponding scan and age
    single_pos = np.random.randint(low=0, high=int(np.shape(dat)[0]))
    single_scan = dat[single_pos:(single_pos + 1), :, :, :, :]
    single_age = age[single_pos]
    # Make a prediction based on the chosen volume
    pred = mod.predict(single_scan)

    # Display the min, max and mean voxel activations if necessary
    if give_stats:
        mean = np.mean(single_scan)
        mx = np.max(single_scan)
        mn = np.min(single_scan)
        print('Mean: {}\nMax: {}\nMin: {}\n'.format(mean, mx, mn))

    # Show the predicted vs actual age
    print('Predicted age: {}\nActual age: {}'.format(pred, single_age))

    # Reshape the scan for display
    single_scan = np.reshape(single_scan, newshape=(233, 189, 197))
    # Initialise the figure
    fig = plt.figure()
    # Point the camera at the figure
    camera = Camera(fig)

    # Sequentially display sections of the volume as 'frames' and take snapshots
    for i in range(single_scan.shape[1]):
        plt.imshow(single_scan[:, -i, :], cmap='gray')
        camera.snap()

    # Compile the animation
    animation = camera.animate()

    plt.show()


model, data, ages = fetch_model_and_data("GCN", "ResNet", "softplus")

# make_single_prediction(model, data, ages)

# Use the built-in Keras evaluation tool
test_loss, test_mape, test_mae = model.evaluate(data, ages, batch_size=16)
# Show evaluation loss and metrics
print("Trained model MAPE: {}\nMAE: {},\nLoss: {}".format(test_mape, test_mae, test_loss))
