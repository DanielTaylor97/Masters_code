################################
# TAKES NORMALISED VOLUMES     #
# CREATES THE FINAL DATA FILES #
################################

from __future__ import division
import numpy as np
from getIDs import GetIDs


# Load all patient information from .csv file, randomised
info = GetIDs()
[IDs, ages, count] = info.getRandomised()


def training_test(ids, a, n, norm='GCN'):
    """
    Splits the dataset into training and test sets randomly

    list of Strings :param ids: The IDs of all the individuals in the dataset

    list of floats :param a:    The ages of all the individuals in the dataset

    int :param n:   The total number of individuals in the dataset

    String :param norm: The norm used on the data to be split
    """

    # Define how many individuals will be in the training and test sets (80/20 split)
    n_train = int(0.8*n)
    n_test = n - n_train
    # Display the number of individuals in each set
    print('{} training samples, {} test samples; {} in total.\n'.format(n_train, n_test, n_test + n_train))

    # Split the ages into the training and test sets
    train_ages = a[0:n_train]
    test_ages = a[n_train:n]

    # Split the IDs into the training and test sets
    train_ids = ids[0:n_train]
    test_ids = ids[n_train:n]

    # Save the training and test sets for ages and IDs
    np.save('finalData/train_ages_{}'.format(norm), train_ages, allow_pickle=True)
    np.save('finalData/test_ages_{}'.format(norm), test_ages, allow_pickle=True)
    np.save('finalData/train_IDs_{}'.format(norm), train_ids, allow_pickle=True)
    np.save('finalData/test_IDs_{}'.format(norm), test_ids, allow_pickle=True)

    # Initialise the test scans array
    test_scans = np.zeros(shape=(n_test, 233, 189, 197))
    # Loop over the individuals in the test set
    for i in range(n_test):

        # Load the (compressed) scan of the current individual
        file = np.load('{}/{}_T1w.npz'.format(norm, test_ids[i]), allow_pickle=True)
        next_scan = np.array([])
        for item in file.files:
            next_scan = file[item]

        # Insert the scan into the test scans set
        test_scans[i, :, :, :] = next_scan
        # Update the user on the progress of the operation
        print('\r{} of {}'.format(i + 1, n_test), end='')

    # Save the test scans set
    np.savez_compressed('finalData/test_scans_{}'.format(norm), test_scans)
    print()
    # Inform the user when the test set has been saved
    print('Final test data shape: {}\n'.format(np.shape(test_scans)))
    # When the test set is saved, clear the space in RAM -- this is usually not necessary and is done automatically
    test_scans = []

    # Initialise the training scans array
    train_scans = np.zeros(shape=(n_train, 233, 189, 197))
    # Loop over the individuals in the training set, what its shape is
    for j in range(n_train):

        # Load the (compressed) scan of the current individual
        file = np.load('{}/{}_T1w.npz'.format(norm, train_ids[j]), allow_pickle=True)
        next_scan = np.array([])
        for item in file.files:
            next_scan = file[item]

        # Insert the scan into the training scans set
        train_scans[j, :, :, :] = next_scan
        # Update the user on the progress of the operation
        print('\r{} of {}'.format(j + 1, n_train), end='')

    # Save the training scans set
    np.savez_compressed('finalData/train_scans_{}'.format(norm), train_scans)
    print()
    # Inform the user when the training set has been saved, what its shape is
    print('Final training data shape: {}\n'.format(np.shape(train_scans)))
    # When the training set is saved, clear the space in RAM -- this is usually not necessary and is done automatically
    train_scans = []


training_test(IDs, ages, count)
