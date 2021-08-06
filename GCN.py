########################################################
# NORMALISATION TAKES REGISTERED VOLUME                #
# FINAL STAGE OF PREPROCESSING BEFORE TRAIN-TEST SPLIT #
########################################################

from __future__ import division
import numpy as np
import nibabel as nib
from getIDs import GetIDs


# Fetch individuals' information from .csv file
info = GetIDs()
[IDs, ages, count] = info.getAll()

# Define threshold for normalisation
eps = 1e-9


def gcn():
    """
    Normalise volumes using Global Contrast Normalisation and save the results
    """

    # Loop over the number of individuals
    for i in range(count):
        # Update the user on where in the process they are
        print('\rNormalising {} of {} by GCN.'.format(i + 1, count), end='')
        # Load registered file
        file = nib.load('registered/{}_FLIRT_file.nii.gz'.format(IDs[i]))
        # Convert from Nifti file to Numpy array
        next_scan = np.array(file.get_fdata())

        # Calculate contrast (we define it as the SD)
        contrast = np.std(next_scan)
        # Ensure that the denominator is not too small
        denominator = max(eps, contrast)
        # Normalise
        normalised_vol = next_scan/denominator

        # Save Numpy file
        np.savez_compressed('GCN/{}_T1w'.format(IDs[i]), normalised_vol)

    print('\nGCN completed on all scans.')


def mean_norm():
    """
    Normalise volumes by the mean and save the results
    """

    # Loop over the number of individuals
    for j in range(count):
        # Update the user on where in the process they are
        print('\rNormalising {} of {} by GCN.'.format(j + 1, count), end='')
        # Load registered file
        file = nib.load('registered/{}_FLIRT_file.nii.gz'.format(IDs[j]))
        # Convert from Nifti file to Numpy array
        next_scan = np.array(file.get_fdata())

        # Calculate mean
        mean = np.mean(next_scan)
        # Prevent denominator from being really small (all activations are non-negative)
        denominator = np.max(eps, mean)
        # Normalise
        normalised_vol = next_scan/denominator

        # Save Numpy file
        np.savez_compressed('meanNorm/{}_T1w_cr'.format(IDs[j]), normalised_vol)

    # Tell the user when the process is finished
    print('\nMean normalisation completed on all scans.')


def check_shape(norm='GCN'):
    """
    Verifies the shape of the output files

    String :param norm: The norm under whose directory to look for files (Default: 'GCN')
    """
    # Load the first file in the norm directory (compressed)
    file = np.load('{}/CC110033_T1w.npz'.format(norm), allow_pickle=True)
    next_scan = np.array([])
    for item in file.files:
        next_scan = file[item]

    # Print file shape
    print('Single scan shape: {}'.format(next_scan.shape))


def save_new():
    """
    Save re-registered and normed files
    """

    for i in range(count):

        # Load the appropriate GCN files (compressed)
        file = np.load('GCN/CC{}_T1w.npz'.format(IDs[i]), allow_pickle=True)
        next_scan = np.array([])
        for item in file.files:
            next_scan = file[item]

        # Save them in a separate folder to be carried to the other laptop
        np.savez_compressed('new_GCN/CC{}_T1w'.format(IDs[i]), next_scan)
        # Update the user on progress
        print('\rSaved {} of {}.'.format(i + 1, count), end='')


save_new()
#gcn()
#mean_norm()

#check_shape()
##Single scan shape: (233, 189, 197)
##Single scan shape: (256, 256, 192) {{{PRE-REGISTRATION SHAPE}}}