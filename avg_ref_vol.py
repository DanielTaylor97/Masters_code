from __future__ import division
import numpy as np


def create_ave_vol():
    all_files = np.load('finalData/test_scans_GCN.npz', allow_pickle=True)
    all_scans = np.array([])
    for item in all_files.files:
        all_scans = all_files[item]

    composite = np.zeros_like(all_scans[0])
    print('Composite file shape: {}'.format(composite.shape))
    frac = 1/all_scans.shape[0]
    print('Fraction of each image to be added: {}\n'.format(frac))

    for i in range(all_scans.shape[0]):
        composite += frac*all_scans[i]
        print('\rAdded {} of {} volumes.'.format(i + 1, all_scans.shape[0]), end='')

    np.savez_compressed('finalData/composite_vol', composite)
    print('\n\nCompleted and saved composite image')


create_ave_vol()
