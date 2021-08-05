from __future__ import division, print_function
import numpy as np

# Load (compressed) atlas file
atlas_file = np.load('embedded_atlas.npz', allow_pickle=True)
atlas = np.array([])
for item in atlas_file.files:
    atlas = atlas_file[item]


def get_top_p(saliency_map, p=0.1):
    '''
    Fetches the top-p percent of pixel activations from the supplied volume as ones and leaves all other activations as
    zero

    np.ndarray :param map:  Saliency map to be worked with (single map)

    float :param p: Top percentage of activations to take from the saliency map (between 0.0 and 1.0)

    np.ndarray :return top_p_map:   Top p percent of activations in map
    '''

    # Check that the given value of p is valid
    if p < 0.0 or p > 1.0:
        print('Invalid choice of p')
        # If not, simply return the map that was given
        return saliency_map
    else:
        # Sort activations
        sorted_acts = np.sort(saliency_map, axis=None)
        n_voxels = int(sorted_acts.shape[0])

        # Find position of p-th percentile activation
        n_p = int(p*n_voxels)

        # Extract p-th percentile activation
        top_p = sorted_acts[-n_p]

        # Get all values in top-p percent as ones on a background of zeros
        top_p_map = (saliency_map >= top_p).astype(int)

        return top_p_map


def save_region_correspondences(method, p=0.1, masking=False):
    """
    Find and save the correspondences between the regions of the atlas and the top-p percent of voxel activations in the
    saliency maps from a given method for each individual in the test set

    String :param method:   The saliency mapping method to be examined

    float :param p: The proportion of top-activation values to examine from each map (Default: 0.1)

    bool :param masking:    Whether or not the maps to be examined are for the masking task, as opposed to the age
    regression task (Default: False)
    """

    # Create the path names according to the task and method, and load the appropriate test IDs data
    if masking:
        path = '{}_masking/{}_masking_{}.npz'
        save_path = 'masking_correspondences/{}_masking_correspondences_{}%'
        ids = np.load('test_data/test_ids_masking.npy')
    else:
        path = '{}_brain_age/{}_ageing_{}.npz'
        save_path = 'ageing_correspondences/{}_ageing_correspondences_{}%'
        ids = np.load('test_data/test_IDs_GCN.npy')

    # Find the number of regions in the atlas -- they are indexed as integers from 1.0 upwards
    max_act = int(np.max(atlas))
    # Initialise the array of region correspondences -- Rows are correspondences per individual, columns are
    # correspondences per region
    top_p_correspondences = np.zeros(shape=(len(ids), max_act))

    # Loop through IDs
    for i, ID in enumerate(ids):
        # Lod appropriate saliency map (compressed)
        saliency_map_file = np.load(path.format(method, ID, method), allow_pickle=True)
        saliency_map = np.array([])
        for item in saliency_map_file.files:
            saliency_map = saliency_map_file[item]

        # Reshape saliency map to atlas shape for comparison
        saliency_map = np.reshape(saliency_map, newshape=atlas.shape)

        # Initialise region array
        region = np.zeros_like(atlas)

        # Fetch top-p percentages from saliency map
        saliency_top_p = get_top_p(saliency_map, p=p)

        # Loop through atlas regions
        for j in range(1, max_act + 1):
            # Get single region
            region = (atlas == j).astype(int)
            # Count how many of the top-p voxel activations are in this region (activations are either 1 or 0)
            total = np.sum(region*saliency_top_p)
            # Record the number of correspondences for the region for this individual
            top_p_correspondences[i, j - 1] = total

        # Keep the user updated on progress
        print('\rFinished top {}% {} correspondences for {}.\t{}/{}'.format(int(100*p), method, ID, i + 1, len(ids)), end='')

    # Save the correspondences according to method and p
    np.save(save_path.format(method, int(100 * p)), top_p_correspondences, allow_pickle=True)
    # Notify the user when the process is finished
    print('\nFinished calculating top {}% correspondences for {} method.\n\n\n'.format(int(100*p), method))


save_region_correspondences('LRP_1', p=0.1, masking=True)
save_region_correspondences('DeepLIFT', p=0.1, masking=True)

save_region_correspondences('LRP_1', p=0.05, masking=True)
save_region_correspondences('DeepLIFT', p=0.05, masking=True)

save_region_correspondences('LRP_1', p=0.02, masking=True)
save_region_correspondences('DeepLIFT', p=0.02, masking=True)

save_region_correspondences('LRP_1', p=0.01, masking=True)
save_region_correspondences('DeepLIFT', p=0.01, masking=True)
