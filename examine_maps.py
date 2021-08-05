from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera


def fetch_random_id_masking(intensity):
    """
    Fetches the identity and masked region of an individual whose masking intensity corresponds with that provided

    int :param intensity:   Intensity of masking to be searched for

    String, int :return identity, region:   ID of the individual and region masked
    """

    # Load all masking test data
    all_ids = np.load('test_data/test_ids_masking.npy', allow_pickle=True)
    all_intensities = np.load('test_data/test_intensities_masking.npy', allow_pickle=True)
    all_regions = np.load('test_data/test_regions_masking.npy', allow_pickle=True)

    # Initialise sample intensity to allow the while loop to execute
    one_intensity = intensity + 1
    # Initialise the position to search
    pos = -1

    while one_intensity != intensity:
        # Exit the while loop immediately if the intensity is not between 0 and 10
        if intensity < 0 or intensity > 10:
            print('Invalid intensity provided')
            break
        # Pick a random position in the data
        pos = np.random.randint(low=0, high=all_intensities.shape[0])
        # Sample intensity to be tested
        one_intensity = all_intensities[pos]

    # Once the loop has finished, fetch the ID and region at the found position
    identity = all_ids[pos]
    region = all_regions[pos]
    # If an invalid intensity is supplied, the ID and region returned is from the end of the test set

    # Show information of masked individual at found position
    print('Patient identity: {}'.format(identity))
    print('Masking intensity: {}/10'.format(int(one_intensity)))
    print('Masked region: {}'.format(int(region)))

    return identity, region


def get_region_volume(region):
    """
    Fetches from the atlas the isolated region requested

    int :param region:  Code corresponding to the requested region

    numpy.ndarray :return region_vol:   The 3D volume of the region requested. Activations are 0 for outside the region
    and 1 for inside the region
    """

    # Load the whole atlas file (compressed)
    atlas_file = np.load('embedded_atlas.npz', allow_pickle=True)
    atlas = np.array([])
    for file in atlas_file.files:
        atlas = atlas_file[file]

    # Get array containing only the region of interest as ones on a background of zeros
    region_vol = (atlas == region).astype(int)

    return region_vol


def get_volume_masking(method, intensity=10, examine_id='', include_region_code=True):
    """
    Fetches the volume from the test data corresponding to the provided method and (optional) intensity and/or ID

    String :param method:   Method whose saliency map is to be fetched

    int :param intensity:   Intensity of the mask in the volume to be fetched

    String :param examine_id:   ID of the individual whose volume is to be fetched (Default: '')

    bool :param include_region_vol: Whether or not to return the name and code of the region (Default: True)

    np.ndarray, np.ndarray, int :return vol, region, region_code:   Returns the volume requested as well as (optionally)
    the volume and code of the masked region
    """

    # If an ID is not supplied, find a random region with the appropriate intensity, otherwise use ID to find the region
    if examine_id == '':
        # Fetch the information for a randomly chosen patient with the appropriate masking intensity
        patient_id, region_code = fetch_random_id_masking(intensity)
    else:
        patient_id = examine_id

        # Load the ID and region files
        all_ids = np.load('test_data/test_ids_masking.npy', allow_pickle=True)
        all_regions = np.load('test_data/test_regions_masking.npy', allow_pickle=True)

        # Get the position corresponding to the provided ID
        patient_pos = np.where(all_ids == patient_id)
        # Fetch the relevant region code
        region_code = all_regions[patient_pos]

    # Load the (compressed) saliency map file corresponding to the method and ID
    path = '{}_masking/{}_masking_{}.npz'.format(method, patient_id, method)
    file = np.load(path, allow_pickle=True)
    vol = np.array([])
    for item in file.files:
        vol = file[item]

    if include_region_code:
        # Fetch the volume of the corresponding region
        region = get_region_volume(region_code)

        return vol, region, region_code
    else:
        return vol


def fetch_random_id_ageing():
    """
    Returns a randomly indexed ID from the ageing test data

    String, float :return identity, age:    Returns both the ID and age of a randomly chosen individual in the test set
    """

    # Load all test IDs and ages
    all_ids = np.load('test_data/test_IDs_GCN.npy', allow_pickle=True)
    all_ages = np.load('test_data/test_ages_GCN.npy', allow_pickle=True)

    # Choose a random position in the array
    pos = np.random.randint(low=0, high=all_ages.shape[0])

    # Fetch the corresponding age and identity
    age = all_ages[pos]
    identity = all_ids[pos]
    # Show the information being returned
    print('Patient identity: {}'.format(identity))
    print('Patient age: {}'.format(int(age)))

    return identity, age


def get_volume_ageing(method, examine_id=''):
    """
    Gets the saliency map volume, ID and age corresponding to the given method and (optionally) ID

    String :param method:   The method whose saliency map is to be returned

    String :param examine_id:   The ID of a specific individual to be examined

    np.ndarray, String, float :return vol, patient_id, patient_age: Returns the saliency map, ID and age corresponding
    to the request
    """

    # If an ID is not provided, a random one is fetched along with its corresponding age; otherwise we find the
    # corresponding age
    if examine_id == '':
        patient_id, patient_age = fetch_random_id_ageing()
    else:
        patient_id = examine_id

        # Load ID and age test files
        all_ids = np.load('test_data/test_IDs_GCN.npy', allow_pickle=True)
        all_ages = np.load('test_data/test_ages_GCN.npy', allow_pickle=True)

        # Find the position of the given ID and find the corresponding age
        pos = np.where(all_ids == patient_id)
        patient_age = all_ages[pos]

    # Fetch the (compressed) volume according to the method and ID
    path = '{}_brain_age/{}_ageing_{}.npz'.format(method, patient_id, method)
    file = np.load(path, allow_pickle=True)
    vol = np.array([])
    for item in file.files:
        vol = file[item]

    return vol, patient_id, patient_age


def show_animation(vol, normalise=True):
    """
    Show an animation of the given volume as sequential cross-sections along an axis

    np.ndarray :param vol:  The volume to be displayed

    bool :param normalise:  Whether or not to normalise the image using the maximum and minimum activation values
    (Default: True)
    """

    # If we want to normalise, we add one pixel to each 'frame' with the minimum activation value and another with the
    # maximum activation value
    if normalise:
        mx = np.max(vol)
        mn = np.min(vol)
        vol[:, :, 0, 0, :] = mx
        vol[:, :, 0, -1, :] = mn

    # Create a figure and focus the camera on it
    fig = plt.figure()
    camera = Camera(fig)

    # Take snapshots of each 'frame'
    for i in range(vol.shape[1]):
        plt.imshow(vol[0, i, :, :, 0], cmap='viridis')
        camera.snap()

    # Compile the animation
    animation = camera.animate()
    # Show the animation
    plt.show()


def show_dual_animation_masking(vol_1, vol_2, method='LRP_1', intensity=0, region_code=0, save_gif=True):
    """
    Show side-by-side animations of masking saliency map for a given method and the corresponding region masked out

    np.ndarray :param vol_1:    The saliency map to be displayed

    np.ndarray :param vol_2:    The region masked in the volume corresponding to the saliency map

    String :param method:   The method used in implementing the given heatmap (Default: 'LRP_1')

    int :param intensity:   The intensity of the masking in the volume corresponding to the saliency map (Default: 0)

    int :param region_code: The code of the masked region (Default: 0)

    bool :param save_gif:   Whether or not to save the animation as a .gif file
    """

    # Initialise the figure and subplots
    fig, (ax_1, ax_2) = plt.subplots(1, 2, figsize=(10, 6))
    # Create title for the whole figure
    plt.suptitle('Comparison of '
                 '${}$ Saliency Map to Region {}\nVolume for Masking of Intensity {}/10'.format(method,
                                                                                                int(region_code),
                                                                                                intensity))
    # Point the camera at the figure
    camera = Camera(fig)

    # Plot 'frames' of cross-sections of the volumes side-by-side along an axis, and take snapshots
    for i in range(vol_1.shape[1]):
        ax_1.imshow(vol_1[0, i, :, :, 0], cmap='viridis')
        # Remove the axis ticks
        ax_1.axes.get_xaxis().set_ticks([])
        ax_1.axes.get_yaxis().set_ticks([])
        # Set the subplot title
        ax_1.set_title('Saliency Map')

        ax_2.imshow(vol_2[i, :, :], cmap='Greys_r')
        # Remove the axis ticks
        ax_2.axes.get_xaxis().set_ticks([])
        ax_2.axes.get_yaxis().set_ticks([])
        # Set the subplot title
        ax_2.set_title('Masked Region Volume')

        camera.snap()

    # Compile the animation
    animation = camera.animate()

    # Save the animation as a .gif file if necessary
    if save_gif:
        animation.save('masking_animations/{}_masking_{}_comparison_region_{}.gif'.format(method, intensity, region_code))

    plt.show()


def show_dual_animation_ageing(vol_1, patient_id, method='', age=0, save_gif=True):
    """
    Shows side-by-side animations of the saliency map of an individual's brain volume, and the corresponding brain
    volume

    np.ndarray :param vol_1:    The saliency map volume

    String :param patient_id:   The ID of the individual to be examined

    String :param method:   The method whose saliency map is being used for the individual (Default: '')

    float :param age:   The age of the individual (Default: 0)

    bool :param save_gif:   Whether or not to save the animation (Default: True)
    """

    # Fetch the corresponding MRI volume
    vol_2_file = np.load('C:/Users/Daniel/PycharmProjects/brainMasking/GCN/{}_T1w.npz'.format(patient_id))
    vol_2 = np.array([])
    for item in vol_2_file.files:
        vol_2 = vol_2_file[item]

    # Create the figure and initialise its subplots
    fig, (ax_1, ax_2) = plt.subplots(1, 2, figsize=(10, 6))
    # Create the main figure title
    plt.suptitle('Comparison of ${}$ Saliency Map to\nBrain Volume for {}y/o individual'.format(method, int(age)))
    # Point the camera to the figure
    camera = Camera(fig)

    # Iterate through the 'frames' of the animation along one axis of the volumes and take snapshots
    for i in range(vol_1.shape[1]):
        ax_1.imshow(vol_1[0, i, :, :, 0], cmap='viridis')
        # Remove the ticks from the axes
        ax_1.axes.get_xaxis().set_ticks([])
        ax_1.axes.get_yaxis().set_ticks([])
        # Set the subplot title
        ax_1.set_title('Saliency Map')

        ax_2.imshow(vol_2[i, :, :], cmap='Greys_r')
        # Remove the ticks from the axes
        ax_2.axes.get_xaxis().set_ticks([])
        ax_2.axes.get_yaxis().set_ticks([])
        # Set the subplot title
        ax_2.set_title('Brain Volume')

        camera.snap()

    # Compile the snapshots to create the animation
    animation = camera.animate()

    # Save the animation as a .gif file if necessary
    if save_gif:
        animation.save('ageing_animations/{}_ageing_comparison_{}.gif'.format(method, int(age)))
    plt.show()


'''one_method = 'DeepLIFT'
one_intensity = 10

one_volume, one_region, one_code = get_volume_masking(one_method, intensity=one_intensity)
show_dual_animation_masking(one_volume, one_region,
                            method=one_method, intensity=one_intensity,
                            region_code=one_code, save_gif=False)'''


one_method = 'LRP_3'
one_id = 'CC221527'

one_volume, one_id, one_age = get_volume_ageing(one_method, examine_id=one_id)
one_age = 34
show_dual_animation_ageing(one_volume, one_id, method=one_method, age=one_age, save_gif=False)
# show_animation(one_volume, normalise=True)
