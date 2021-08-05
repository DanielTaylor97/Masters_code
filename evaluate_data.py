from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def visualise_correspondences_ageing(methods, p=0.1, log=False,
                                     average=True, dividing_line=True,
                                     gap=0.2, use_symmetry=False,
                                     label_areas=False, normalise=True,
                                     save_fig=True):
    """
    Creates a bar chart of the voxel-wise number of correspondences between
    the areas of the top p relevances for the given methods

    list of strings :param methods: Name(s) of method(s) to be examined

    float :param p: Top percentage of voxel relevances to be examined -- between 0 and 1; dependent on available files
    (Default: 0.1)

    bool :param log:    Whether or not to use log scale (Default: False)

    bool :param average:    Whether or not to take the average number per region over all patients. If False, a random
    member is examined (Default: True)

    bool :param dividing_line:  Whether or not to include a halfway dividing line for the region symmetry
    (Default: True)

    float :param gap:   Width of the gap separating bars or bar clusters

    bool :param use_symmetry:   Whether or not to examine only the one side of the region correspondences
    (Default: False)

    bool :param label_areas:    Whether or not to use the names of the areas in the x-tick labels (Default: False)

    bool :param normalise:  Whether or not to divide through by the number of voxels per brain region (Default: True)

    bool :param save_fig:   Whether or not to save the bar chart automatically. The size of the graphic will
    automatically be made much larger than if the image is not to be saved, in order to fit labels in (Default: True)
    """

    # Default max and min values for bars; updated in the presence of more extremes
    y_max = 0
    y_min = 1e3

    # Adjust the bar width to fit multiple bars into one bin, leaving a gap between groups of bars
    n_methods = len(methods)
    bar_width = (1-gap)/n_methods

    # If the figure is to be saved, we make it larger to fit the axis labels in
    if save_fig:
        fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Initialise the arrays for correspondences and for the x-axis
    corresp = []
    x_axis = []

    for m, method in enumerate(methods):
        # If a list of percentages is given they must correspond to the listed methods, or all to the same single method
        if isinstance(p, list):
            percent = p[m]
        else:
            percent = p
        # Load the appropriate correspondences file
        corresp = np.load('ageing_correspondences/{}_ageing_correspondences_{}%.npy'.format(method, int(100*percent)))

        # Either take the average of the correspondences per area, or sample the correspondences for a random ID
        if average:
            corresp = np.mean(corresp, axis=0)
        else:
            individual = np.random.randint(low=0, high=corresp.shape[0])
            corresp = corresp[individual]

        # Normalise by the total number of voxels per region
        if normalise:
            region_volumes = np.load('ageing_correspondences/region_volumes.npy')
            corresp = np.divide(corresp, region_volumes)

        # Visualise only the right-hemisphere correspondences, making use of the bilateral symmetry
        if use_symmetry:
            corresp = corresp[:int(0.5*corresp.shape[0])]

        # Update upper and lower bounds
        if np.max(corresp) > y_max:
            y_max = np.max(corresp)
        if np.min(corresp) < y_min:
            y_min = np.min(corresp)

        # Create x-axis
        x_axis = np.linspace(start=1, stop=corresp.shape[0], num=corresp.shape[0])

        # Plot correspondence bars particular to the method and percentage
        ax.bar(x_axis - m*bar_width,
               corresp, width=bar_width,
               log=log,
               label='${}$ {}%'.format(method, int(100*percent)))

    # Create title string according to the parameterisation of the figure
    if isinstance(p, list):
        title = 'of Top-p% Highlighted Voxels per Brain Region'
    else:
        title = 'of Top-{}% Highlighted Voxels per Brain Region'.format(int(100 * p))

    if normalise:
        title = 'Proportion ' + title
    else:
        title = 'Number ' + title

    if average:
        title = 'Average ' + title

    # Set the figure title
    ax.set_title(title, fontdict={'fontsize': 10,
                                  'fontweight': 10,
                                  'color': 'Black',
                                  'verticalalignment': 'baseline',
                                  'horizontalalignment': 'center'})
    # Set the labels for the axes
    ax.set_xlabel('Brain Region')
    ax.set_ylabel('Number of Corresponding Voxels')

    # Create a line showing the division between right- and left-hemispheric regions
    if dividing_line:
        halfway = np.ones(2) * (0.5 * corresp.shape[0] + gap)
        length_div_line = np.linspace(start=0, stop=5 * y_max, num=2)
        ax.plot(halfway, length_div_line, linewidth=1.0, color='red', label='Region Midline\n(L/R Symmetry)')

        # Set y-limits for aesthetics
        if log:
            ax.set_ylim(y_min/2.0, 2.5*y_max)
        else:
            ax.set_ylim(0, 1.1*y_max)

    # Label areas with full region names
    if label_areas:
        # Fetch region names
        labels = get_region_names(x_axis)
        # Put ticks at all bins
        ax.set_xticks(x_axis)
        # Set the label parameters
        ax.set_xticklabels(labels, fontdict={'fontsize': 5,
                                             'fontweight': 5,
                                             'verticalalignment': 'baseline',
                                             'horizontalalignment': 'center'})
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.legend()

    # Save the figure automatically. File name not automated as it is difficult
    if save_fig:
        plt.savefig('ageing_graphs/age_correspondences_figure')
    plt.show()


def visualise_lr_symmetry(method, p=0.1, log=False,
                          gap=0.2, label_areas=True,
                          normalise=True, save_fig=True):
    """
    Creates a bar chart of the voxel-wise number of correspondences between the areas of the top p relevances
    for the given method, grouped such that left- and right-hemispheric structures are adjacent, so as to illustrate
    the prevalence of symmetry in the saliency mapping

    String :param method:   The method to be used in the visualisation

    float :param p: The top percentage of voxel activations to be visualised for the given method (Default: 0.1)

    bool :param log:    Whether or not to use logarithmic scale in the graph (Default: False)

    float :param gap:   Size of the gap between bars (Default: 0.2)

    bool :param label_areas:    Whether to label the areas with their names as opposed to the area code (Default: True)

    bool :param normalise:  Whether or not to normalise the number of activations per region by the total number of
    activations in that region (Default: True)

    bool :param save_fig:   Whether or not to save the figure upon creating it. If True, the figure will be generated
    much larger than if False (Default: True)
    """

    # Adjust the bar width to fit two bars into one bin, leaving a gap between pairs of bars
    bar_width = (1-gap)/2

    # If the figure is to be saved, it is made much bigger so as to fit the axis labels
    if save_fig:
        fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Load appropriate correspondences file
    corresp = np.load('ageing_correspondences/{}_ageing_correspondences_{}%.npy'.format(method, int(100 * p)))
    # Take the mean across individuals
    corresp = np.mean(corresp, axis=0)

    # Normalise by the total number of voxels per region if necessary
    if normalise:
        region_volumes = np.load('ageing_correspondences/region_volumes.npy')
        corresp = np.divide(corresp, region_volumes)

    # Split correspondences into right- and left-hemispheric regions
    corresp_1 = corresp[:int(0.5*corresp.shape[0])]
    corresp_2 = corresp[int(0.5 * corresp.shape[0]):]

    # Set upper and lower bounds on y-values
    y_max = np.max(corresp)
    y_min = np.min(corresp)

    # Create x-axis
    x_axis = np.linspace(start=1, stop=corresp_1.shape[0], num=corresp_1.shape[0])

    # Plot right-hemispheric correspondences
    ax.bar(x_axis - bar_width,
           corresp_1, width=bar_width,
           log=log,
           label='${}$ {}% RH'.format(method, int(100 * p)))
    # Plot left-hemispheric correspondences
    ax.bar(x_axis - 2*bar_width,
           corresp_2, width=bar_width,
           log=log,
           label='${}$ {}% LH'.format(method, int(100 * p)))

    # Create plot title according to method, percentage and quantification
    title = 'of Top-{}% Highlighted Voxels per Brain Region'.format(int(100 * p))
    if normalise:
        title = 'Proportion ' + title
    else:
        title = 'Number ' + title
    title = 'Average ' + title

    # Set title
    ax.set_title(title, fontdict={'fontsize': 10,
                                  'fontweight': 10,
                                  'color': 'Black',
                                  'verticalalignment': 'baseline',
                                  'horizontalalignment': 'center'})

    # Set labels for axes
    ax.set_xlabel('Brain Region')
    ax.set_ylabel('Number of Corresponding Voxels')

    # Set y-limits according to figure scale
    if log:
        ax.set_ylim(y_min/2.0, 2.5*y_max)
    else:
        ax.set_ylim(0, 1.1*y_max)

    # Label areas with region names if necessary
    if label_areas:
        # Fetch region names
        labels = get_region_names(x_axis, distinction=False)
        # Put ticks at every bin
        ax.set_xticks(x_axis)
        # Parameterise labels
        ax.set_xticklabels(labels, fontdict={'fontsize': 5,
                                             'fontweight': 5,
                                             'verticalalignment': 'baseline',
                                             'horizontalalignment': 'center'})
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.legend()

    # Save figure automatically according to method and percentage
    if save_fig:
        plt.savefig('ageing_graphs/{}_lr_symmetry_top_{}%'.format(method, int(100 * p)))
    plt.show()


def visualise_correspondences_masking(method, p=0.1,
                                      index=-1, log=False,
                                      gap=0.2, label_areas=False,
                                      normalise=True, save_fig=True):
    """
    Creates a bar chart of the voxel-wise number of correspondences between the areas of the top p relevances for the
    given method, illustrating simultaneously the masked area

    String :param method:   The method to be used in the visualisation

    float :param p: The top percentage of voxel activations to be visualised for the given method (Default: 0.1)

    int :param index:   Optional index from which to choose a patient. If in the range of 0 to 131, the correspondences
    will be shown for the subject at that index in the test set (Default: -1)

    bool :param log:    Whether or not to use logarithmic scale inthe graph (Default: False)

    float :param gap:   Size of the gap between bars (Default: 0.2)

    bool :param label_areas:    Whether or not to label areas with their names as opposed to the area code
    (Default: False)

    bool :param normalise:  Whether or not to normalise the number of voxel activations per area by the total number of
    voxels in that area (Default: True)

    bool :param save_fig:   Whether or not to save the figure upon its creation. If True, the generated figure will be
    significantly large then if False (Default: True)
    """

    # Set bar width to consider the gap
    bar_width = 1 - gap

    # If the figure is to be saved automatically, it is generated much larger to fit the axis labels
    if save_fig:
        fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Load the appropriate correspondences
    corresp = np.load('masking_correspondences/{}_masking_correspondences_{}%.npy'.format(method, int(100 * p)))

    # Fetch regions and intensities corresponding to maskings for heatmaps
    all_regions = np.load('test_data/test_regions_masking.npy', allow_pickle=True)
    all_intensities = np.load('test_data/test_intensities_masking.npy', allow_pickle=True)

    # Find an individual at the given index, or find a random individual, and fetch the region and masking intensity
    if index >= 0:
        corresp = corresp[index]
        masked_region = all_regions[index]
        intensity = all_intensities[index]
    else:
        individual = np.random.randint(low=0, high=corresp.shape[0])
        corresp = corresp[individual]
        masked_region = all_regions[individual]
        intensity = all_intensities[individual]

    # Print the region that is masked, if there is one
    if masked_region == 0:
        print('No Region Masked')
    else:
        print('Masked Region: {}. {}'.format(masked_region, get_region_names([masked_region])))

    # Normalise by the total number of voxels per region, if necessary
    if normalise:
        # Load region volumes
        region_volumes = np.load('masking_correspondences/region_volumes.npy')
        corresp = np.divide(corresp, region_volumes)

    # Create x-axis
    x_axis = np.linspace(start=1, stop=corresp.shape[0], num=corresp.shape[0])

    # Show the area that is masked on the bar chart
    one_hot_region = np.zeros_like(corresp)
    if masked_region != 0:
        # Create one red bar to indicate which area is masked, slightly higher than any others so that it is not hidden
        one_hot_region[int(masked_region - 1)] = np.max(corresp) + 0.05

    # Plot the correspondences
    ax.bar(x_axis, one_hot_region,
           color='r', width=bar_width, log=log,
           label='Masked region {}/10'.format(intensity))
    # Plot the bar corresponding to the masked area, if there is one
    ax.bar(x_axis, corresp,
           color='b', width=bar_width,
           log=log, label='${}$ {}%'.format(method, int(100 * p)))

    # Create the plot title according to the method, percentage and quantification
    title = 'of Top-{}% Highlighted Voxels per Brain Region'.format(int(100 * p))
    if normalise:
        title = 'Proportion ' + title
    else:
        title = 'Number ' + title

    # Set the title
    ax.set_title(title, fontdict={'fontsize': 10,
                                  'fontweight': 10,
                                  'color': 'Black',
                                  'verticalalignment': 'baseline',
                                  'horizontalalignment': 'center'})

    # Set the axis labels
    ax.set_xlabel('Brain Region')
    ax.set_ylabel('Number of Corresponding Voxels')

    # Label the areas with their region names if necessary
    if label_areas:
        # Fetch region names
        labels = get_region_names(x_axis)
        # Put ticks at each bin
        ax.set_xticks(x_axis)
        # Format tick labels
        ax.set_xticklabels(labels, fontdict={'fontsize': 5,
                                             'fontweight': 5,
                                             'verticalalignment': 'baseline',
                                             'horizontalalignment': 'center'})
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.legend()

    # Save the figure name automatically according to the method, percentage and masking intensity
    if save_fig:
        plt.savefig('masking_graphs/{}_top_{}%_masking_{}'.format(method, int(100 * p), intensity))
    plt.show()


def get_region_names(labels, distinction=True):
    """
    Returns the names corresponding to the given region codes

    list of ints :param labels:  Region codes for which the names are sought

    bool :param distinction:    Whether or not to make the distinction between left- and right-hemispheric structures

    list of Strings :return region_names:    The list of region names corresponding to the given codes
    """

    # Read all data from the .csv file
    all_info = pd.read_csv('CerebrA_LabelDetails.csv')
    # Extract all label names
    names = all_info['Label Name']

    # initialise the names array and the position variable
    region_names = []
    pos = 0

    # Get each region name
    for lbl in labels:
        # Initialise the name string
        region_name = ''

        # Fetch labels for right-hemispheric areas
        if lbl <= 51:
            if distinction:
                # Show that this is the right-hemispheric structure
                region_name = ' (R)'
            # Fetch the position corresponding to the name
            pos = all_info.index[all_info['RH Label'] == lbl].tolist()[0]
        # Fetch labels for right-hemispheric areas
        elif lbl <= 102:
            if distinction:
                # Show that this is the right-hemispheric structure
                region_name = ' (L)'
            # Fetch the position corresponding to the name
            pos = all_info.index[all_info['LH Labels'] == lbl].tolist()[0]
        else:
            # Invalid region codes return empty strings
            print('Invalid region {}'.format(lbl))

        # Update region name and add it to the list
        region_name = names[pos] + region_name
        region_names.append(region_name)

    return region_names


def get_region_volume(region):
    """
    Finds and returns the volume of the brain region corresponding to a given region code, from the atlas

    int :param region:  The region code for the desired volume

    numpy.ndarray :return region_vol: The 3D volume of the region requested. Activations are 0 for outside the region
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


def compare_masking_heatmap(vol, index):
    region_vol = get_region_volume(index)

    comparison = (vol/np.max(vol)) - region_vol

    return comparison


def masking_analysis(method):
    print('Not finished yet')


'''visualise_correspondences_masking('LRP_1', p=0.01, label_areas=True, save_fig=False)'''


'''visualise_lr_symmetry('LRP_1', p=0.01, log=False,
                      gap=0.2, label_areas=True,
                      normalise=True, save_fig=False)'''


'''visualise_correspondences_ageing(['DeepLIFT', 'DeepLIFT'], p=[0.1, 0.05], log=False,
                                 average=True, dividing_line=False,
                                 gap=0.2, use_symmetry=True,
                                 label_areas=True, normalise=True,
                                 save_fig=True)'''
