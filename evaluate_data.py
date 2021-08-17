from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from celluloid import Camera


def visualise_correspondences_ageing(methods, p=0.1, log=False,
                                     prediction_filter='All', filter_threshold=25.0,
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

    String :param prediction_filter:    The group of test subjects to be examined, depending on the disparity between
    chronological age and predicted brain age (Default: 'All')

    float :param filter_threshold:  The threshold for the inlier/outlier categorisation in absolute percent
    (Default: 25.0)

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

        if isinstance(prediction_filter, list):
            filter = prediction_filter[m]
        else:
            filter = prediction_filter

        # Load the appropriate correspondences file
        corresp = np.load('ageing_correspondences/{}_ageing_correspondences_{}%.npy'.format(method, int(100*percent)))

        # Load the ages file
        ages = np.load('test_data/test_ages_GCN.npy', allow_pickle=True)
        # Load the predictions file
        predictions = np.load('test_data/all_age_predictions.npy', allow_pickle=True).astype(np.float32)
        # Calculate age prediction disparity
        disparity = 100 * (np.abs(ages - predictions) / ages)

        # Find individuals with relevant disparities according to the filter
        if filter == 'Outliers':
            disparity = (disparity > filter_threshold).astype(int)
        elif filter == 'Inliers':
            disparity = (disparity <= filter_threshold).astype(int)
        else:
            disparity = ages / ages

        # Filter out the non-relevant correspondences and ages
        corresp = corresp[disparity == 1, :]

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

        # Create the appropriate label
        lbl = '${}$ {}%'.format(method, int(100*percent))
        if filter != 'All':
            lbl = '{}% {} '.format(filter_threshold, filter) + lbl
        # Plot correspondence bars particular to the method and percentage
        ax.bar(x_axis - m*bar_width,
               corresp, width=bar_width,
               log=log,
               label=lbl)

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

    if (not isinstance(prediction_filter, list)) and (prediction_filter != 'All'):
        title = '{}% {}` '.format(filter_threshold, prediction_filter) + title

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
                          prediction_filter='All', filter_threshold=25.0,
                          gap=0.2, label_areas=True,
                          normalise=True, save_fig=True):
    """
    Creates a bar chart of the voxel-wise number of correspondences between the areas of the top p relevances
    for the given method, grouped such that left- and right-hemispheric structures are adjacent, so as to illustrate
    the prevalence of symmetry in the saliency mapping

    String :param method:   The method to be used in the visualisation

    float :param p: The top percentage of voxel activations to be visualised for the given method (Default: 0.1)

    bool :param log:    Whether or not to use logarithmic scale in the graph (Default: False)

    String :param prediction_filter:    The group of test subjects to be examined, depending on the disparity between
    chronological age and predicted brain age (Default: 'All')

    float :param filter_threshold:  The threshold for the inlier/outlier categorisation in absolute percent
    (Default: 25.0)

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

    # Load the ages file
    ages = np.load('test_data/test_ages_GCN.npy', allow_pickle=True)
    # Load the predictions file
    predictions = np.load('test_data/all_age_predictions.npy', allow_pickle=True).astype(np.float32)
    # Calculate age prediction disparity
    disparity = 100 * (np.abs(ages - predictions) / ages)

    # Find individuals with relevant disparities according to the filter
    if prediction_filter == 'Outliers':
        disparity = (disparity > filter_threshold).astype(int)
    elif prediction_filter == 'Inliers':
        disparity = (disparity <= filter_threshold).astype(int)
    else:
        disparity = ages / ages

    # Filter out the non-relevant correspondences and ages
    corresp = corresp[disparity == 1, :]

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

    if prediction_filter != 'All':
        title = '{}% {}` '.format(filter_threshold, prediction_filter) + title

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


def sliding_age_window(method, p=0.1,
                       prediction_filter='All', filter_threshold=25.0,
                       window_size=10, n_to_label=5,
                       normalise=True, save_fig=True,
                       show_fig=True):
    """
    Displays graphs showing the correspondences between top-p% relevance activations per brain regions for a given
    method across age brackets in the test set. A number of regions are selected as extremes for greatest/least maximum
    relevance proportion and greatest/least standard deviation in relevance proportion.

    String :param method:   The method whose heatmap correspondences to analyse

    float :param p: The percentage of top-activations being examined for the given method (Default: 0.1)

    String :param prediction_filter:    The group of test subjects to be examined, depending on the disparity between
    chronological age and predicted brain age (Default: 'All')

    float :param filter_threshold:  The threshold for the inlier/outlier categorisation in absolute percent
    (Default: 25.0)

    int :param window_size: The size (in years) of the age window over which correspondences in patients in the test set
    are averaged (Default: 10)

    int :param n_to_label:  The number of curves in each statistical extreme to examine (Default: 5)

    bool :param normalise:  Whether or not to normalise region correspondence counts by the total number of voxels per
    area (Default: True)

    bool :param save_fig:   Whether or not to save the figures automatically (Default: True)

    bool :param show_fig:   Whether or not to show the figures upon completion (Default: True)
    """

    # Load the appropriate correspondences file
    corresp = np.load('ageing_correspondences/{}_ageing_correspondences_{}%.npy'.format(method, int(100 * p)),
                      allow_pickle=True)
    # Normalise the correspondence count by the total number of voxels in each region if necessary
    if normalise:
        all_vols = np.load('ageing_correspondences/region_volumes.npy', allow_pickle=True)
        corresp = np.divide(corresp, all_vols)
    # Load the ages file
    ages = np.load('test_data/test_ages_GCN.npy', allow_pickle=True)
    # Load the predictions file
    predictions = np.load('test_data/all_age_predictions.npy', allow_pickle=True).astype(np.float32)
    # Calculate age prediction disparity
    disparity = 100 * (np.abs(ages - predictions) / ages)

    # Find individuals with relevant disparities according to the filter
    if prediction_filter == 'Outliers':
        disparity = (disparity > filter_threshold).astype(int)
    elif prediction_filter == 'Inliers':
        disparity = (disparity <= filter_threshold).astype(int)
    else:
        disparity = ages/ages

    # Filter out the non-relevant correspondences and ages
    corresp = corresp[disparity == 1, :]
    ages = ages[disparity == 1]

    # Find the extreme ages
    mx = np.max(ages)
    mn = np.min(ages)
    # print("Max age in test set: {}\nMin age in test set: {}\n".format(mx, mn))

    # The first value assumed by the lowest value in the window (at the first step)
    window_min = mn
    # The highest possible value to be assumed by the lowest window value (at the final step)
    window_max = mx - window_size
    # The number of scans that will be completed by the window -- equal to the ceiling of the range in ages
    rng = np.ceil(window_max - window_min + 1).astype(int)

    # Initialise the array containing average of the correspondences for each window pass
    corresp_per_window = np.zeros(shape=(rng, corresp.shape[1]))

    for i in range(rng):
        # Shift the beginning and end of the window to the next position
        age_min = window_min + i
        age_max = age_min + window_size

        # Isolate ages lying in the window as ones on a background of zeros
        ages_in_window = (ages >= age_min).astype(int) * (ages <= age_max).astype(int)

        # Get the correspondences of the patients whose ages lie in the window
        window_corresp = corresp[ages_in_window == 1]
        # Average the correspondences across the individuals in the window
        window_corresp = np.mean(window_corresp, axis=0)

        # Enter the window correspondences into the array
        corresp_per_window[i, :] = window_corresp

    # Create the x-axis
    x_axis = np.linspace(start=window_min, stop=window_min + rng - 1, num=rng)

    # Get the maximum correspondence per brain region over age brackets
    max_per_region = np.max(corresp_per_window, axis=0)
    # Get the positions of the (Default: 5) regions with highest max correspondence proportion
    max_n_regions = np.argsort(max_per_region)[-n_to_label:]
    # Get the positions of the (Default: 5) regions with lowest max correspondence proportion
    min_n_regions = np.argsort(max_per_region)[:n_to_label]

    # Get the standard deviation of correspondences per brain region over age brackets
    mean_per_region = np.std(corresp_per_window, axis=0)
    # Get the positions of the (Default: 5) regions with highest SD in correspondences over age brackets
    max_n_region_mean = np.argsort(mean_per_region)[-n_to_label:]
    # Get the positions of the (Default: 5) regions with lowest SD in correspondences over age brackets
    min_n_region_mean = np.argsort(mean_per_region)[:n_to_label]

    # Get the standard deviation of correspondences per brain region over age brackets
    sd_per_region = np.std(corresp_per_window, axis=0)
    # Get the positions of the (Default: 5) regions with highest SD in correspondences over age brackets
    max_n_region_sd = np.argsort(sd_per_region)[-n_to_label:]
    # Get the positions of the (Default: 5) regions with lowest SD in correspondences over age brackets
    min_n_region_sd = np.argsort(sd_per_region)[:n_to_label]

    # Get the positions of the (Default: 5) regions with highest SD in correspondences over age brackets
    max_n_region_rel_sd = np.argsort(sd_per_region/mean_per_region)[-n_to_label:]
    # Get the positions of the (Default: 5) regions with lowest SD in correspondences over age brackets
    min_n_region_rel_sd = np.argsort(sd_per_region/mean_per_region)[:n_to_label]

    # Create the plotting criteria list over which to loop
    criteria = [max_n_regions, min_n_regions,
                max_n_region_mean, min_n_region_mean,
                max_n_region_sd, min_n_region_sd,
                max_n_region_rel_sd, min_n_region_rel_sd]

    # Create the corresponding titles for plots
    titles = [
        'Maximum {} Top-{}% Correspondences per Region\nacross Age Groups via ${}$'.format(n_to_label,
                                                                                           int(100 * p),
                                                                                           method),
        'Minimum {} Top-{}% Correspondences per Region\nacross Age Groups via ${}$'.format(n_to_label,
                                                                                           int(100 * p),
                                                                                           method),
        '{} Top-{}% Correspondences with Maximum Mean\nper Region across Age Groups via ${}$'.format(n_to_label,
                                                                                                     int(100 * p),
                                                                                                     method),
        '{} Top-{}% Correspondences with Minimum Mean\nper Region across Age Groups via ${}$'.format(n_to_label,
                                                                                                     int(100 * p),
                                                                                                     method),
        '{} Top-{}% Correspondences with Maximum Standard Deviation\nper Region across Age Groups via ${}$'.format(
            n_to_label,
            int(100 * p),
            method),
        '{} Top-{}% Correspondences with Minimum Standard Deviation\nper Region across Age Groups via ${}$'.format(
            n_to_label,
            int(100 * p),
            method),
        '{} Top-{}% Correspondences with Maximum Relative Deviation\nper Region across Age Groups via ${}$'.format(
            n_to_label,
            int(100 * p),
            method),
        '{} Top-{}% Correspondences with Minimum Relative Deviation\nper Region across Age Groups via ${}$'.format(
            n_to_label,
            int(100 * p),
            method)
    ]
    # Create the corresponding save paths for plots
    save_files = [
        '{}_top{}%_max_{}_per_age'.format(method, int(100 * p), n_to_label),
        '{}_top{}%_min_{}_per_age'.format(method, int(100 * p), n_to_label),
        '{}_top{}%_max_{}_mean_per_age'.format(method, int(100 * p), n_to_label),
        '{}_top{}%_min_{}_mean_per_age'.format(method, int(100 * p), n_to_label),
        '{}_top{}%_max_{}_sd_per_age'.format(method, int(100 * p), n_to_label),
        '{}_top{}%_min_{}_sd_per_age'.format(method, int(100 * p), n_to_label),
        '{}_top{}%_max_{}_rel_sd_per_age'.format(method, int(100 * p), n_to_label),
        '{}_top{}%_min_{}_rel_sd_per_age'.format(method, int(100 * p), n_to_label)
    ]

    # Loop over the criteria to plot
    for c, criterion in enumerate(criteria):
        # Initialise the figure and axes
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Loop over the brain regions
        for j in range(corresp.shape[1]):
            # If the index satisfies the criterion, plot and label the correspondences per age bracket for this region
            if j in criterion:
                ax.plot(x_axis, corresp_per_window[:, j], label=get_region_names([j + 1])[0])
            # else:
            #     ax.plot(x_axis, corresp_per_window[:, j])

        if prediction_filter == 'Outliers':
            # Edit title accordingly
            titles[c] = '{}% MAPE Outliers` '.format(filter_threshold) + titles[c]
            # Edit the save file name accordingly
            save_files[c] = '{}%_out_'.format(int(filter_threshold)) + save_files[c]
        elif prediction_filter == 'Outliers':
            # Edit title accordingly
            titles[c] = '{}% MAPE Inliers` '.format(filter_threshold) + titles[c]
            # Edit the save file name accordingly
            save_files[c] = '{}%_in_'.format(int(filter_threshold)) + save_files[c]
        # Set the plot title
        ax.set_title(titles[c])

        # Set the x-axis label
        ax.set_xlabel('Age bracket')

        # Set the y-axis label according to whether or not we have normalised by the region volumes
        if normalise:
            ax.set_ylabel('Proportion of correspondences')
        else:
            ax.set_ylabel('Number of correspondences')

        '''# Initialise the x-axis tick labels
        labels = []
        for tick in x_axis:
            # Create label for each age bracket tick
            labels.append('{} - {}'.format(tick, tick + window_size))

        # Put ticks at all bins
        ax.set_xticks(x_axis)
        # Set the label parameters
        ax.set_xticklabels(labels, fontdict={'fontsize': 5,
                                             'fontweight': 5,
                                             'verticalalignment': 'baseline',
                                             'horizontalalignment': 'center'})
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")'''

        # Create the plot legend
        ax.legend()

        # Save the figure in the appropriate path if necessary
        if save_fig:
            plt.savefig('age_windows/' + save_files[c])

        # Show the figure if necessary
        if show_fig:
            plt.show()

        # Close the plot before continuing
        plt.close()


def age_window_animation(method, p=0.1,
                         prediction_filter='All', filter_threshold=25.0,
                         window_size=10, normalise=True,
                         gap=0.2, label_areas=True,
                         save_gif=True, show_animation=True):
    """
    Show an animated bar chart of the change in region correspondences over age brackets for a given method and
    percentage.

    String :param method:   The method whose correspondences to examine

    float :param p: The top percentage of saliency map activation correspondences to examine (Default: 0.1)

    String :param prediction_filter:    The group of test subjects to be examined, depending on the disparity between
    chronological age and predicted brain age (Default: 'All')

    float :param filter_threshold:  The threshold for the inlier/outlier categorisation in absolute percent
    (Default: 25.0)

    int :param window_size: The number of years to include in the age window for each 'scan' of correspondences
    (Default: 10)

    bool :param normalise:  Whether or not to normalise the region correspondence count by the total number of voxels
    per region (Default: True)

    float :param gap:   The size of the gap between bars in the histogram (Default: 0.2)

    bool :param label_areas:    Whether or not to label the areas of the brain on the x-axis with their names (Default:
    True)

    bool :param save_gif:   Whether or not to save the compiled animation (Default: True)

    bool :param show_animation: Whether or not to show the compiled animation (Default: True)
    """

    # Initialise bar width according to the gap
    bar_width = 1.0 - gap
    # Load the appropriate correspondences file
    corresp = np.load('ageing_correspondences/{}_ageing_correspondences_{}%.npy'.format(method, int(100 * p)),
                      allow_pickle=True)
    # Normalise the correspondence count by the total number of voxels in each region if necessary
    if normalise:
        all_vols = np.load('ageing_correspondences/region_volumes.npy', allow_pickle=True)
        corresp = np.divide(corresp, all_vols)
    # Load the ages file
    ages = np.load('test_data/test_ages_GCN.npy', allow_pickle=True)
    # Load the predictions file
    predictions = np.load('test_data/all_age_predictions.npy', allow_pickle=True).astype(np.float32)
    # Calculate age prediction disparity
    disparity = 100 * (np.abs(ages - predictions)/ages)

    # Create the plot title
    title = 'Top-{}% Correspondences per Region over {}-year Age Windows via ${}$'.format(int(100 * p),
                                                                                          window_size,
                                                                                          method)
    # Create the save file name
    save_path = '{}_top-{}%_age_brackets.gif'.format(method, int(100 * p))

    # Find individuals with relevant disparities according to the filter
    if prediction_filter == 'Outliers':
        disparity = (disparity > filter_threshold).astype(int)

        # Edit title accordingly
        title = '{}% MAPE Outliers` '.format(filter_threshold) + title
        # Edit the save file name accordingly
        save_path = '{}%_out_'.format(int(filter_threshold)) + save_path
    elif prediction_filter == 'Inliers':
        disparity = (disparity <= filter_threshold).astype(int)

        # Edit title accordingly
        title = '{}% MAPE Inliers` '.format(filter_threshold) + title
        # Edit the save file name accordingly
        save_path = '{}%_in_'.format(int(filter_threshold)) + save_path
    else:
        disparity = ages/ages

    # Filter out the non-relevant correspondences and ages
    corresp = corresp[disparity == 1, :]
    ages = ages[disparity == 1]

    # Find the maximum region correspondence
    mx_prop = (np.max(np.mean(corresp, axis=0)) + np.max(corresp))/2

    # Find the extreme ages
    mx = np.max(ages)
    mn = np.min(ages)
    # print("Max age in test set: {}\nMin age in test set: {}\n".format(mx, mn))

    # The first value assumed by the lowest value in the window (at the first step)
    window_min = mn
    # The highest possible value to be assumed by the lowest window value (at the final step)
    window_max = mx - window_size
    # The number of scans that will be completed by the window -- equal to the ceiling of the range in ages
    rng = np.ceil(window_max - window_min + 1).astype(int)

    # Initialise the figure and subplots
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    # Create the x-axis
    x_axis = np.linspace(start=1, stop=corresp.shape[1], num=corresp.shape[1])
    # Create title for the whole figure
    plt.suptitle(title)
    ax.text(0, mx_prop, '')
    # Point the camera at the figure
    camera = Camera(fig)

    for i in range(rng):
        # Shift the beginning and end of the window to the next position
        age_min = window_min + i
        age_max = age_min + window_size

        # Isolate ages lying in the window as ones on a background of zeros
        ages_in_window = (ages >= age_min).astype(int) * (ages <= age_max).astype(int)

        # Count how many individuals lie in the given age bracket
        window_count = np.sum(ages_in_window)
        # print("\tNumber of individuals in the test set lying within the age window: {}".format(window_count))

        # Get the correspondences of the patients whose ages lie in the window
        window_corresp = corresp[ages_in_window == 1]
        # Average the correspondences across the individuals in the window
        window_corresp = np.mean(window_corresp, axis=0)

        # Plot 'frames' of bar chart for each age bracket
        ax.bar(x_axis,
               window_corresp,
               width=bar_width,
               color='b')
        # Show the age bracket
        ax.text(0, mx_prop, '{}y - {}y with {} individuals'.format(age_min, age_max, window_count))
        # Set the axis labels
        ax.set_xlabel('Brain Region')
        ax.set_ylabel('Proportion of Corresponding Voxels')
        # Set the y-axis limits
        ax.set_ylim(0, 1.1*mx_prop)

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

        # Take snapshot of the barchart
        camera.snap()
        ax.text(0, mx_prop, '')

    # Compile the animation
    animation = camera.animate(blit=False, interval=900)

    # Save the animation as a .gif file if necessary
    if save_gif:
        animation.save('age_windows/' + save_path)

    # Display animation if necessary
    if show_animation:
        plt.show()

    # Close pyplot after the saving and/or display to clear memory
    plt.close()


def prediction_disparities(show_disparity=False, save_fig=False):
    """
    Creates and (optionally) saves plot of the chronological age and the predicted age of test set subjects, or the
    disparity between the two

    bool :param show_disparity: Whether to show the disparity  measures, as opposed to the ages and predictions
    themselves (Default: False)

    bool :param save_fig:   Whether or not to save the figure automatically (Default: False)
    """

    # Load the ages and predictions files
    ages = np.load('test_data/test_ages_GCN.npy', allow_pickle=True)
    predictions = np.load('test_data/all_age_predictions.npy', allow_pickle=True).astype(np.float32)

    # Create the x-axis
    x_axis = np.linspace(0, len(ages), len(ages))

    # Plot only the differences between the ages and predictions if necessary
    if show_disparity:
        # Calculate the disparity (absolute, in years)
        disparity = np.abs(ages - predictions)
        # Calculate the percentage absolute disparity
        percent_disparity = 100*(disparity/ages)

        # Find the maximum of all the disparities
        mx_1 = np.max(disparity)
        mx_2 = np.max(percent_disparity)
        mx = max(mx_1, mx_2)

        # Plot the disparity curves and label them appropriately
        plt.plot(x_axis, disparity,
                 label='Absolute Difference\n(Mean = {}y)'.format(np.round(np.mean(disparity), 2)))
        plt.plot(x_axis, percent_disparity,
                 label='Absolute Percentage Difference\n(Mean: {}%)'.format(np.round(np.mean(percent_disparity), 2)))
        # Create the appropriate plot title
        plt.title('Disparity Between Chronological Age and Predicted Brain Age')

        # Create the appropriate save title
        save_title = 'age_prediction_disparity'
    # Plot only the ages and predictions otherwise
    else:
        # Find the maximum of all the ages (predicted or actual)
        mx_1 = np.max(ages)
        mx_2 = np.max(predictions)
        mx = max(mx_1, mx_2)

        # Plot the age and prediction curves
        plt.plot(x_axis, ages, label='Chronological Age')
        plt.plot(x_axis, predictions, label='Predicted Brain Age')
        # Create the appropriate plot title
        plt.title('Chronological Ages vs. Predicted Brain Ages in Test Set')
        # Label the y-axis
        plt.ylabel('Years')

        # Create the appropriate save title
        save_title = 'age_vs_prediction'

    # Create the legend
    plt.legend()
    # Set the y-axis limits
    plt.ylim(0, 1.1*mx)
    # Label the x-axis
    plt.xlabel('Patient in Test Set')

    # Save the figure automatically if necessary
    if save_fig:
        plt.savefig('test_data/{}'.format(save_title))

    # Show the image
    plt.show()


def correlation_matrix(method, p, save_fig=True, show_fig=True):
    """
    Creates and displays the correlation matrix for the region correspondences file of the given method and top-p%
    activations

    String :param method:   The method whose correspondence matrix to examine

    float :param p: The percentage of top voxel activations to consider in the correspondences from the saliency maps of
    the given method

    bool :param save_fig:   Whether or not to save the generated figure (Default: True)

    bool :param show_fig:   Whether or not to show the generated figure (Default: True)
    """

    # Load correspondence file -- shape: (132, 102) (patients, regions)
    corresp = np.load('ageing_correspondences/{}_ageing_correspondences_{}%.npy'.format(method, int(100*p)),
                      allow_pickle=True)
    # Load all region volumes
    vols = np.load('ageing_correspondences/region_volumes.npy', allow_pickle=True)
    # Normalise the correspondences by the total region volumes
    corresp = corresp/vols

    # Calculate the correlation matrix for regions across individuals in the test set
    # 'rowvar' set to False to allow regions to be considered variables and individuals the observations thereof
    # (rows vs columns)
    correlation_mtx = np.corrcoef(corresp, rowvar=False)

    # create the figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(11, 8))
    # Create axis
    axis = np.linspace(start=1, stop=correlation_mtx.shape[0], num=correlation_mtx.shape[0])

    # Fetch region names
    labels = get_region_names(axis)
    # Put ticks at all points
    ax.set_xticks(axis - 1)
    ax.set_yticks(axis - 1)
    # Puth the x-axis ticks at the top of the image
    ax.xaxis.tick_top()
    # Set the label parameters
    ax.set_xticklabels(labels, fontdict={'fontsize': 5,
                                         'fontweight': 5,
                                         'verticalalignment': 'center',
                                         'horizontalalignment': 'center'})
    ax.set_yticklabels(labels, fontdict={'fontsize': 5,
                                         'fontweight': 5,
                                         'verticalalignment': 'center',
                                         'horizontalalignment': 'center'})
    plt.setp(ax.get_xticklabels(), rotation=90, ha="left", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", rotation_mode="anchor")

    # Display the matrix on axes
    plt.imshow(correlation_mtx, cmap='Reds')
    # Create the title, at the bottom of the figure
    plt.title('${}$ Top-{}% Region Correspondences Correlation Matrix'.format(method, int(100*p)), y=-0.075)
    # Display colour bar
    plt.colorbar()

    # Show the figure if necessary
    if show_fig:
        plt.show()

    if save_fig:
        plt.savefig('ageing_correlations/{}_{}%_corr_mtx'.format(method, int(100*p)))


def compare_masking_heatmap(vol, index):
    region_vol = get_region_volume(index)

    comparison = (vol/np.max(vol)) - region_vol

    return comparison


def masking_analysis(method):
    print('Not finished yet')


'''for mthd in ['DeepLIFT', 'LRP_1', 'LRP_2', 'LRP_3']:
    for pct in [0.1, 0.05, 0.02, 0.01]:
        # sliding_age_window(mthd, p=pct, n_to_label=5, save_fig=True, show_fig=False)
        # age_window_animation(mthd, p=pct, save_gif=True, show_animation=False)
        # correlation_matrix(mthd, pct, save_fig=True, show_fig=False)
        print('\rSaved images for {} {}%'.format(mthd, int(100 * pct)), end='')'''

'''correlation_matrix('DeepLIFT', 0.1, save_fig=False)'''

'''sliding_age_window('DeepLIFT', p=0.01, prediction_filter='Inliers', filter_threshold=25.0, save_fig=False)'''

'''age_window_animation('DeepLIFT', p=0.01, prediction_filter='Outliers', filter_threshold=15.0, save_gif=False)'''

'''prediction_disparities(False)'''

'''visualise_correspondences_masking('LRP_1', p=0.01, label_areas=True, save_fig=False)'''


'''visualise_lr_symmetry('LRP_1', p=0.01, log=False,
                      prediction_filter='Outliers', filter_threshold=25.0,
                      gap=0.2, label_areas=True,
                      normalise=True, save_fig=False)'''


'''visualise_correspondences_ageing(['LRP_1'], p=[0.01], log=False,
                                 prediction_filter=['All'], filter_threshold=25.0,
                                 average=True, dividing_line=False,
                                 gap=0.2, use_symmetry=False,
                                 label_areas=True, normalise=True,
                                 save_fig=False)'''
