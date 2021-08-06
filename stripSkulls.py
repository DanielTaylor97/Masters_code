####################################################
# SKULL-STRIPPING IS THE FIRST PRE-PROCESSING STEP #
# NEXT STEP IS REGISTRATION                        #
####################################################

from __future__ import division
from os import path
from nipype.interfaces.fsl import BET
#S.M. Smith. Fast robust automated brain extraction. Human Brain Mapping, 17(3):143-155, November 2002.
import time
from getIDs import GetIDs

######
#print(os.getenv('FSLDIR'))
#print(fsl.Info().version())
######

info = GetIDs()

ids, ages, count = info.getAll()


def strip_single():
    """
    Strip the standard MNI volume
    """

    # Start skull-strip and timer
    print('Started skull-strip')
    t0 = time.time()
    # Define in and out file paths
    i_f = 'brainAtlases/adjusted_standard.nii'
    o_f = 'brainAtlases/stripped_standard.nii'

    # Initialise skull-stripper tool
    skullstrip = BET(in_file=i_f, out_file=o_f, frac=0.5, output_type='NIFTI_GZ')

    # Executing the skull-strip gives a ValueError, but it can be ignored and the file is saved perfectly fine
    try:
        result = skullstrip.run()
    except ValueError:
        pass

    # End timer
    t1 = time.time()
    # Print elapsed time
    print('Total time elapsed: {}'.format(t1 - t0))


def strip_batch():
    """
    Strip an entire batch of skulls by index of ID. Doing all at once is extremely taxing on RAM for some reason;
    batches of 60 work well
    """

    # Start the timer
    t0 = time.time()

    print('Starting on next batch.\n')

    # Loop over IDs
    for i in range(len(ids)):
        # Define in and out file paths
        i_f = 'anat/sub-{}/anat/sub-{}_T1w.nii.gz'.format(ids[i], ids[i])
        o_f = 'stripped/{}_str.nii.gz'.format(ids[i])

        # Initialise skull-stripper
        skullstrip = BET(in_file=i_f, out_file=o_f, frac=0.2, output_type='NIFTI_GZ')
        # Update user on progress
        print('\r{} of {} being stripped'.format(i + 1 - 0, len(ids)), end='')

        # Executing the skull-strip gives a ValueError, but it can be ignored and the file is saved perfectly fine
        try:
            result = skullstrip.run()
        except ValueError:
            pass

    # End timer
    t1 = time.time()
    # Show elapsed time
    print('\n\nTotal time elapsed: {}'.format(t1 - t0))


def check_all_stripped():
    """
    Loop through all IDs and check that a skull-stripped file exists for each of them
    """

    # Loop through IDs
    for one_id in ids:
        # Create relevant path name
        path_str = 'stripped/{}_str.nii.gz'.format(one_id)

        # Check that path exists
        if not path.exists(path_str):
            # If not, notify the user
            print('{} not skull stripped'.format(one_id))
    # Notify the user when the process is finished
    print('All IDs checked')


# strip_single()

strip_batch()

# check_all_stripped()
