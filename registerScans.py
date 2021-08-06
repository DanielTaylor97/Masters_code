#############################################
# REGISTRATION TAKES SKULL-STRIPPED VOLUMES #
# NEXT STEP IS NORMALISATION                #
#############################################

from getIDs import GetIDs
from nipype.interfaces.fsl import FLIRT, FNIRT


# Fetch the information on the individuals from the .csv file
info = GetIDs()
ids, ages, count = info.getAll()
'''ids = [110174, 120376, 320651, 321107, 321595,
       410129, 410447, 420143, 420167, 420356, 420454,
       510163, 510480, 510534, 510609, 520477, 520585,
       610058, 610210, 610285, 610469, 610496, 610575,
       620090, 620152, 620454, 620919, 621184, 621199,
       621284, 710664, 711128, 711245, 720238, 721114]'''

'''ids = [510534, 520477,
       620919]'''


def transform_volume(patient_id, linear=True):
    """
    Register brain MRI volumes to an MNI standard atlas

    String :param patient_id:   The ID of the patient to be registered

    bool :param linear: Whether or not to use the linear registration tool, as opposed to the nonlinear tool
    (default: True)
    """

    # Update the user on which individual's volume is being transformed
    print('\rTransform begun on {}.'.format(patient_id), end='')

    # Create the path for the skull-stripped volume to be registered
    file_path = 'stripped/{}_str.nii.gz'.format(patient_id)
    # file_path = 'reregistered/{}_FLIRT_file.nii.gz'.format(patient_id)
    # Create the reference file path
    ref_path = 'brainAtlases/stripped_standard.nii.gz'

    # Use FLIRT
    if linear:
        # Initialise FLIRT tool
        flt = FLIRT()
        # Create the registered file
        new_file = flt.run(reference=ref_path,
                           in_file=file_path,
                           output_type='NIFTI_GZ',
                           out_file='reregistered/{}_FLIRT_file.nii.gz'.format(patient_id),
                           dof=12)
        # Update the user on registration progress
        print('\rTransform completed on {}.\n'.format(patient_id))
    # Use FNIRT
    else:
        # Initialise FNIRT tool
        fnt = FNIRT()
        # Create the registered file
        new_file = fnt.run(ref_file=ref_path,
                           in_file=file_path,
                           output_type='NIFTI_GZ',
                           warped_file='registered/{}_warped_file.nii.gz'.format(patient_id),
                           log_file='registered/{}_log_file'.format(patient_id))
        # Update the user on registration progress
        print('Transform completed on {}.\n'.format(patient_id))


# Loop over IDs and register the corresponding volume to the atlas
for one_id in ids:
    transform_volume(one_id)
    # transform_volume('CC{}'.format(id))
