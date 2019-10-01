"""
Functions to load the contours from the nifit files and correct the spatial
information so that they align with the corresponding image
"""
import SimpleITK as sitk

from PhDProject.Build_Dataset.special_patients import invert_mask
from PhDProject.Build_Dataset.util import patient_name


def load_contour(path_to_nifit_file, reference_image):
    """
    Load the contour from the nifti file,

    For some patients -> special_patients, the order of the slices needs to
    be inverted in order to match the corresponding (T2) image

    As images are provided as dicom files and masks as nifti, different
    coordinate systems are used. To compensate for this, the spatial
    information from the image is used for the mask
    """
    mask = sitk.ReadImage(path_to_nifit_file)

    # invert slice order if needed
    if patient_name(path_to_nifit_file) in invert_mask:
        mask = sitk.Flip(mask, [False, False, True])

    # correct spatial properies
    mask.CopyInformation(reference_image)

    return mask