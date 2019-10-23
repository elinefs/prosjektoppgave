import os
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold

from processData import get_array_from_nii_image

########################################################################################################################

def select_cross_validator(method, splits=5):
    """
    A function to choose the wanted method for cross-validation.
    :param method: a string with the name of the method.
    :param splits: optional, an int with number of splits is the method is K-fold.
    :return: the cross-validation method form sklearn.
    """
    if method == "leave-One-Out":
        cross_validator = LeaveOneOut()
    elif method == "K-fold":
        cross_validator = KFold(n_splits=splits)
    else:
        raise Exception("Cross-validator unknown or not implemented.")
    return cross_validator


def radiologist(method, patientPath):
    """
    A function to create the mask/ground truth we want to used based on the two radiologists.
    The mask can be from either one of the radiologists, an intersection or a union.
    :param method: a string with the choice for method.
    :param patientPath: a string with the path to the mask files.
    :return: an array with the ground truth for the patient.
    """
    maskfile_an = "Mask_an.nii"
    maskfile_shh = "Mask_shh.nii"

    if method == "an":
        maskPath = os.path.join(patientPath, maskfile_an)
        mask = get_array_from_nii_image(maskPath)

    elif method == "shh":
        maskPath = os.path.join(patientPath, maskfile_shh)
        mask = get_array_from_nii_image(maskPath)

    elif method == "intersection":
        maskPath1 = os.path.join(patientPath, maskfile_an)
        mask1 = get_array_from_nii_image(maskPath1)
        maskPath2 = os.path.join(patientPath, maskfile_shh)
        mask2 = get_array_from_nii_image(maskPath2)
        mask = np.zeros(len(mask1), dtype=np.bool)
        for i in range(len(mask1)):
            mask[i] = mask1[i] and mask2[i]

    elif method == "union":
        maskPath1 = os.path.join(patientPath, maskfile_an)
        mask1 = get_array_from_nii_image(maskPath1)
        maskPath2 = os.path.join(patientPath, maskfile_shh)
        mask2 = get_array_from_nii_image(maskPath2)
        mask = np.zeros(len(mask1), dtype=np.bool)
        for i in range(len(mask1)):
            mask[i] = mask1[i] or mask2[i]

    else:
        raise Exception("Unknown choice for mask.")

    return mask