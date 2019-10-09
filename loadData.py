import os
import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt
import sklearn
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import dask.array as da
###########################################################################################################
# Functions


def get_array_from_nii_image(path):
    """
    A function to read an image file and convert the image data to a 1D array.
    :param path: a string with the path to the image.
    :return: 1D array with image data.
    """
    image = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(image)
    array = array.flatten()
    return array


def select_cross_validator(method):
    """
    A function to choose the wanted method for cross-validation.
    :param method: a string with the name of the method.
    :return: the cross-validation method form sklearn.
    """
    if method == "leave-One-Out":
        cross_validator = LeaveOneOut()
    elif method == "K-fold":
        cross_validator = KFold(n_splits=5)
    else:
        raise Exception("Cross-validator unknown or not implemented.")
    return cross_validator


def normalization(data):
    """
    A function to normalize the data with mean 0 and std 1.
    :param data: a matrix or a list with the data you want to normalize.
    :return: a matrix with normalized data.
    """
    if len(np.shape(data)) == 1:
        data = np.expand_dims(data, axis=1)
    scaler = StandardScaler(with_mean=False, with_std=False)
    scaler.fit(data)
    normData = scaler.transform(data)

    return normData


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


def GetPatients(directoryPath):
    """
    A function to get the paths to the different patients from a directory.
    :param directoryPath: a string with the path to the directory with the patients.
    :return: a list with the paths to the patients.
    """
    patientPaths = []
    for entry in os.listdir(basepath):
        patientPath = os.path.join(basepath, entry)
        if os.path.isdir(patientPath):
            patientPaths.append(patientPath)
    return patientPaths


def buildDataset(patientPaths, scans, maskchoice, patientIndex):
    """
    A function to build the data matrix and the list with ground truth.
    :param patientPaths: a list with strings that are paths to the patient files.
    :param scans: a matrix containing the different scan types.
    :param maskchoice: a string with the method for creating the ground truth.
    :param patientIndex: empty matrix to store start and stop index for each patient.
    :return: return a matrix with all the image data and a list with the ground truth.
    """
    dataMatrixList = []
    groundTruthList = []
    for patientNo, patient in enumerate(patientPaths):

        mask = radiologist(maskchoice, patient)
        groundTruthList.append(mask)
        numberOfVoxels = len(mask)
        patientIndex[patientNo][0] = patientIndex[patientNo-1][1]
        patientIndex[patientNo][1] = patientIndex[patientNo][0] + numberOfVoxels
        patientData = []

        for type in scans:
            scandata = np.zeros((numberOfVoxels, len(type)), dtype=np.single)
            for index, scan in enumerate(type):
                filename = scan + ".nii"
                path = os.path.join(patient, filename)
                image = get_array_from_nii_image(path)
                scandata[:, index] = image
            scandata = normalization(scandata)
            patientData.append(scandata)

        patientMatrix = np.concatenate(patientData, axis=1)
        dataMatrixList.append(patientMatrix)

    dataMatrix = da.concatenate(dataMatrixList, axis=0)
    groundTruth = da.concatenate(groundTruthList, axis=0)

    return dataMatrix, groundTruth


def get_data_for_training(imageSizes, dataMatrix, groundTruth, train_index):
    return

###########################################################################################################

# Main program

basepath = "/home/eline/OneDrive/__NiiFormat"
patientPaths = GetPatients(basepath)

print(patientPaths)

t2 = ["T2"]
dwi = ["DWI_b0", "DWI_b1", "DWI_b2", "DWI_b3", "DWI_b4", "DWI_b5", "DWI_b6"]
ffe = ["FFE_e0"]
t1t2sense = []

scantypes = [t2, dwi, ffe, t1t2sense]
scans = []
for type in scantypes:
    if type:
        scans.append(type)
print(scans)


maskchoice = "intersection" # an, shh, intersection or union

numberOfPatients = len(patientPaths)
#patientNumbers = np.linspace(0, numberOfPatients, numberOfPatients, endpoint=False, dtype=int)
patientIndex = np.zeros((numberOfPatients,2))

dataMatrix, groundTruth = buildDataset(patientPaths, scans, maskchoice, patientIndex)
print(dataMatrix)
print(groundTruth)
print(patientIndex)

crossvalidator = select_cross_validator("K-fold")
for train_index, test_index in crossvalidator.split(patientIndex):
    print("TRAIN:", train_index, "TEST:", test_index)




'''
testarray = da.concatenate([dataMatrix[0:2220400][:], dataMatrix[4002256:][:]], axis=0)
print(testarray)
'''

