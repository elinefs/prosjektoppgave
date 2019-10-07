import os
import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt
import sklearn
from sklearn.preprocessing import StandardScaler
import dask.array as da
###########################################################################################################
# Functions


def get_array_from_nii_image(path):
    image = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(image)
    array = array.flatten()
    return array


def select_cross_validator(method): # Kan legge til flere cross-validatorer her?
    if method == "leave-One-Out":
        cross_validator = sklearn.model_selection.LeaveOneOut()
    elif method == "K-fold":
        cross_validator = sklearn.model_selection.KFold(n_splits=2)
    else:
        print("Cross-validator unknown or not implemented.")
    return cross_validator


def normalization(data):
    if len(np.shape(data)) == 1:
        data = np.expand_dims(data, axis=1)
    scaler = StandardScaler(with_mean=False, with_std=False)
    scaler.fit(data)
    normData = scaler.transform(data)

    return normData


def radiologist(method, patientPath):

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
            mask[i] = mask1[i] or mask2[i]

    elif method == "union":
        maskPath1 = os.path.join(patientPath, maskfile_an)
        mask1 = get_array_from_nii_image(maskPath1)
        maskPath2 = os.path.join(patientPath, maskfile_shh)
        mask2 = get_array_from_nii_image(maskPath2)
        mask = np.zeros(len(mask1), dtype=np.bool)
        for i in range(len(mask1)):
            mask[i] = mask1[i] and mask2[i]

    else:
        print("Unknown choice for mask.")

    return mask

def GetPatients(directoryPath):
    patientPaths = []
    for entry in os.listdir(basepath):
        patientPath = os.path.join(basepath, entry)
        if os.path.isdir(patientPath):
            patientPaths.append(patientPath)
    return patientPaths


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
patientIndex = np.zeros(numberOfPatients)
dataMatrixList = []
groundTruthList = []
imageSizes = np.zeros(numberOfPatients)

patientNo = 0

for patient in patientPaths:

    mask = radiologist(maskchoice, patient)
    groundTruthList.append(mask)
    patientIndex[patientNo] = patientNo
    numberOfVoxels = len(mask)
    imageSizes[patientNo] = numberOfVoxels
    patientData = []

    for type in scans:
        index = 0
        scandata = np.zeros((numberOfVoxels, len(type)), dtype=np.single)
        for scan in type:
            filename = scan + ".nii"
            path = os.path.join(patient, filename)
            image = get_array_from_nii_image(path)
            scandata[:, index] = image
            index += 1
        scandata = normalization(scandata)
        patientData.append(scandata)

    patientMatrix = np.concatenate(patientData, axis=1)
    dataMatrixList.append(patientMatrix)
    patientNo += 1
    

dataMatrix = da.concatenate(dataMatrixList, axis=0)
print(dataMatrix)
groundTruth = da.concatenate(groundTruthList, axis=0)
print(groundTruth)

testarray = da.concatenate([dataMatrix[0:2220400][:], dataMatrix[4002256:][:]], axis=0)
print(testarray)