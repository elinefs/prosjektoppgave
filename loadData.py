import os
import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt
import sklearn
import dask.array as da
###########################################################################################################
# Functions


def select_cross_validator(method): # Kan legge til flere cross-validatorer her?
    if method == "leave-One-Out":
        cross_validator = sklearn.model_selection.LeaveOneOut()
    elif method == "K-fold":
        cross_validator = sklearn.model_selection.KFold(n_splits=2)
    else:
        print("Cross-validator unknown or not implemented.")
    return cross_validator


def normalization(image): # Ikke implementert enda.
    mean = 0
    std = 1


def radiologist(mask1, mask2,  method): # Bør sjekkes!!!!
    if method == "an":
        mask = mask1
    elif method == "shh":
        mask = mask2
    elif method == "intersection":
        mask = np.zeros(len(mask1))
        for i in range(len(mask1)):
            mask[i] = mask1[i] or mask2[i]
    elif method == "union":
        mask = np.zeros(len(mask1))
        for i in range(len(mask1)):
            mask[i] = mask1[i] and mask2[i]
    else:
        print("Unknown choice for mask.")
    return mask


###########################################################################################################

# Main program

basepath = "/home/eline/Documents/Image_data/"
patients = []


for entry in os.listdir(basepath):
    patientPath = os.path.join(basepath, entry)
    if os.path.isdir(patientPath):
        patients.append(patientPath)

print(patients)

scans = ['T2', 'DWI_b6', 'DWI_b0']cd

maskfile = "Mask_an.nii"

numberOfPatients = len(patients)
patientData = []
groundTruthList = []
imageSizes = np.zeros(numberOfPatients)

patientNo = 0

for patient in patients:

    maskPath = os.path.join(patient, maskfile)
    mask = sitk.ReadImage(maskPath)
    mask = sitk.GetArrayFromImage(mask)
    mask = mask.flatten()
    groundTruthList.append(mask)

    numberOfVoxels = len(mask)
    imageSizes[patientNo] = numberOfVoxels
    imageData = np.zeros((numberOfVoxels, len(scans)), dtype=np.single)

    scanIndex = 0
    for scan in scans:
        filename = scan + ".nii"
        path = os.path.join(patient, filename)
        image = sitk.ReadImage(path)
        image = sitk.GetArrayFromImage(image)
        image = image.flatten()
        imageData[:, scanIndex] = image
        scanIndex += 1
    patientData.append(imageData)
    patientNo += 1
    
print(np.shape(patientData[1]))
print(imageSizes)
dataMatrix = da.concatenate(patientData, axis=0)
print(dataMatrix)
groundTruth = da.concatenate(groundTruthList, axis=0)
print(groundTruth)

testarray = da.concatenate([dataMatrix[0:2220400][:], dataMatrix[4002256:][:]], axis=0)
print(testarray)