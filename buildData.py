import os
import numpy as np
import dask.array as da

import options
import processData

########################################################################################################################

def buildDataset(patientPaths, scans, maskchoice):
    """
    A function to build the data matrix and the list with ground truth.
    :param patientPaths: a list with strings that are paths to the patient files.
    :param scans: a matrix containing the different scan types.
    :param maskchoice: a string with the method for creating the ground truth.
    :return: return a matrix with all the image data and a list with the ground truth.
    """
    dataDict = {}
    groundTruthDict = {}
    imsizes = {}
    for patientNo, patient in enumerate(patientPaths):

        mask = options.radiologist(maskchoice, patient)
        numberOfVoxels = len(mask)
        patientData = []

        for type in scans:
            scandata = np.zeros((numberOfVoxels, len(type)), dtype=np.single)
            for index, scan in enumerate(type):
                filename = scan + ".nii"
                path = os.path.join(patient, filename)
                image, size = processData.get_array_from_nii_image(path)
                scandata[:, index] = image
                if patientNo not in imsizes.keys():
                    imsizes[patientNo] = size
            scandata = processData.normalization(scandata)
            patientData.append(scandata)

        patientMatrix = np.concatenate(patientData, axis=1)

        dataDict[patientNo] = patientMatrix
        groundTruthDict[patientNo] = mask

    return dataDict, groundTruthDict, imsizes


def get_data_for_training(dataDict, groundTruthDict, indexList):
    '''
    A function to combine the data used to train the model in a dask array.
    :param dataDict: a dictionary with the data.
    :param groundTruthDict: a dictionary with the ground truth.
    :param indexList: indexes (keys in dictionary) that should be included in the dask array.
    :return: two dask arrays; one with data and one with ground truth.
    '''
    dataList = []
    groundtruthList = []
    for index in indexList:
        indexData, indexTruth = dataDict[index], groundTruthDict[index]

        remove = np.where(~indexData.all(axis=1))[0]
        indexData = np.delete(indexData, remove, 0)
        indexTruth = np.delete(indexTruth, remove)

        indexData, indexTruth = processData.downsample50_50(indexData, indexTruth)

        dataList.append(indexData)
        groundtruthList.append(indexTruth)

    data = da.concatenate(dataList, axis=0)
    groundTruth = da.concatenate(groundtruthList, axis=0)
    return data, groundTruth


def get_data_for_test(dataDict, groundTruthDict, indexList):
    '''
    A function to combine the data used to test the model in a dask array.
    :param dataDict: a dictionary with the data.
    :param groundTruthDict: a dictionary with the ground truth.
    :param indexList: indexes (keys in dictionary) that should be included in the dask array.
    :return: two dask arrays; one with data and one with ground truth.
    '''
    dataList = []
    groundtruthList = []
    for index in indexList:
        indexData, indexTruth = dataDict[index], groundTruthDict[index]
        dataList.append(indexData)
        groundtruthList.append(indexTruth)
    data = da.concatenate(dataList, axis=0)
    groundTruth = da.concatenate(groundtruthList, axis=0)
    return data, groundTruth
