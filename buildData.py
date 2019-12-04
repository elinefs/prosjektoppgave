import os
import numpy as np
import dask.array as da

import options
import processData


def buildDataset(patientPaths, patientNames, scans, maskchoice):
    """
    A function to build the dictionary with the image data and the dictionary with ground truth.
    :param patientPaths: a list with strings that are paths to the patient files.
    :param patientNames: a list with strings that are names for the patients.
    :param scans: a matrix containing the different scan types.
    :param maskchoice: a string with the method for creating the ground truth.
    :return: return a dictionary with all the image data and a dictionary with the ground truth.
    """
    dataDict = {}
    groundTruthDict = {}
    imsizes = {}
    for patientNo, patient in enumerate(patientPaths):

        mask = options.radiologist(maskchoice, patient)
        numberOfVoxels = len(mask)
        patientData = []
        key = patientNames[patientNo]

        for type in scans:
            scandata = np.zeros((numberOfVoxels, len(type)), dtype=np.single)
            for index, scan in enumerate(type):
                filename = scan + ".nii"
                path = os.path.join(patient, filename)
                image, size = processData.get_array_from_nii_image(path)
                scandata[:, index] = image
                if key not in imsizes.keys():
                    imsizes[key] = size
            scandata = processData.normalization(scandata)
            patientData.append(scandata)

        patientMatrix = np.concatenate(patientData, axis=1)

        dataDict[key] = patientMatrix
        groundTruthDict[key] = mask

    return dataDict, groundTruthDict, imsizes


def get_data_for_training(dataDict, groundTruthDict, indexList):
    '''
    A function to combine the data used to train the model in a Dask array.
    :param dataDict: a dictionary with the data.
    :param groundTruthDict: a dictionary with the ground truth.
    :param indexList: indexes (keys in dictionary) that should be included in the Dask array.
    :return: two Dask arrays; one with data and one with ground truth.
    '''
    dataList = []
    groundtruthList = []
    for index in indexList:
        indexData, indexTruth = dataDict[index], groundTruthDict[index]

        # Remove rows where at least one value is zeros.
        remove = np.where(~indexData.all(axis=1))[0]
        indexData = np.delete(indexData, remove, 0)
        indexTruth = np.delete(indexTruth, remove)

        # Downsample to 50/50 with tumor voxels and normal tissue voxels.
        indexData, indexTruth = processData.downsample50_50(indexData, indexTruth)

        dataList.append(indexData)
        groundtruthList.append(indexTruth)

    # Join to a Dask array.
    data = da.concatenate(dataList, axis=0)
    groundTruth = da.concatenate(groundtruthList, axis=0)
    return data, groundTruth


def get_data_for_test(dataDict, groundTruthDict, indexList, zeroIndexDict):
    '''
    A function to combine the data used to test the model in a Dask array.
    :param dataDict: a dictionary with the data.
    :param groundTruthDict: a dictionary with the ground truth.
    :param indexList: indexes (keys in dictionary) that should be included in the Dask array.
    :param zeroIndexDict: a dictionary to store indexes where data value is zero.
    :return: two Dask arrays; one with data and one with ground truth.
    '''
    dataList = []
    groundtruthList = []
    for index in indexList:
        indexData, indexTruth = dataDict[index], groundTruthDict[index]
        dataList.append(indexData)
        groundtruthList.append(indexTruth)

        # Store the indexes of the rows where at least one value is zero.
        zeros = np.where(~indexData.all(axis=1))[0]
        zeroIndexDict[index] = zeros

    # Join to a Dask array.
    data = da.concatenate(dataList, axis=0)
    groundTruth = da.concatenate(groundtruthList, axis=0)
    return data, groundTruth
