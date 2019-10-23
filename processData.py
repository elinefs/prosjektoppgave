import numpy as np
import SimpleITK as sitk
from sklearn.preprocessing import StandardScaler

########################################################################################################################

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

def normalization(data):
    """
    A function to normalize the data with mean 0 and std 1.
    :param data: a matrix or a list with the data you want to normalize.
    :return: a matrix with normalized data.
    """
    if len(np.shape(data)) == 1:
        data = np.expand_dims(data, axis=1)
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(data)
    normData = scaler.transform(data)

    return normData

def downsample50_50(patientdata, patientGroundTruth):
    """
    A function to down sample the data so that the dataset is balanced with 50% true and 50% false for each patient.
    :param patientdata: a matrix with the data for one patient.
    :param patientGroundTruth: a list with the ground truth corresponding to the patientdata.
    :return: a down sampled matrix with the data and a corresponding down sampled list with the ground truth.
    """
    tumorIndex = np.where(patientGroundTruth)[0]
    normalIndex = np.where(patientGroundTruth == False)[0]
    number = len(tumorIndex)

    keepIndex = np.random.choice(normalIndex, number, replace=False)

    downsamplePatientData = np.concatenate((patientdata[tumorIndex], patientdata[keepIndex]), axis=0)
    downsampleGroundTruth = np.concatenate((patientGroundTruth[tumorIndex], patientGroundTruth[keepIndex]), axis=0)

    return downsamplePatientData, downsampleGroundTruth
