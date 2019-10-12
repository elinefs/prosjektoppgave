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