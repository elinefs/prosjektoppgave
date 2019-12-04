import numpy as np
import SimpleITK as sitk
from skimage import morphology


def remove_small_areas(array, imsize):
    mask = np.reshape(array, imsize)
    for slice in mask:
        morphology.remove_small_objects(slice, 50, in_place=True)
    mask = mask.flatten()
    return mask


def calculate_dice(confusionMatrix):
    """
    A function to calculate the DICE score from a confusion matrix.
    :param confusionMatrix: a 2x2 matrix.
    :return: the calculated DICE score (float).
    """
    dice = (2 * confusionMatrix[1, 1]) / ((2 * confusionMatrix[1, 1]) + confusionMatrix[0, 1] + confusionMatrix[1, 0])
    return dice


def array_to_image(array, imsize):
    """
    A function to convert a 1D array to a image.
    :param array: a 1D array with the image data.
    :param imsize: an array with the dimensions of the image (shape).
    :return: an image.
    """
    im = np.reshape(array, imsize)
    im = im.astype(int)
    im = sitk.GetImageFromArray(im)
    return im


def save_dice_scores(dice, filename):
    """
    A function that saves the DICE scores to a text file.
    :param dice: a 2D matrix containing patient names and corresponding DICE scores.
    :param filename: a string with the name of the file that will be created/overwritten.
    """
    f = open(filename + ".txt", "w")
    for element in dice:
        f.write(element[0] + "\t" + str(element[1]) + "\n")
    f.close()