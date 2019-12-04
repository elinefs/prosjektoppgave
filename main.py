import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
from sklearn.metrics import confusion_matrix
import dask.array as da
from dask_ml.wrappers import Incremental
import SimpleITK as sitk
import time

import options
import getData
import buildData
import plot
import processResults


def main():
    t0 = time.time()

    basepath = "/home/eline/OneDrive/__NiiFormat1" # Path to the patient folders.
    patientPaths, patientIDs = getData.GetPatients(basepath)
    patientIDs = np.array(patientIDs)

    # Choose which scans to include.
    t2 = ["T2"]
    dwi = ["DWI_b00", "DWI_b01", "DWI_b02", "DWI_b03", "DWI_b04", "DWI_b05", "DWI_b06"]
    ffe = []
    t1t2sense = []

    scantypes = [t2, dwi, ffe, t1t2sense]
    scans = []
    for type in scantypes:
        if type:
            scans.append(type)


    # Choose the mask/ground truth.
    maskchoice = "intersection" # an, shh, intersection or union

    # Creating dictionaries to store patient image data and the masks.
    dataDict, groundTruthDict, imsizes = buildData.buildDataset(patientPaths, patientIDs, scans, maskchoice)

    # Choose cross-validator.
    crossvalidator = options.select_cross_validator("K-fold")

    loadtime = time.time()

    totalPred = []
    totalTruth = []

    zeroIndex = {}

    dice = []

    # Train model.
    for train_index, test_index in crossvalidator.split(patientIDs):
        # First splitting the data and building dask arrays.
        trainingX, trainingY = buildData.get_data_for_training(dataDict, groundTruthDict, patientIDs[train_index])
        testX, testY = buildData.get_data_for_test(dataDict, groundTruthDict, patientIDs[test_index], zeroIndex)

        # Using incremental learning (out of core learning) because of the large amount of data.
        estimator = sklearn.linear_model.SGDClassifier() # Estimator have to have partial_fit API implemented.
        clf = Incremental(estimator, scoring='accuracy')
        clf.fit(trainingX, trainingY, classes=[True, False])
        data = clf.predict(testX)

        totalPred.append(data)
        totalTruth.append(testY)

        # Per patient predictions.
        index = 0
        for patientID in patientIDs[test_index]:
            # Get the voxels belonging to the patient.
            size = len(groundTruthDict[patientID])
            pred = data[index:index+size].compute() # compute() is needed to access the values in a Dask array.
            truth = testY[index:index+size].compute()

            # Set rows which contained at least one zero as background (0).
            for element in zeroIndex[patientID]:
                pred[element] = 0

            pred = processResults.remove_small_areas(pred, imsizes[patientID])

            # Calculate the confusion matrix.
            confusionMatrix = confusion_matrix(truth, pred)

            # Calculate the DICE score.
            diceScore = processResults.calculate_dice(confusionMatrix)
            dice.append([patientID, diceScore])

            # Save prediction as nifti file.
            filename = 'predict' + patientID + '.nii'
            predimage = processResults.array_to_image(pred, imsizes[patientID])
            sitk.WriteImage(predimage, filename)

            # Increase index to the starting index of the next patient.
            index += size

            # Make a plot of the confusion matrix for the patient.
            # plot.plot_confusion_matrix(confusionMatrix, classes=['Normal', 'Tumour'], normalize=True, title="Confusion matrix, patient "+str(patientID))

    t1 = time.time()
    print('loadtime: ' + str(loadtime-t0))
    print('traintime: ' + str(t1-loadtime))
    print('runtime: ' + str(t1-t0))

    # Save the DICE scores in a text file.
    processResults.save_dice_scores(dice, "diceScores")
    # Make a plot with the DICE scores.
    plot.plot_dice_scores(dice)

    '''
    # Overall confusion matrix
    totalPred = da.concatenate(totalPred, axis=0)
    totalTruth = da.concatenate(totalTruth, axis=0)
    confusionMatrix = confusion_matrix(totalTruth.compute(), totalPred.compute())

    plot.plot_confusion_matrix(confusionMatrix, classes=['Normal', 'Tumour'], normalize=False)
    print(processResults.calculate_dice(confusionMatrix))
    '''

    sum=0
    n=0
    for i in dice:
        sum+=i[1]
        n+=1
    print(sum/n)

if __name__ == "__main__":
    main()
