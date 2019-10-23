import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.linear_model
from dask_ml.wrappers import Incremental
import time

import options
import getData
import buildData
import plot
from sklearn.metrics import confusion_matrix
########################################################################################################################

t0 = time.time()

basepath = "/home/eline/OneDrive/__NiiFormat" # Path to the patient folders.
patientPaths = getData.GetPatients(basepath)

# Choose which scans to include.
t2 = ["T2"]
dwi = ["DWI_b6"]#"DWI_b0", "DWI_b1", "DWI_b2", "DWI_b3", "DWI_b4", "DWI_b5", "DWI_b6"]
ffe = []#"FFE_e0", "FFE_e2", "FFE_e2", "FFE_e3", "FFE_e4"]
t1t2sense = ["T1T2SENSE_e0_t00", "T1T2SENSE_e0_t01", "T1T2SENSE_e0_t02", "T1T2SENSE_e0_t03", "T1T2SENSE_e0_t04",
             "T1T2SENSE_e0_t05", "T1T2SENSE_e0_t06", "T1T2SENSE_e0_t07", "T1T2SENSE_e0_t08", "T1T2SENSE_e0_t09",
             "T1T2SENSE_e0_t10", "T1T2SENSE_e0_t11", "T1T2SENSE_e0_t12", "T1T2SENSE_e0_t13",
             "T1T2SENSE_e1_t00", "T1T2SENSE_e1_t01", "T1T2SENSE_e1_t02", "T1T2SENSE_e1_t03", "T1T2SENSE_e1_t04",
             "T1T2SENSE_e1_t05", "T1T2SENSE_e1_t06", "T1T2SENSE_e1_t07", "T1T2SENSE_e1_t08", "T1T2SENSE_e1_t09",
             "T1T2SENSE_e1_t10", "T1T2SENSE_e1_t11", "T1T2SENSE_e1_t12", "T1T2SENSE_e1_t13",
             "T1T2SENSE_e2_t00", "T1T2SENSE_e2_t01", "T1T2SENSE_e2_t02", "T1T2SENSE_e2_t03", "T1T2SENSE_e2_t04",
             "T1T2SENSE_e2_t05", "T1T2SENSE_e2_t06", "T1T2SENSE_e2_t07", "T1T2SENSE_e2_t08", "T1T2SENSE_e2_t09",
             "T1T2SENSE_e2_t10", "T1T2SENSE_e2_t11", "T1T2SENSE_e2_t12", "T1T2SENSE_e2_t13"]
t1t2sense = []

scantypes = [t2, dwi, ffe, t1t2sense]
scans = []
for type in scantypes:
    if type:
        scans.append(type)


# Choose the mask/ground truth.
maskchoice = "intersection" # an, shh, intersection or union

# Creating a list with patient numbers which will be keys in the dictionaries.
numberOfPatients = len(patientPaths)
patientNumbers = np.linspace(0, numberOfPatients, numberOfPatients, endpoint=False, dtype=int)

# Creating dictionaries.
dataDict, groundTruthDict = buildData.buildDataset(patientPaths, scans, maskchoice)

# Choose cross-validator.
crossvalidator = options.select_cross_validator("K-fold", 5)

loadtime = time.time()

# Train model.
for train_index, test_index in crossvalidator.split(patientNumbers):
    # First splitting the data and building dask arrays.
    trainingX, trainingY = buildData.get_data_for_training(dataDict, groundTruthDict, train_index)
    testX, testY = buildData.get_data_for_training(dataDict, groundTruthDict, test_index)

    # Rechunk to be sure to not run into memory issues later.
    trainingX = trainingX.rechunk((100000, -1))
    trainingY = trainingY.rechunk((100000, -1))
    testX = testX.rechunk((100000, -1))
    testY = testY.rechunk((100000, -1))

    # Using incremental learning (out of core learning) because of the large amount of data.
    estimator = sklearn.linear_model.SGDClassifier() # Estimator have to have partial_fit API implemented.
    clf = Incremental(estimator, scoring='accuracy')
    clf.fit(trainingX, trainingY, classes=[True, False])
    data = clf.predict(testX)

    plot.plot_confusion_matrix(testY.compute(), data.compute(), classes=['Normal', 'Tumour'])
    plt.show()
    #print(confusion_matrix(testY.compute(), data.compute()))


t1 = time.time()
print('loadtime: ' + str(loadtime-t0))
print('traintime: ' + str(t1-loadtime))
print('runtime: ' + str(t1-t0))