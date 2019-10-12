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

########################################################################################################################

t0 = time.time()

basepath = "/home/eline/OneDrive/__NiiFormat" # Path to the patient folders.
patientPaths = getData.GetPatients(basepath)

print(patientPaths)

# Choose which scans to include.
t2 = ["T2"]
dwi = ["DWI_b5", "DWI_b6"]
ffe = []
t1t2sense = []

scantypes = [t2, dwi, ffe, t1t2sense]
scans = []
for type in scantypes:
    if type:
        scans.append(type)
print(scans)

# Choose the mask/ground truth.
maskchoice = "intersection" # an, shh, intersection or union

# Creating a list with patient numbers which will be keys in the dictionaries.
numberOfPatients = len(patientPaths)
patientNumbers = np.linspace(0, numberOfPatients, numberOfPatients, endpoint=False, dtype=int)

# Creating dictionaries.
dataDict, groundTruthDict = buildData.buildDataset(patientPaths, scans, maskchoice)

# Choose cross-validator.
crossvalidator = options.select_cross_validator("K-fold")


# Train model.
for train_index, test_index in crossvalidator.split(patientNumbers):
    # First splitting the data and building dask arrays.
    trainingX, trainingY = buildData.get_data_for_training(dataDict, groundTruthDict, train_index)
    testX, testY = buildData.get_data_for_training(dataDict, groundTruthDict, test_index)

    # Using incremental learning (out of core learning) because of the large amount of data.
    estimator = sklearn.linear_model.SGDClassifier() # Estimator have to have partial_fit API implemented.
    clf = Incremental(estimator, scoring='accuracy')
    clf.fit(trainingX, trainingY, classes=[True, False])
    data = clf.predict(testX)
    plot.plot_confusion_matrix(testY.compute(), data.compute(), classes=['Normal', 'cancer'], normalize=True)
    plt.show()


t1 = time.time()
print('runtime: ' + str(t1-t0))