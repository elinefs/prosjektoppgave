import sklearn
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold


def make_partition(method):
    if method == "leave-One-Out":
        cross_validator = LeaveOneOut()
    elif method == "K-fold":
        cross_validator = KFold(n_splits=2)
    else:
        print("Cross-validator unknown or not implemented.")

    return cross_validator


data = np.array([1,2,3,4,5,6,7,8,9,10])
cv = make_partition("K-fold")
for train_index, test_index in cv.split(data):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = data[train_index], data[test_index]

    print(X_train, X_test)
matrix = np.zeros((len(data), 2))
matrix[:,0] = data
print(matrix)
data = np.expand_dims(data, axis=1)
print(np.shape(data))

part = data[:,0]
print(len(np.shape(part)))
