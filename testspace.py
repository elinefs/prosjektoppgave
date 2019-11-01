import sklearn
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import sklearn


def make_partition(method):
    if method == "leave-One-Out":
        cross_validator = LeaveOneOut()
    elif method == "K-fold":
        cross_validator = KFold(n_splits=5)
    else:
        print("Cross-validator unknown or not implemented.")

    return cross_validator


data = np.array([1,2,3,4,5,6,7,8,9,10, 11, 12, 13, 14, 15])
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

x = np.linspace(0, 10, 10, endpoint=False, dtype=int)
print(x)

print(sklearn.base.is_classifier(sklearn.linear_model.SGDClassifier()))

array = np.array(((1,2,3,4,5), (2,3,0,3,2), (5,4,3,2,1)))
remove =  np.where(~array.all(axis=1))[0]
array = np.delete(array, remove, 0)
print(array)