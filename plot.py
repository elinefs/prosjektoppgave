from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

########################################################################################################################

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False):
    """
    A function to calculate and plot the confusion matrix.
    :param y_true: a list with the ground truth for the test data.
    :param y_pred: a list with the predictions for the test data.
    :param classes: a list with names (strings) of the different classes
    :param normalize: optional, set to True if normalization of the confusion matrix is wanted.
    :return: a plot with the confusion matrix.
    """

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Confusion matrix, normalized'
    else:
        title = 'Confusion matrix'

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes,
           title=title, ylabel='True label', xlabel='Predicted label')

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax