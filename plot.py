from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm, classes, normalize=False, title="Overall confusion matrix"):
    """
    A function to calculate and plot the confusion matrix.
    :param cm: the confusion matrix.
    :param classes: a list with names (strings) of the different classes
    :param normalize: optional, set to True if normalization of the confusion matrix is wanted.
    :param title: optional, a string with the title of the plot.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


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
    plt.show()


def plot_dice_scores(dice):
    """
    A function to plot the DICE scores.
    :param dice: a 2D matrix containing patient names and corresponding DICE scores.
    """
    score = []
    #patient = []
    for element in dice:
        score.append(element[1])
        #patient.append(element[0])

    plt.plot(score, 'o')
    #plt.xticks(rotation=90)
    plt.xlabel('Patient')
    plt.ylabel('DSC')
    plt.show()
