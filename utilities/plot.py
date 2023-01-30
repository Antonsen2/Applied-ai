"""This module contains helper functions for plotting"""

import itertools
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix
from PIL import Image

def plot_random_images(df: pd.DataFrame, path: str = 'Filepath', label: str = 'Label') -> None:
    """Plot 15 random images from a dataframe
    params:
        df: Pandas DataFrame, dataframe with the images paths
        path: str, name of the column containing the image path. Default 'Filepath'
        label: str, name of the column containing the image label. Default 'Label'
    return:
        None
    """
    random_index = np.random.randint(0, len(df), 15)
    _, axes = plt.subplots(
        nrows=3,
        ncols=5,
        figsize=(25, 15),
        subplot_kw={'xticks': [], 'yticks': []}
    )

    for i, ax in enumerate(axes.flat):
        image = Image.open(df[path][random_index[i]])
        ax.imshow(image)
        ax.set_title(df[label][random_index[i]])

    plt.tight_layout()
    plt.show()

def plot_predictions(df: pd.DataFrame, preds: list, path: str = 'Filepath', label: str = 'Label') -> None:
    """Plot 15 predictions with color differences (green for correct prediction and red for uncorrect)
    params:
        df: Pandas DataFrame, dataframe with the images paths
        preds: list, list with predictions
        path: str, name of the column containing the image path. Default 'Filepath'
        label: str, name of the column containing the image label. Default 'Label'
    return:
        None
    """
    random_index = np.random.randint(0, len(df)-1, 15)
    _, axes = plt.subplots(
        nrows=3,
        ncols=5,
        figsize=(25, 15),
        subplot_kw={'xticks': [], 'yticks': []}
    )

    for i, ax in enumerate(axes.flat):
        image = Image.open(df[path].iloc[random_index[i]])
        ax.imshow(image)
        if df[label].iloc[random_index[i]] == preds[random_index[i]]:
          color = 'green'
        else:
          color = 'red'
        ax.set_title(
            f'True: {df[label].iloc[random_index[i]]}\nPredicted: {preds[random_index[i]]}',
            color=color
        )

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true: list, y_pred: list, classes: list  = None, figsize: tuple = (25, 15),
                          text_size: int = 10, norm: bool = False, savefig: bool = False, file_name: str = None) -> None: 
    """Plot a labelled confusion matrix comparing predictions and truth labels
    params:
        y_true: list, list of truth labels (must be same shape as y_pred).
        y_pred: list, list of predicted labels (must be same shape as y_true).
        classes: list, list of class labels. Default None (integer labels are used)
        figsize: tuple, figure size. Default (25, 15)
        text_size: int, text size. Default 10
        norm: bool, whether to normalize values. Default False
        savefig: bool, save confusion matrix to file. Default False
        file_name: str, name of save file. Default None
    return:
        None
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    n_classes = cm.shape[0]

    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    ax.set(title='Wildfire Confusion Matrix',
         xlabel='Predicted',
         ylabel='True',
         xticks=np.arange(n_classes),
         yticks=np.arange(n_classes), 
         xticklabels=labels,
         yticklabels=labels)

    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()
    plt.xticks(rotation=90, fontsize=text_size)
    plt.yticks(fontsize=text_size)

    threshold = (cm.max() + cm.min()) / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            plt.text(j, i, f'{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)',
                horizontalalignment='center',
                color='white' if cm[i, j] > threshold else 'black',
                size=text_size)
        else:
            plt.text(j, i, f"{cm[i, j]}",
              horizontalalignment="center",
              color='white' if cm[i, j] > threshold else 'black',
              size=text_size)

    if savefig:
        fig.savefig(file_name)

def plot_loss_curves(history) -> None:
    """Plot loss curves for training and validation metrics
    params:
        history: TensorFlow model History object
    return:
        None
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
