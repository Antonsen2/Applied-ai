import os
import torch
import numpy as np

from PIL import Image
from cnn_to_rcnn import CNN
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

IMAGE_PATH = 'C:/code/projects/python/applied-ai/data/new-data/datasets/50-50/'


def load_images_from_folder(img_dir: str) -> list[str]:
    images = list()

    for file in os.listdir(img_dir):
        file = os.path.join(img_dir, file)

        if not os.path.isfile(file):  # Ignore directories etc.
            continue

        img = Image.open(file)
        if img is not None:
            img = np.asarray(img)
            img = torch.from_numpy(img)
            images.append(img)
    
    return images


if __name__ == '__main__':
    X = load_images_from_folder(IMAGE_PATH + 'fire')
    X.extend(load_images_from_folder(IMAGE_PATH + 'no-fire'))
    
    y = ['fire'] * 500
    y.extend(['no-fire'] * 500)

    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)

    # X = torch.from_numpy(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.4
    )

    model = CNN(X_train[0].shape[0])

    optimizer = Adam(model.parameters(), lr=0.03)
    loss_func = CrossEntropyLoss()

    for epoch in range(100):
        model.train_phase(
            epoch=epoch,
            optimizer=optimizer,
            criterion=loss_func,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
        )

    with torch.no_grad():
        train_output = model(X_train.cuda()) if torch.cuda.is_available() else model(X_train)
        test_output = model(X_test.cuda()) if torch.cuda.is_available() else model(X_test)

    softmax = torch.exp(train_output).cpu()
    prob = list(softmax.numpy())
    preds = np.argmax(prob, axis=1)
    print(f'Train Accuracy Score: {accuracy_score(y_train, preds)}')
    print(f'Train MAE: {mean_absolute_error(y_train, preds)}')

    softmax = torch.exp(test_output).cpu()
    prob = list(softmax.numpy())
    preds = np.argmax(prob, axis=1)
    print(f'Test Accuracy Score: {accuracy_score(y_test, preds)}')
    print(f'Test MAE: {mean_absolute_error(y_test, preds)}')