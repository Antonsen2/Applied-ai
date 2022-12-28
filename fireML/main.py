import os
import numpy as np
from PIL import Image
from tensorflow import keras


MODEL_PATH="model/wildfiremodel.h5"
FIRE_IMAGES = 'sample/fire/'  # Enter the path to the fire images
NON_FIRE_IMAGES = 'sample/nofire/'  # Enter the path to the non-fire images


def load_model():
    return keras.models.load_model(MODEL_PATH)


def load_images(dir: str, size=np.inf) -> list[np.array]:
    """
    Load all images from directory.
    params:
        dir: str, directory for the images
        size=np.inf: int, Size of how many images to take. Set to np.inf to take all images.
    return:
        list[np.array], list of all iamges converted to numpy arrays
    """
    images = []

    for idx, file in enumerate(os.listdir(dir)):
        file = os.path.join(dir, file)

        if not os.path.isfile(file):  # Ignore directories etc.
            continue

        if idx >= size:
            break

        with Image.open(file) as img:
            if img is not None:
                img = np.asarray(img)
                images.append(img)
    return images


def dataset(size: int):
    X = load_images(FIRE_IMAGES, size=size)
    y = [[0]] * size  # Set fire images to 0

    X.extend(load_images(NON_FIRE_IMAGES, size=size))
    y.extend([[1]] * size)  # Set non-fire images to 1

    X = np.array(X)
    y = np.asarray(y, dtype='uint8')
    return X, y


def test(model, X, y):
    sample, correct = 0, 0
    for idx, img in enumerate(X):
        prediction = model(img)
        sample += 1
        if np.round(prediction) == y[idx]:
            correct += 1
    acceracy = (correct / sample) * 100
    print(f"Sample size: {sample}, Correct: {correct}, Acceracy: {acceracy}%")
    return acceracy


def main():
    model = load_model()
    X, y = dataset(5)
    #test(model, X, y)


if __name__ == '__main__':
    main()
