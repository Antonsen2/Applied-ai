import sys
sys.path.append('./')
# TODO: Change sys.path, this works for now

import numpy as np

from cnn import CNN
from utilities.image import load_images


FIRE_IMAGES = ''  # Enter the path to the fire images
NON_FIRE_IMAGES = ''  # Enter the path to the non-fire images

def main() -> None:
    X = load_images(FIRE_IMAGES)
    y = [[0]] * len(X)  # Set fire images to 0

    X.extend(load_images(NON_FIRE_IMAGES))
    y.extend([[1]] * (len(X) / 2))  # Set non-fire images to 1

    X = np.array(X)
    y = np.asarray(y, dtype='uint8')

    model = CNN(X, y)

    model.train()
    model.save('test_model')


if __name__ == '__main__':
    main()
