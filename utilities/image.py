"""This module contains helper functions for image processing"""

import os
import shutil
import numpy as np

from PIL import Image


def resize_image(image_dir: str, new_size: tuple[int, int],
                 image_category: str, num_suffix: int, new_dir: str = None) -> None:
    """
    Helper to resize a single image.
    params:
        image_dir: str, image directory
        new_size: tuple[int, int], pixelsize (x, y)
        image_category: str, category for the image, ex. 'fire' for fire images
        num_suffix: int, the new image number (xxxxx-num_suffix.jpg)
        new_dir: str, new image directory. Default None
    return:
        None
    """
    img = Image.open(image_dir)
    img = img.resize(new_size)
    img = img.convert('RGB')  # Remove Alpha-Channel
    img.save(f'{new_dir}/{image_category}-resized/{image_category}-{num_suffix}.jpg')

def bulk_resize_images(data_dir: str, image_category: str,
                       new_size: tuple[int, int], new_dir: str = None) -> None:
    """
    Helper to bulk resize images.
    params:
        data_dir: str, data root directory
        image_category: str, category for the image, ex. 'fire' for fire images
        new_size: tuple[int, int], pixelsize (x, y)
        new_dir: str, new image directory. Default None (uses data_dir as output)
    return:
        None
    """
    for img_folder in os.listdir(data_dir):
        if os.path.isfile(os.path.join(data_dir, img_folder)):  # Ignore files
            continue

        for i, image in enumerate(os.listdir(f'{data_dir}{img_folder}')):
            resize_image(
                image_dir=f'{data_dir}/{img_folder}/{image}',
                new_size=new_size,
                image_category=image_category,
                num_suffix=i,
                new_dir=new_dir if new_dir else data_dir
            )

def bulk_move_images(old_dir: str, new_dir: str) -> None:
    """
    Helper to bulk move images.
    params:
        old_dir: str, initial directory
        new_dir: str, new directory
    return:
        None
    """
    for img in os.listdir(old_dir):
        if os.path.isfile(os.path.join(old_dir, img)):
            shutil.move(
                src=old_dir + img,
                dst=new_dir + img,
                copy_function=shutil.copy2
            )


def load_images(dir: str) -> list[np.array]:
    """
    Load all images from directory.
    params:
        dir: str, directory for the images
    return:
        list[np.array], list of all iamges converted to numpy arrays
    """
    images = list()

    for file in os.listdir(dir):
        file = os.path.join(dir, file)

        if not os.path.isfile(file):  # Ignore directories etc.
            continue

        img = Image.open(file)
        if img is not None:
            img = np.asarray(img)
            images.append(img)
    
    return images
