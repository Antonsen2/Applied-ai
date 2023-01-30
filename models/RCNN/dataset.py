"""Custom Dataset Module"""


import os
from typing import Any

import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset


class LabelMap:
    """
    Class to map and encode labels for training

    Attributes:
        map: dict, {'class-label': id (int)} aka first label = 1 etc.
        reversed_map: dict, self.map reversed {id: 'class-label'}

    Example usage:
        In our case the map looks like this:
            {'smoke': 1, 'fire': 2}
    """
    def __init__(self, labels: list) -> None:
        self.map: dict = {label: i+1 for i, label in enumerate(labels)}
        self.reversed_map: dict = {i+1: label for i, label in enumerate(labels)}

    def fit(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        df[col] = df[col].map(self.map)
        return df


class WildfireDataset(Dataset):
    """Custom Dataset for Wildfire Images"""
    def __init__(self, df: pd.DataFrame, img_path: str,
                 labels: list, transforms: Any = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.img_path = img_path
        self.labels = labels
        self.__label_map = LabelMap(self.labels)
        self.df = self.__label_map.fit(df, 'class')
        self.images = self.df['filename'].unique()
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> tuple:
        """Get image and target dict at index i"""
        img_file = os.path.join(self.img_path, self.images[i])
        img_data = self.df.loc[self.df['filename'] == self.images[i]]

        img = Image.open(img_file)
        img = img.astype(np.float32)
        img = img/255.0


        xmins = img_data['xmin'].values
        xmaxs = img_data['xmax'].values
        ymins = img_data['ymin'].values
        ymaxs = img_data['ymax'].values

        boxes = torch.as_tensor(np.stack([xmins, ymins, xmaxs, ymaxs], axis=1), dtype=torch.float32)
        labels = torch.as_tensor(img_data['class'].values, dtype=torch.int64)

        areas = (boxes[:,3] - boxes[:,1]) * (boxes[:,2] - boxes[:,0])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([i]),
            'areas': areas,
            'iscrowd': iscrowd
        }

        if self.transforms:
            transformed = self.transform(image=img, bboxes=boxes, labels=labels)
            img = transformed['image']
            target['boxes'] = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)

        return torch.as_tensor(img, dtype=torch.float32), target

    def get_w_h(self, image: str) -> tuple:
        """Get image (width x height)"""
        img_data = self.df.loc[self.df['filename'] == image]
        return img_data['width'].values[0], img_data['height'].values[0]
