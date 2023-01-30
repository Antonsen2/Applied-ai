import pandas as pd
import albumentations as A
import torch

from albumentations.pytorch.transforms import ToTensorV2

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

SAVE_PATH = '../outputs/models/'  # Path to saved models
MODEL_NAME = 'new_model.pt'  # Model name (.pt suffix)

TRAIN_DF = pd.read_csv('')  # Path to train_annotations.csv
VAL_DF = pd.read_csv('')  # Path to valid_annotations.csv
TRAIN_IMAGE_PATH = ''  # Path to the directory containg train images
VAL_IMAGE_PATH = ''  # Path to the directory containg validation images

LABELS = TRAIN_DF['class'].unique()
NUM_OF_CLASSES = len(LABELS)+1

TRAIN_TRANSFORM = A.Compose(
    [
      A.HorizontalFlip(p=0.5),
      A.RandomBrightnessContrast(p=0.2),
      ToTensorV2(p=1)
    ],
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
)
VAL_TRANSFORM = A.Compose(
    [ToTensorV2(p=1)],
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
)