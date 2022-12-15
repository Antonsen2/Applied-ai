# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

from torchvision.utils import make_grid
from torchvision.io import read_image
from pathlib import Path


# %%
pred = [100, 100, 135, 175]


# %%
def show(imgs: list) -> None:
    if not isinstance(imgs, list):
        imgs = [imgs]

    _, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

plt.rcParams['savefig.bbox'] = 'tight'

# %%
fire_0 = read_image('./fire-9.jpg')
fire_1 = read_image('./fire-24.jpg')
fire_2 = read_image('./fire-87.jpg')
fires = [fire_0, fire_1, fire_2]

grid = make_grid(fires)
show(grid)

# %%
from torchvision.utils import draw_bounding_boxes


boxes = torch.tensor([pred], dtype=torch.float)
colors = ["blue"]
result = draw_bounding_boxes(fire_0, boxes, colors=colors, width=1)
show(result)


# %%
