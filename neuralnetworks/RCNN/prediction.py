import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision
import torch

from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from PIL import Image
from torchvision.models.detection import FasterRCNN


LABELS = ['smoke', 'fire']
NUM_OF_CLASSES = len(LABELS)+1
SAVE_PATH = './outputs/models/'
MODEL_NAME = 'model_include_fire.pt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRANSFORM = transforms.Compose([transforms.ToTensor()])
IMG_PATH = '../data/new-data/datasets/full/fire/fire-2065.jpg'

# Colors are not working correctly yet
COLORS = [
    '#d90166',
    '#8f00f1',
    '#d0ff14',
    '#eb5030',
    '#ff000d',
    '#66ff00',
    '#0203e2',
    '#04d9ff',
    '#ff00ff',
    '#fffd01',
    '#e56024',
    '#dfff4f',
    '#ff3503',
    '#6600ff',
    '#f7b718',
    '#fe0002',
    '#45cea2',
    '#ff85ff',
    '#1974d2',
    '#fe6700',
]

def load_model() -> FasterRCNN:
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=False,
        num_classes=NUM_OF_CLASSES
    )
    IN_FEATURES = model.roi_heads.box_predictor.cls_score.in_features

    model.load_state_dict(torch.load(SAVE_PATH + MODEL_NAME))
    model.eval()

    return model

def get_relevant_scores(threshold: float, boxes: list, scores: list, labels: list) -> tuple:
    x = len([score for score in scores if score >= threshold])
    return boxes[:x], scores[:x], labels[:x]

def predict(model, img_path: str, transform: torchvision.transforms, threshold: float) -> tuple:
    img = Image.open(img_path)
    pred_img = transform(img)
    pred_img = pred_img.view(1, 3, pred_img.shape[1], pred_img.shape[2])

    preds = model(pred_img)
    outputs = [{k: v.to(torch.device('cpu')) for k, v in target.items()} for target in preds]

    boxes = outputs[0]['boxes'].data.cpu().numpy().astype(np.int32)
    scores = outputs[0]['scores'].data.cpu().numpy()
    labels = outputs[0]['labels'].data.cpu().numpy().astype(np.int32)

    boxes, scores, labels = get_relevant_scores(threshold, boxes, scores, labels)
    
    return boxes, scores, labels

def plot_prediction(img_path: str, predictions: tuple) -> None:
    patches = []

    img = Image.open(img_path)
    _, ax = plt.subplots(figsize=(13,7))
    plt.imshow(img)

    box_counter = 0
    for box, score, label in zip(predictions[0], predictions[1], predictions[2]):
        box_counter = 1
        score *= 100
        label = f'{str(LABELS[label-1])} : {score: .2f}%'

        x_min = int(box[0])
        y_min = int(box[1])
        x_max = int(box[2])
        y_max = int(box[3])

        plt.gca().add_patch(Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            edgecolor=COLORS[box_counter],
            facecolor=None,
            fill=False,
            lw=1
        ))

        patch = Line2D(
            [0], [0],
            marker='o',
            color='w',
            markerfacecolor=COLORS[box_counter],
            label=label
        )
        patches.append(patch)
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.,
        handles=[patch for patch in patches]
    )

    plt.show()