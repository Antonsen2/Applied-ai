import numpy as np
from keras_preprocessing.image import img_to_array, ImageDataGenerator
from keras.applications.mobilenet_v2 import preprocess_input
import keras.models
from PIL import Image
import io
import torchvision
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import plotly.graph_objects as go
import matplotlib.pyplot as plt, mpld3
import matplotlib.colors as mcolors
import random

COLORS = list(mcolors.cnames.values())
LABELS = ['smoke', 'fire']


def load_obj_model():
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=False,
        num_classes=len(LABELS)+1
    )
    IN_FEATURES = model.roi_heads.box_predictor.cls_score.in_features

    model.load_state_dict(torch.load('C:/Code/Applied-ai/model_include_fire.pt', map_location='cpu'))
    model.eval()

    return model

def get_relevant_scores(threshold: float, boxes: list, scores: list, labels: list) -> tuple:
    x = len([score for score in scores if score >= threshold])
    return boxes[:x], scores[:x], labels[:x]


def run_obj_model(image):
    image = Image.open(io.BytesIO(image))
    model = load_obj_model()
    transform = transforms.Compose([transforms.ToTensor()])

    pred_img = transform(image)
    pred_img = pred_img.view(1, 3, pred_img.shape[1], pred_img.shape[2])

    preds = model(pred_img)
    outputs = [{k: v.to(torch.device('cpu')) for k, v in target.items()} for target in preds]

    boxes = outputs[0]['boxes'].data.cpu().numpy().astype(np.int32)
    scores = outputs[0]['scores'].data.cpu().numpy()
    labels = outputs[0]['labels'].data.cpu().numpy().astype(np.int32)

    boxes, scores, labels = get_relevant_scores(0.80, boxes, scores, labels)
    new_img = plot_prediction(image, (boxes, scores, labels))
    return new_img


def plot_prediction(img, predictions: tuple) -> None:

    patches = []
    colors = list(mcolors.cnames.values())
    _, ax = plt.subplots(figsize=(13,7))
    plt.imshow(img)
    
    
    for box, score, label in zip(predictions[0], predictions[1], predictions[2]):
        box_counter = random.randint(0, len(COLORS))
        print(box_counter)
        print(COLORS[box_counter])
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
            edgecolor=colors[box_counter],
            facecolor=None,
            fill=False,
            lw=1
        ))

        patch = Line2D(
            [0], [0],
            marker='o',
            color='w',
            markerfacecolor=colors[box_counter],
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
    return mpld3.fig_to_html(plt.gcf())