import logging
from io import BytesIO
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms

LOGGER_NAME = "prediction"
LOGGER = logging.getLogger(LOGGER_NAME)


def load_model(model_path):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=False,
        num_classes=len(LABELS)+1
    )

    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    # TODO try execpt on model.eval
    model.eval()

    LOGGER.debug("Model loaded & evaulated %s", model_path)
    return model


MODEL_PATH = "model/model.pt"
LABELS = ('smoke', 'fire')
THRESHOLD = 0.80
MODEL = load_model(MODEL_PATH)


def preprocess_image(image):
    image = Image.open(BytesIO(image))
    transform = transforms.Compose([transforms.ToTensor()])

    pred_img = transform(image)
    pred_img = pred_img.view(1, 3, pred_img.shape[1], pred_img.shape[2])
    return pred_img


def get_relevant_scores(boxes: list, scores: list, labels: list) -> tuple:
    threshold = len([score for score in scores if score >= THRESHOLD])
    return boxes[:threshold], scores[:threshold], labels[:threshold]


def model_predict(image):
    preds = MODEL(image)
    outputs = [{k: v.to(torch.device('cpu')) for k, v in target.items()} for target in preds]

    boxes = outputs[0]['boxes'].data.cpu().numpy().astype(np.int32)
    scores = outputs[0]['scores'].data.cpu().numpy()
    labels = outputs[0]['labels'].data.cpu().numpy().astype(np.int32)

    boxes, scores, labels = get_relevant_scores(boxes, scores, labels)
    return boxes, scores, labels
