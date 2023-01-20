import torch

from vision.references.detection.engine import train_one_epoch, evaluate
from torch.utils.data import DataLoader
from torchvision.models.detection import faster_rcnn, fasterrcnn_resnet50_fpn

from neuralnetworks.RCNN import config
from neuralnetworks.RCNN.dataset import WildfireDataset

# TODO: Test train in VSC

class RCNN:
    def __init__(self) -> None:
        self.train_loader = DataLoader(
            dataset=WildfireDataset(
                df=config.TRAIN_DF,
                img_path=config.TRAIN_IMAGE_PATH,
                labels=config.LABELS,
                transforms=config.TRAIN_TRANSFORM
            ),
            batch_size=4,
            shuffle=True,
            num_workers=2,
            collate_fn=self.collate_fn
        )
        self.valid_loader = DataLoader(
            dataset=WildfireDataset(
                df=config.VAL_DF,
                img_path=config.VAL_IMAGE_PATH,
                labels=config.LABELS,
                transforms=config.VAL_TRANSFORM
            ),
            batch_size=4,
            shuffle=False,
            num_workers=2,
            collate_fn=self.collate_fn
        )
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(
            in_channels=self.in_features,
            num_classes=config.NUM_OF_CLASSES
        )
        self.model.to(config.DEVICE)

    @staticmethod
    def collate_fn(batch: tuple) -> tuple:
        return tuple(zip(*batch))

    def train(self, epochs: int = 20) -> None:
        params = [param for param in self.model.parameters() if param.requires_grad]
        optimizer = torch.optim.SGD(
            params=params,
            lr=0.005,
            momentum=0.9,
            weight_decay=0.0005
        )
        lr_schedule = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=3,
            gamma=0.1
        )

        for epoch in range(epochs):
            train_one_epoch(
                model=self.model,
                optimizer=optimizer,
                data_loader=self.train_loader,
                device=config.DEVICE,
                epoch=epoch,
                print_freq=len(self.train_loader)
            )

            lr_schedule.step()
            evaluate(
                model=self.model,
                data_loader=self.valid_loader,
                device=config.DEVICE
            )

    def save(self, save_path: str) -> None:
        torch.save(self.model.state_dict(), save_path)

    def load(self, load_path: str) -> None:
        self.model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=config.NUM_OF_CLASSES)
        self.in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        self.model.load_state_dict(torch.load(load_path))
        self.model.to(config.DEVICE)
        self.model.eval()

    def predict(self) -> None:
        # TODO: Add single image/folder with images prediction function
        pass
