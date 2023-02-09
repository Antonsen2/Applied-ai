"""Downloader downloads the pretrained model resnet50 from pytorch

    This function is used in the docker build step to
    avoid downloading every for deployment and instead
    is part of the image.
"""
from torch import hub


def downloader():
    hub.load_state_dict_from_url(
            "https://download.pytorch.org/models/resnet50-0676ba61.pth")


if __name__ == '__main__':
    downloader()
