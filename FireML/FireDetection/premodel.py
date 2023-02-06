from torch import hub


def downloader():
    hub.load_state_dict_from_url("https://download.pytorch.org/models/resnet50-0676ba61.pth")


if __name__ == '__main__':
    downloader()
