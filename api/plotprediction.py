import io
import random
from PIL import Image
import matplotlib.pyplot as plt, mpld3  # nopep8 E401
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors


COLORS = list(mcolors.cnames.values())
LABELS = ('smoke', 'fire')


def plot_prediction(img, predictions: tuple) -> None:
    img = Image.open(io.BytesIO(img))
    patches = []
    colors = list(mcolors.cnames.values())
    _, ax = plt.subplots(figsize=(5, 5))
    plt.imshow(img)

    predictions_zip = zip(predictions[0], predictions[1], predictions[2])
    for box, score, label in predictions_zip:
        box_counter = random.randint(0, len(COLORS))
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
        handles=patches
    )

    return mpld3.fig_to_html(plt.gcf())
