import tensorflow as tf
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from train import get_path, process_path, configure_for_performance


REPO_DIR = Path(__file__).parent.parent
CLASS_NAMES = np.array(['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])
AUTOTUNE = tf.data.experimental.AUTOTUNE


def display_batch(ds):
    image_batch, label_batch = next(iter(ds))
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        label = label_batch[i]
        plt.title(CLASS_NAMES[label])
        plt.axis("off")
    plt.show()


def main():
    train_ds = get_path(REPO_DIR / "data/prepared/train.csv")
    train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    train_ds = configure_for_performance(train_ds)
    display_batch(train_ds)


if __name__ == '__main__':
    main()
