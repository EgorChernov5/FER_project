import tensorflow as tf
import numpy as np
from pathlib import Path

from train import prepare
from ml.util.util import save_as_json


REPO_DIR = Path(__file__).parent.parent
CLASS_NAMES = np.array(['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])
AUTOTUNE = tf.data.experimental.AUTOTUNE
tf.random.set_seed(5)


def main():
    test_ds = prepare(source=REPO_DIR / "data/prepared/test.csv", batch_size=1024)

    model = tf.keras.models.load_model(REPO_DIR / "model/h5_format/baseline_v1/model.h5")

    test_loss, test_acc = model.evaluate(test_ds, verbose=2)
    save_as_json(path=(REPO_DIR / "metric/baseline_v1/test_loss.json"), data=test_loss)
    save_as_json(path=(REPO_DIR / "metric/baseline_v1/test_accuracy.json"), data=test_acc)


if __name__ == '__main__':
    main()
