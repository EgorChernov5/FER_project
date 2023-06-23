import tensorflow as tf
import numpy as np
from pathlib import Path
from train import get_path, process_path
from util.util import save_as_json


REPO_DIR = Path(__file__).parent.parent
CLASS_NAMES = np.array(['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])
AUTOTUNE = tf.data.experimental.AUTOTUNE
tf.random.set_seed(5)


def main():
    test_ds = get_path(REPO_DIR / "data/prepared/val.csv")
    test_ds = (test_ds
               .map(process_path, num_parallel_calls=AUTOTUNE)
               .batch(100, num_parallel_calls=AUTOTUNE)
               .prefetch(buffer_size=AUTOTUNE))

    model = tf.keras.models.load_model(REPO_DIR / "model/baseline_v2/model.h5")

    test_loss, test_acc = model.evaluate(test_ds, verbose=2)
    save_as_json(path=(REPO_DIR / "metric/baseline_v2/test_loss.json"), data=test_loss)
    save_as_json(path=(REPO_DIR / "metric/baseline_v2/test_accuracy.json"), data=test_acc)


if __name__ == '__main__':
    main()
