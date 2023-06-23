import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
import os

from util.util import save_as_json


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
REPO_DIR = Path(__file__).parent.parent
CLASS_NAMES = np.array(['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])
AUTOTUNE = tf.data.experimental.AUTOTUNE
tf.random.set_seed(5)


def get_path(csv_file):
    images_df = pd.read_csv(csv_file)
    images_df = images_df['path'].apply(lambda path: str(REPO_DIR / path).replace('\\', '/'))
    paths_ds = tf.data.Dataset.from_tensor_slices(images_df)
    paths_ds = paths_ds.shuffle(buffer_size=len(paths_ds), reshuffle_each_iteration=False)
    return paths_ds


def get_label(file_path):
    # Convert the path to a list of path components
    parts = tf.strings.split(file_path, '/')
    # The second to last is the class-directory
    one_hot = parts[-2] == CLASS_NAMES
    # Integer encode the label
    return tf.argmax(one_hot)


def preprocess_img(img):
    img = tf.image.resize(img, [48, 48])
    img = tf.cast(img, dtype=tf.float32) / tf.constant(256, dtype=tf.float32)
    return img


def decode_img(img):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_image(img, channels=1, expand_animations=False)
    img = preprocess_img(img)
    return img


def process_path(file_path):
    label = get_label(file_path)
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


def configure_for_performance(ds, shuffle=True, reshuffle=False, batch_size=1000):
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=1024, reshuffle_each_iteration=reshuffle)

    ds = (ds
          .batch(batch_size, num_parallel_calls=AUTOTUNE)
          .prefetch(buffer_size=AUTOTUNE))
    return ds


def create_model():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=11,
                               kernel_size=(3, 3),
                               activation='relu',
                               input_shape=(48, 48, 1)),
        tf.keras.layers.Conv2D(filters=11,
                               kernel_size=(3, 3),
                               activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=11,
                               kernel_size=(3, 3),
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=11,
                               kernel_size=(3, 3),
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=11,
                               kernel_size=(3, 3),
                               activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(7, activation='softmax'),
    ])


def main():
    train_ds = get_path(REPO_DIR / "data/prepared/train.csv")
    train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    train_ds = configure_for_performance(train_ds, batch_size=1500)

    val_ds = get_path(REPO_DIR / "data/prepared/val.csv")
    val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    val_ds = configure_for_performance(val_ds, shuffle=False, batch_size=1000)

    model = create_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    # print(model.summary())

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=(REPO_DIR / "model/baseline_v2/weight/weights.{epoch:02d}-{val_accuracy:.2f}.h5"),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        save_weights_only=True,
        save_freq='epoch'
    )

    history = model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=[checkpoint], batch_size=100, verbose=1)
    save_as_json(path=(REPO_DIR / f"metric/baseline_v2/history.json"), data=history.history)
    model.save(REPO_DIR / "model/baseline_v2/model.h5")


if __name__ == "__main__":
    main()
