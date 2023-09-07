import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
import os

from ml.util.util import save_as_json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
REPO_DIR = Path(__file__).parent.parent
CLASS_NAMES = np.array(['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])
AUTOTUNE = tf.data.experimental.AUTOTUNE
tf.random.set_seed(5)


def get_path(csv_file, spec_sample=''):
    images_df = pd.read_csv(csv_file)
    if spec_sample:
        images_df = images_df[images_df['label'] == spec_sample]
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


def process_image(img, label):
    p_img = tf.image.rgb_to_grayscale(img)
    p_img = tf.image.resize(p_img, [48, 48])
    return p_img, label


def process_path(file_path):
    label = get_label(file_path)
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = tf.io.decode_image(img, channels=3, dtype=tf.float32, expand_animations=False)
    return img, label


def configure_for_performance(ds, batch_size=512, test=False, reshuffle=False):
    if not test:
        ds = (ds
              .cache()
              .shuffle(buffer_size=1024, reshuffle_each_iteration=reshuffle))

    ds = (ds
          .batch(batch_size, num_parallel_calls=AUTOTUNE)
          .prefetch(buffer_size=AUTOTUNE))
    return ds


def augment(image, label):
    augment_data = tf.keras.Sequential([
        tf.keras.layers.RandomFlip(mode='horizontal'),
        tf.keras.layers.RandomZoom(height_factor=(-0.1, -0.2),
                                   width_factor=(-0.1, -0.2)),
        tf.keras.layers.RandomRotation(
            factor=(-0.2, 0.2),
            fill_mode='constant',
            fill_value=0.),
        tf.keras.layers.RandomBrightness(factor=0.3, value_range=(0.0, 1.0)),
    ])
    return augment_data(image, training=True), label


def prepare(source, batch_size, train=False, aug_label=None, reshuffle=False):
    dataset = get_path(source)
    dataset = dataset.map(process_path, num_parallel_calls=AUTOTUNE)
    if aug_label:
        if aug_label == 'all':
            dataset = dataset.map(augment, num_parallel_calls=AUTOTUNE)
        else:
            aug_ds = get_path(source, aug_label)
            aug_ds = aug_ds.map(process_path, num_parallel_calls=AUTOTUNE)
            aug_ds = aug_ds.map(augment, num_parallel_calls=AUTOTUNE)
            dataset = dataset.concatenate(aug_ds)

    dataset = dataset.map(process_image, num_parallel_calls=AUTOTUNE)
    if train:
        return configure_for_performance(dataset, batch_size=batch_size, reshuffle=reshuffle)

    return configure_for_performance(dataset, batch_size=batch_size, test=True, reshuffle=reshuffle)


def create_model():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=16,
                               kernel_size=(3, 3),
                               activation='relu',
                               input_shape=(48, 48, 1)),
        tf.keras.layers.Conv2D(filters=16,
                               kernel_size=(3, 3),
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=16,
                               kernel_size=(3, 3),
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=16,
                               kernel_size=(3, 3),
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=16,
                               kernel_size=(3, 3),
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=16,
                               kernel_size=(3, 3),
                               activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(7, activation='softmax'),
    ])


def scheduler(epoch, lr):
    if epoch <= 10:
        return lr
    else:
        return 1e-4


def main():
    # Prepare train dataset
    train_ds = prepare(source=REPO_DIR / "data/prepared/train.csv", batch_size=128, train=True)
    # Prepare val dataset
    val_ds = prepare(source=REPO_DIR / "data/prepared/val.csv", batch_size=256)

    model = create_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    # print(model.summary())

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=(REPO_DIR / "model/h5_format/baseline_v1/weight/weights.{epoch:02d}-{val_accuracy:.2f}.h5"),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        save_weights_only=True,
        save_freq='epoch'
    )
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

    history = model.fit(train_ds, epochs=30, verbose=1, callbacks=[checkpoint], validation_data=val_ds)
    # del history.history['lr']
    save_as_json(path=(REPO_DIR / f"metric/baseline_v1/history.json"), data=history.history)
    model.save(REPO_DIR / "model/sv_format/saved_model/1")
    model.save(REPO_DIR / "model/h5_format/baseline_v1/model.h5")


if __name__ == "__main__":
    main()
