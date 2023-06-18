import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
import os

import time


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


def decode_img(img):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_image(img, channels=1, expand_animations=False)
    img = tf.image.resize(img, [48, 48])
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


class FERBaselineModel(tf.keras.Model):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.d1 = tf.keras.layers.Dense(128, activation='relu', input_shape=(48, 48, 1))
        self.f = tf.keras.layers.Flatten()
        self.d2 = tf.keras.layers.Dense(7, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.d1(inputs)
        x = self.f(x)
        return self.d2(x)


@tf.function
def train_step(images, labels, model, loss_object, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def val_step(images, labels, model, loss_object, val_loss, val_accuracy):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    val_loss(t_loss)
    val_accuracy(labels, predictions)


def main():
    train_ds = get_path(REPO_DIR / "data/prepared/train.csv")
    train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    train_ds = configure_for_performance(train_ds, batch_size=1500)

    val_ds = get_path(REPO_DIR / "data/prepared/val.csv")
    val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    val_ds = configure_for_performance(val_ds, shuffle=False, batch_size=1000)
    # Create an instance of the model
    model = FERBaselineModel()
    # Choose an optimizer and loss function for training
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    # Select metrics to measure the loss and the accuracy of the model
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
    # Train model
    EPOCHS = 5
    for epoch in range(EPOCHS):
        # Start timer for each epoch
        start_time = time.perf_counter()
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()

        val_loss.reset_states()
        val_accuracy.reset_states()

        for images, labels in train_ds:
            train_step(images, labels, model, loss_object, optimizer, train_loss, train_accuracy)

        for val_images, val_labels in val_ds:
            val_step(val_images, val_labels, model, loss_object, val_loss, val_accuracy)

        print(
            f'Epoch {epoch + 1}, time - {round(time.perf_counter() - start_time, 3)}s: '
            f'train loss - {round(train_loss.result(), 3)}; '
            f'train accuracy - {round(train_accuracy.result(), 3)}; '
            f'val loss - {round(val_loss.result(), 3)}; '
            f'val accuracy - {round(val_accuracy.result(), 3)}'
        )


if __name__ == "__main__":
    main()
