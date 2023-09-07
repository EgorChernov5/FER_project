import tensorflow as tf
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import math

from ml.util.util import open_json, save_as_json
from ml.src.train import prepare

REPO_DIR = Path(__file__).parent.parent
CLASS_NAMES = np.array(['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])
AUTOTUNE = tf.data.experimental.AUTOTUNE
tf.random.set_seed(5)
sns.set(style="whitegrid")


def display_distribution(ds):
    colors = sns.color_palette("deep")
    label_counter = np.zeros(7)
    for _, labels in ds:
        for label in labels:
            label_counter[label] = label_counter[label] + 1

    plt.bar(CLASS_NAMES, label_counter, color=colors)
    plt.show()


def display_batch(ds):
    image_batch, label_batch = next(iter(ds))
    plt.figure(figsize=(8, 6))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image_batch[i], cmap='gray')
        label = label_batch[i]
        plt.title(CLASS_NAMES[label])
        plt.axis("off")

    plt.show()


def display_conv_filter(filters, channel, save_path=None):
    """
    Display Conv2D filters.

    Args:
        filters: filters with shape: [kernel_height, kernel_width, input_channel, output_channel]
        channel: index of channel
        save_path: save plot or not
    """
    print(
        f'Kernel size - {filters.shape[:2]}; '
        f'number of filters - {filters.shape[-1]}; '
        f'number of input channels - {filters.shape[-2]};'
    )
    fig = plt.figure(figsize=(8, 6))
    n_filters = filters.shape[-1]
    rows = int(math.sqrt(n_filters))
    columns = int(math.ceil(n_filters / rows))
    for i in range(n_filters):
        f = filters[:, :, :, i]
        ax = fig.add_subplot(rows, columns, i + 1)
        plt.xticks([])
        plt.yticks([])
        ax.imshow(f[:, :, channel], cmap='gray')

    if save_path:
        plt.savefig(save_path)

    plt.show()


def display_conv_output(model, img, n, save_path=None):
    conv_layer = []
    for layer in model.layers:
        if "conv" in layer.name:
            conv_layer.append(layer.output)

    short_model = tf.keras.Model(inputs=model.inputs, outputs=conv_layer[:n])
    # Get last output
    outputs = short_model(img)[-1]
    if len(conv_layer[:n]) == 1:
        outputs = tf.expand_dims(outputs, axis=0)

    fig = plt.figure(figsize=(8, 6))
    n_channels = outputs.shape[-1]
    rows = int(math.sqrt(n_channels))
    columns = int(math.ceil(n_channels / rows))
    for i in range(n_channels):
        ax = fig.add_subplot(rows, columns, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.ylabel(i + 1)
        ax.imshow(outputs[0, :, :, i], cmap='gray')

    if save_path:
        plt.savefig(save_path)

    plt.show()


def get_actual_predicted_labels(model, dataset):
    """
    Create a list of actual ground truth values and the predictions from the model.

    Args:
        model: Model using for predictions
        dataset: An iterable data structure, such as a TensorFlow Dataset, with features and labels.

    Return:
        Ground truth and predicted values for a particular dataset.
    """
    actual = [labels for _, labels in dataset.unbatch()]
    predicted = model.predict(dataset)

    actual = tf.stack(actual, axis=0)
    predicted = tf.concat(predicted, axis=0)
    predicted = tf.argmax(predicted, axis=1)

    return actual, predicted


def plot_confusion_matrix(actual, predicted, labels, ds_type, save_path=None):
    cm = tf.math.confusion_matrix(actual, predicted)
    ax = sns.heatmap(cm, annot=True, fmt='g')
    sns.set(rc={'figure.figsize': (12, 12)})
    sns.set(font_scale=1.4)
    ax.set_title('Confusion matrix of action recognition for ' + ds_type)
    ax.set_xlabel('Predicted Action')
    ax.set_ylabel('Actual Action')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def calculate_classification_metrics(y_actual, y_pred, labels):
    """
    Calculate the precision and recall of a classification model using the ground truth and
    predicted values.

    Args:
        y_actual: Ground truth labels.
        y_pred: Predicted labels.
        labels: List of classification labels.

    Return:
        Precision and recall measures.
    """
    cm = tf.math.confusion_matrix(y_actual, y_pred)
    # Diagonal represents true positives
    tp = np.diag(cm)
    precision = dict()
    recall = dict()
    for i in range(len(labels)):
        # Sum of column minus true positive is false negative
        col = cm[:, i]
        fp = np.sum(col) - tp[i]
        # Sum of row minus true positive, is false negative
        row = cm[i, :]
        fn = np.sum(row) - tp[i]
        # Precision
        precision[labels[i]] = tp[i] / (tp[i] + fp)
        # Recall
        recall[labels[i]] = tp[i] / (tp[i] + fn)

    return precision, recall


def plot_history(history, save_path=None):
    """
    Plotting training and validation learning curves.

    Args:
        history: model history with all the metric measures
        save_path: where will save the plot
    """
    fig, (ax1, ax2) = plt.subplots(2)

    fig.set_size_inches(8, 6)
    plt.subplots_adjust(hspace=0.5)

    # Plot loss
    ax1.set_title('Loss')
    ax1.plot(history['loss'], label='train')
    ax1.plot(history['val_loss'], label='val')
    ax1.set_ylabel('Loss')

    # Determine upper bound of y-axis
    max_loss = max(history['loss'] + history['val_loss'])

    ax1.set_ylim([0, np.ceil(max_loss)])
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'])

    # Plot accuracy
    ax2.set_title('Accuracy')
    ax2.plot(history['accuracy'], label='train')
    ax2.plot(history['val_accuracy'], label='val')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim([0, 1])
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'])

    if save_path:
        plt.savefig(save_path)

    plt.show()


def main():
    test_ds = prepare(source=REPO_DIR / "data/prepared/test.csv", batch_size=1024)

    # Display data
    # display_batch(test_ds)

    # Display dataset distribution
    # display_distribution(test_ds)

    model = tf.keras.models.load_model(REPO_DIR / "model/h5_format/baseline_v1/model.h5")

    # Compute loss, accuracy
    history = open_json(REPO_DIR / f"metric/baseline_v1/history.json")
    plot_history(history, REPO_DIR / f"metric/baseline_v1/history.jpg")

    # Display model architecture
    # tf.keras.utils.plot_model(model, to_file=(REPO_DIR / "model/baseline_v2/structure.png"),
    #                           expand_nested=True, show_shapes=True)
    # print(model.summary())

    # Display confusion matrix
    actual, predicted = get_actual_predicted_labels(model, test_ds)
    plot_confusion_matrix(actual, predicted, CLASS_NAMES,
                          'test', save_path=(REPO_DIR / "metric/baseline_v1/confusion_matrix.png"))

    # Compute  precision, recall
    precision, recall = calculate_classification_metrics(actual, predicted, CLASS_NAMES)
    save_as_json(path=(REPO_DIR / "metric/baseline_v1/test_precision.json"), data=precision)
    save_as_json(path=(REPO_DIR / "metric/baseline_v1/test_recall.json"), data=recall)

    # Display conv filters
    # model.load_weights(REPO_DIR / "model/h5_format/baseline_v3/weight/weights.21-0.64.h5")
    # display_conv_filter(filters=model.layers[9]  # Get specific conv layer
    #                     .get_weights()[0],  # Get weights without bias
    #                     channel=0)

    # Display output after using conv filter
    img = next(iter(test_ds))[0][0]
    img = np.expand_dims(img, axis=0)
    display_conv_output(model, img, 6, save_path=(REPO_DIR / 'metric/baseline_v1/6_conv_output.jpg'))


if __name__ == '__main__':
    main()
