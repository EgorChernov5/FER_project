from pathlib import Path
import json
import tensorflow as tf


REPO_DIR = Path(__file__).parent.parent


def open_json(path):
    data = None
    with open(path, 'r') as json_file:
        data = json.load(json_file)
    return data


def save_as_json(path, data):
    with open(path, "w") as json_file:
        json.dump(data, json_file)


def rescale(image, a=0., b=1.):
    max_pixel = tf.math.reduce_max(image)
    min_pixel = tf.math.reduce_min(image)
    rescale_image = (image - min_pixel) / (max_pixel - min_pixel)
    return rescale_image * (b - a) + a
