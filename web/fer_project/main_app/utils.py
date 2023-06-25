import tensorflow as tf
import cv2
import base64
from PIL import Image
import numpy as np
import io


faceCascade = cv2.CascadeClassifier('static/model/haarcascade_frontalface_default.xml')
CLASS_NAMES = np.array(['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])


def decode_image(b_image):
    """
    Decode bytes to image.

    Args:
        b_image: bytes (base64).

    Return:
         Numpy array image.
    """
    img = io.BytesIO(base64.b64decode(b_image))
    img = Image.open(img).convert('RGB')
    return np.array(img)


def preprocess_image(image):
    """
    Process image for the model format.

    Args:
        image: image (np.array).

    Return:
        Cropped scaled and resized image or None.
    """
    # Convert to grayscale for detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Get the bounding box
    bbox = faceCascade.detectMultiScale(gray_image,
                                        scaleFactor=1.1,
                                        minNeighbors=5,
                                        minSize=(20, 20))
    if len(bbox):
        x, y, w, h = bbox[0]
        # Convert to grayscale for classification
        image = tf.image.rgb_to_grayscale(image)
        # Crop the image
        image = image[y:(y + h), x:(x + w), :]
        # Resize the image
        image = tf.image.resize(image, [48, 48])
        # Scale the image
        image = tf.cast(image, dtype=tf.float32) / tf.constant(256, dtype=tf.float32)
        # Add batch size
        image = np.expand_dims(image, axis=0)
        return image

    return None


def get_predict(image):
    """
    Get predict.

    Args:
        image: image [b, h, w, c].

    Return:
         Predict (str).
    """
    model = tf.keras.models.load_model("static/model/model.h5")
    y_pred = model.predict(tf.convert_to_tensor(image))
    predict = CLASS_NAMES[np.argmax(y_pred[0])]
    return predict
