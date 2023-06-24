import tensorflow as tf
import cv2
import base64
from PIL import Image
import numpy as np
import io


faceCascade = cv2.CascadeClassifier('static/model/haarcascade_frontalface_default.xml')
CLASS_NAMES = np.array(['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])


def decode_image(b_image):
    img = io.BytesIO(base64.b64decode(b_image))
    img = Image.open(img).convert('RGB')
    return np.array(img)


def preprocess_image(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Crop image
    bbox = faceCascade.detectMultiScale(gray_image,
                                        scaleFactor=1.1,
                                        minNeighbors=5,
                                        minSize=(20, 20))
    if len(bbox) > 0:
        x, y, w, h = bbox[0]
        image = tf.image.rgb_to_grayscale(image)
        image = image[y:(y + h), x:(x + w), :]
        image = tf.image.resize(image, [48, 48])
        # Add batch size
        image = np.expand_dims(image, axis=0)
        return image

    return None


def get_predict(image):
    model = tf.keras.models.load_model("static/model/model.h5")
    y_pred = model.predict(tf.convert_to_tensor(image))
    predict = CLASS_NAMES[np.argmax(y_pred[0])]
    return predict
