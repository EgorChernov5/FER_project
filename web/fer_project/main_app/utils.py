import requests
import os
from dotenv import load_dotenv
import json
import cv2
import base64
from PIL import Image
import numpy as np
import io
from pathlib import Path


PROJECT_DIR = Path(__file__).parent.parent
load_dotenv()
faceCascade = cv2.CascadeClassifier(str(PROJECT_DIR / 'static/model/haarcascade_frontalface_default.xml'))
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
    img = Image.open(img).convert('L')
    return np.array(img)


def preprocess_image(image):
    """
    Process image for the model format.

    Args:
        image: image (np.array).

    Return:
        Cropped scaled and resized image or None.
    """
    # Get the bounding box
    bbox = faceCascade.detectMultiScale(image,
                                        scaleFactor=1.1,
                                        minNeighbors=5,
                                        minSize=(20, 20))
    if len(bbox):
        x, y, w, h = bbox[0]
        # Crop the image
        image = image[y:(y + h), x:(x + w)]
        # Scale the image
        image = image / 255.0
        # Resize the image
        image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_AREA)
        # Add channel`s number and batch size
        image = image[:, :, np.newaxis]
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
    data = {"instances": image.tolist()}
    data = json.dumps(data)
    y_pred = requests.post(os.getenv('URL'), data=data)
    y_pred = np.array(y_pred.json()['predictions'][0])
    predict = CLASS_NAMES[np.argmax(y_pred)]
    return predict
