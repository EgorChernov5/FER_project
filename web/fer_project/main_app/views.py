from django.shortcuts import render
from django.views.decorators import gzip
from django.http import StreamingHttpResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt

import io
import base64
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2


faceCascade = cv2.CascadeClassifier('static/model/haarcascade_frontalface_default.xml')


def welcome(request):
    return render(request, 'main_app/welcome.html')


def webcam(request):
    return render(request, 'main_app/webcam.html')


# @gzip.gzip_page
# def video_stream(request):
#     try:
#         cam = VideoCamera()
#         return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
#     except:  # This is bad! replace it with proper handling
#         print('Smthg wrong...')
#         pass
#
#     return render(request, 'main_app/welcome.html')


def test_webcam(request):
    return render(request, 'main_app/test_webcam.html')


@csrf_exempt
def test_hello(request):
    response = None
    if request.method == 'POST':
        # Decode and convert into np.array
        img = io.BytesIO(base64.b64decode(request.body))
        img = Image.open(img)
        img = np.array(img)

        # Convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Crop image
        face = faceCascade.detectMultiScale(img,
                                            scaleFactor=1.1,
                                            minNeighbors=5,
                                            minSize=(20, 20))
        x, y, w, h = face[0]
        img = img[y:(y + h), x:(x + w)]

        # model = tf.keras.models.load_model("model/model.h5")
        # model.predict()

        # cv2.imwrite('contour.png', img)
        print(img.shape)

        # ## converting RGB to BGR, as opencv standards
        # frame = cv2.cvtColor(np.array(img), cv2.G)
        #
        # # Process the image frame
        # imgencode = cv2.imencode('.jpg', frame)[1]
        #
        # # base64 encode
        # stringData = base64.b64encode(imgencode).decode('utf-8')
        # b64_src = 'data:image/jpg;base64,'
        # stringData = b64_src + stringData

        response = str([1, 2])
    return HttpResponse(response)
