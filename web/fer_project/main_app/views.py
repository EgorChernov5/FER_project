from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

from .utils import decode_image, preprocess_image, get_predict


def welcome(request):
    return render(request, 'main_app/welcome.html')


def result(request):
    return render(request, 'main_app/result.html')


def record(request):
    return render(request, 'main_app/record.html')


@csrf_exempt
def predict(request):
    response = None
    if request.method == 'POST':
        img = decode_image(request.body)
        img = preprocess_image(img)
        if img is not None:
            response = get_predict(img)

    return HttpResponse(response)
