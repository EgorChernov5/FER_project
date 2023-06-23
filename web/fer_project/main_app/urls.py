from django.urls import path

from .views import *


urlpatterns = [
    path('', main_view, name='main'),
    path('webcam/stream/', video_stream, name='stream'),
    path('webcam/', webcam, name='webcam'),
    path('fer_project/', test, name='welcome'),
]


