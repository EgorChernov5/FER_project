from django.urls import path

from .views import *


urlpatterns = [
    path('', welcome),
    path('fer_project/', welcome, name='welcome'),
    path('result/', result, name='result'),
    path('record/', record, name='record'),
    path('predict/', predict, name="predict"),
]
