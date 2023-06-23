from django.shortcuts import render
from django.views.decorators import gzip
from django.http import StreamingHttpResponse

from .utils import VideoCamera, gen


def main_view(request):
    return render(request, 'main_app/main.html')


def test(request):
    return render(request, 'main_app/welcome.html')


def webcam(request):
    return render(request, 'main_app/webcam.html')


@gzip.gzip_page
def video_stream(request):
    try:
        cam = VideoCamera()
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:  # This is bad! replace it with proper handling
        pass

    return render(request, 'main_app/main.html')
