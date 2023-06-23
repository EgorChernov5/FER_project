from django.contrib import admin
from django.urls import path, include

from main_app.views import main_view


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('main_app.urls')),
]
