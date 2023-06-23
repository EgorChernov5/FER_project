from django.db import models
from django.core.validators import MinValueValidator


class Emotion(models.Model):
    id_emotion = models.IntegerField(primary_key=True, verbose_name='ID Emotion')
    start = models.DateTimeField(auto_now_add=True, verbose_name='Start time')
    end = models.DateTimeField(auto_now=True, verbose_name='End time')
    is_record = models.BooleanField(default=True)
    angry = models.FloatField(validators=[MinValueValidator(0)], verbose_name='Angry time')
    disgust = models.FloatField(validators=[MinValueValidator(0)], verbose_name='Disgust time')
    fear = models.FloatField(validators=[MinValueValidator(0)], verbose_name='Fear time')
    happy = models.FloatField(validators=[MinValueValidator(0)], verbose_name='Happy time')
    neutral = models.FloatField(validators=[MinValueValidator(0)], verbose_name='Neutral time')
    sad = models.FloatField(validators=[MinValueValidator(0)], verbose_name='Sad time')
    surprise = models.FloatField(validators=[MinValueValidator(0)], verbose_name='Surprise time')
