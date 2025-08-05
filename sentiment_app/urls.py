from django.urls import path
from . import views

app_name = 'sentiment_app'

urlpatterns = [
    path('', views.index, name='index'),
    path('analyze/<str:model_name>/', views.analyze, name='analyze'),
]