from django.urls import path
from . import views

urlpatterns = [
    path('import/', views.import_air_quality_data, name='import_air_quality_data'),
]
