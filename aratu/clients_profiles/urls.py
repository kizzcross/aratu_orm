"""
URL configuration for aratu project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import views
from .views import generate_heatmap, get_date_limits, create_cluster, define_regions, train_model, train_model_2, \
    define_regions_2

urlpatterns = [
    path('', views.home, name='home'),
    path('forecast/', views.previsao, name='previsao'),
    path('heatmap/', views.mapadecalor, name='mapadecalor'),
    path('generate-heatmap/', generate_heatmap, name='generate_heatmap'),
    path('report/', views.relatorio, name='relatorio'),
    path('data/', views.data, name='data'),

    path('date-limits/', get_date_limits, name='get_date_limits'),
    path('create-cluster/', create_cluster, name='create_cluster'),
    path('define-regions/', define_regions, name='define_regions'),
    path('define-regions-a/', define_regions_2, name='define_regions-a'),
    path('train-model/', train_model, name='train_model'),
    path('train-model-a/', train_model_2, name='train_model_a'),
    #----------------------------------------------------------------------------------------------------
    path('task-status/<str:task_id>/', views.task_status, name='task_status'),
    path('region-result/<int:region_id>/', views.get_region_result, name='get_region_result'),
    path('download-trained-model/<int:file_id>/', views.download_trained_csv, name='download_trained_model')
    #----------------------------------------------------------------------------------------------------
    #path('clients_profiles/forecast/cluster_geografico/', views.cluster_geografico, name='cluster_geografico'),
]
