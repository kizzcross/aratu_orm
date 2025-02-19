from django.shortcuts import render

# Create your views here.
import os
from django.http import JsonResponse
from django.core.management import call_command

def import_air_quality_data(request):
    try:
        call_command('import_air_quality_data')
        return JsonResponse({'status': 'success', 'message': 'Importação concluída com sucesso!'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)})
