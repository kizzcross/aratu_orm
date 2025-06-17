from django.shortcuts import render

# Create your views here.
import os
from django.http import JsonResponse
from django.core.management import call_command

import logging
logger = logging.getLogger(__name__)

def import_air_quality_data(request):
    try:
        call_command('import_air_quality_data')
        return JsonResponse({'status': 'success', 'message': 'Importação concluída com sucesso!'})
    except Exception as e:
        logger.exception("Erro na importação dos dados")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)