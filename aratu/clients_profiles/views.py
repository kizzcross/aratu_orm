from django.shortcuts import render
from django.http import JsonResponse
from django.db.models import Min, Max
import pandas as pd
import folium
from folium.plugins import HeatMap
from sensor.models import AirQualityData  # Certifique-se de que o modelo está correto
from django.contrib.auth.decorators import permission_required
from django.core.exceptions import PermissionDenied

# Lista de tipos de PM válidos
VALID_PM_TYPES = ['pm1m', 'pm25m', 'pm4m', 'pm10m']

def home(request):
    return render(request, 'clients_profiles/home.html')

def previsao(request):
    return render(request, 'clients_profiles/previsao.html')

def mapadecalor(request):
    return render(request, 'clients_profiles/mapadecalor.html')

def relatorio(request):
    return render(request, 'clients_profiles/relatorio.html')

def data(request):
    return render(request, 'clients_profiles/data.html')

# Endpoint para obter limites de datas, cluisterizar e criar modelo
def get_date_limits(request):
    try:
        date_limits = AirQualityData.objects.aggregate(
            min_date=Min('measure_time'),
            max_date=Max('measure_time')
        )

        min_date = date_limits.get('min_date')
        max_date = date_limits.get('max_date')

        return JsonResponse({'start_date': str(min_date), 'end_date': str(max_date)})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

# Endpoint para criar cluster geográfico
def create_cluster(request):
    if request.method == 'POST':
        # Lógica para criar o cluster
        return JsonResponse({'message': 'Cluster geográfico criado com sucesso!'})
    return JsonResponse({'error': 'Método não permitido'}, status=405)

# Endpoint para definir regiões
def define_regions(request):
    if request.method == 'POST':
        # Lógica para definir regiões
        return JsonResponse({'message': 'Regiões definidas com sucesso!'})
    return JsonResponse({'error': 'Método não permitido'}, status=405)

# Endpoint para treinar modelo
def train_model(request):
    if request.method == 'POST':
        # Lógica para treinar o modelo
        return JsonResponse({'message': 'Modelo treinado com sucesso!'})
    return JsonResponse({'error': 'Método não permitido'}, status=405)

@permission_required('clients_profiles.view_airqualitydata', raise_exception=True)
def generate_heatmap(request):
    try:
        pm_type = request.GET.get('pm_type', 'pm25m')  # Padrão: 'pm25m'

        # Verificar se o tipo de PM é válido
        if pm_type not in VALID_PM_TYPES:
            return JsonResponse({'error': f'Invalid PM type. Valid types are: {", ".join(VALID_PM_TYPES)}'})

        # Buscar os dados do banco de dados
        data = AirQualityData.objects.values('lat', 'lon', pm_type)

        # Converter os dados para um DataFrame
        df = pd.DataFrame.from_records(data)

        # Remover valores NaN em PM, latitude e longitude
        df = df.dropna(subset=[pm_type, 'lat', 'lon'])

        # Verificar se há dados válidos
        if df.empty:
            return JsonResponse({'error': 'No valid data to display. All selected PM values are NaN.'})

        # Selecionar os dados de localização e valores
        locations = df[['lat', 'lon']].values.tolist()
        pm_values = df[pm_type].values.tolist()

        # Configurar o mapa
        center = [-20.27931919525688, -40.287294463682024]  # Ajuste para a localização desejada
        m = folium.Map(
            location=center,
            zoom_start=12,
            width='100%',
            height='70%'
        )

        # Adicionar camada de calor
        heat_data = [[loc[0], loc[1], pm] for loc, pm in zip(locations, pm_values)]
        HeatMap(heat_data, max_intensity=100, radius=8).add_to(m)

        # Renderizar o mapa como HTML
        map_html = m._repr_html_()

    except PermissionDenied:
        # Se o usuário não tiver permissão, retornamos um erro personalizado em JSON
        return JsonResponse({'error': 'Você não tem permissão para gerar o heatmap!'}, status=403)

    # Retornar o HTML do mapa como parte da resposta JSON
    return JsonResponse({'map_html': map_html})

