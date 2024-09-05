import pandas as pd
import folium
from django.core.management.base import BaseCommand
from sensor.models import AirQualityData


class Command(BaseCommand):
    help = 'Plot a heatmap of a specific PM measurement based on latitude and longitude'

    def add_arguments(self, parser):
        parser.add_argument('pm_type', type=str, help="Type of PM measurement (e.g., 'pm1m', 'pm25m', 'pm4m', 'pm10m')")

    def handle(self, *args, **options):
        pm_type = options['pm_type']

        # Fetch data from the database
        data = AirQualityData.objects.values('lat', 'lon', pm_type)

        # Convert data to DataFrame
        df = pd.DataFrame.from_records(data)

        # Filter out rows where the PM value is NaN
        df = df.dropna(subset=[pm_type])

        # Drop rows with NaN in lat or lon
        df = df.dropna(subset=['lat', 'lon'])

        # Verifica se ainda há dados
        if df.empty:
            self.stdout.write(self.style.WARNING('No valid data to display. All selected PM values are NaN.'))
            return

        # Seleciona as colunas de latitude e longitude
        locations = df[['lat', 'lon']].values.tolist()
        pm_values = df[pm_type].values.tolist()

        # Configuração do mapa
        center = [-20.27931919525688, -40.287294463682024]
        m = folium.Map(location=center, zoom_start=12)

        # Adiciona a camada de calor
        from folium.plugins import HeatMap
        heat_data = [[loc[0], loc[1], pm] for loc, pm in zip(locations, pm_values)]
        HeatMap(heat_data, max_intensity=100, radius=8).add_to(m)

        # Salvar o mapa como HTML
        output_file = f'{pm_type}_heatmap.html'
        m.save(output_file)

        # Mensagem de sucesso
        self.stdout.write(self.style.SUCCESS(f'Heatmap for {pm_type} generated successfully and saved as {output_file}!'))
