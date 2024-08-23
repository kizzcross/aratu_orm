import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
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
        print(df.head())

        # Filter out rows where the PM value is NaN
        df = df.dropna(subset=[pm_type])

        # Drop rows with NaN in lat or lon
        df = df.dropna(subset=['lat', 'lon'])

        # Check if there is any data left after filtering
        if df.empty:
            self.stdout.write(self.style.WARNING('No valid data to display. All selected PM values are NaN.'))
            return

        # Generate grid data
        lat_grid, lon_grid = np.mgrid[df['lat'].min():df['lat'].max():100j, df['lon'].min():df['lon'].max():100j]

        # Interpolate the values on the grid
        grid_values = griddata((df['lat'], df['lon']), df[pm_type], (lat_grid, lon_grid), method='cubic')
        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(grid_values.T, origin='lower', cmap='coolwarm',
                   extent=[df['lon'].min(), df['lon'].max(), df['lat'].min(), df['lat'].max()])
        plt.colorbar(label=pm_type)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f'Heatmap of {pm_type} by Latitude and Longitude')
        plt.show()

        self.stdout.write(self.style.SUCCESS(f'Heatmap for {pm_type} generated successfully!'))
