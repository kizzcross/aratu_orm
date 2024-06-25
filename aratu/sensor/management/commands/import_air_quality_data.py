import sys
import psutil
import pandas as pd
import numpy as np
import warnings
from django.core.management.base import BaseCommand

from sensor.models import AirQualityData

# Ignore 'SettingWithCopyWarning'
warnings.filterwarnings('ignore')


# Function to monitor resource usage
def monitor_usage():
    process = psutil.Process()
    memory_usage_mb = process.memory_info().rss / (1024 * 1024)  # Memory in MB
    cpu_usage_percent = process.cpu_percent(interval=None)  # CPU usage percentage
    return memory_usage_mb, cpu_usage_percent


class Command(BaseCommand):
    help = 'Process and import air quality data into the AirQualityData model'

    def handle(self, *args, **kwargs):
        self.stdout.write(self.style.SUCCESS('Starting data processing...'))

        # Start monitoring resources
        start_memory, start_cpu = monitor_usage()

        # Load data from Excel
        file_path = "static/MQA_DATA.xlsx"
        air_quality_data = pd.read_excel(file_path)
        self.stdout.write(self.style.SUCCESS("Excel imported to DataFrame"))

        # Convert specified columns to numeric
        cols_to_convert = ['temp', 'umi', 'lat', 'lon', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'pm1m', 'pm25m', 'pm4m',
                           'pm10m', 'pm1n', 'pm25n', 'pm4n', 'pm10n', 'pts']
        for col in cols_to_convert:
            air_quality_data[col] = pd.to_numeric(air_quality_data[col], errors='coerce')

        # Fill NaN values with random normal distribution
        for column in air_quality_data.select_dtypes(include=['number']).columns:
            mean = air_quality_data[column].mean()
            std = air_quality_data[column].std()
            null_count = air_quality_data[column].isnull().sum()
            random_values = np.random.normal(mean, std, null_count)
            air_quality_data[column].fillna(pd.Series(random_values), inplace=True)

        air_quality_data.dropna(how='all', inplace=True)
        air_quality_data.reset_index(drop=True, inplace=True)

        # Drop 'vel' and 'endereco' columns
        air_quality_data.drop(columns=['vel', 'endereco'], inplace=True)
        air_quality_data.reset_index(drop=True, inplace=True)

        air_quality_data = air_quality_data.dropna()
        self.stdout.write(self.style.SUCCESS("Dataframe pre-filtered"))

        # Filter and adjust data
        threshold = 50.0
        air_quality_data['temp'] = air_quality_data['temp'].apply(lambda x: x / 100000000.0 if x > threshold else x)
        air_quality_data = air_quality_data[air_quality_data['temp'] >= 0]
        air_quality_data = air_quality_data[air_quality_data['umi'] <= 100]
        air_quality_data = air_quality_data[air_quality_data['temp'] <= 100]

        # Scale accelerometer and gyroscope values
        accelerometer_gyroscope_columns = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
        scale_factors = [1e12, 1e11, 1e10, 1e9, 1e8, 1e7, 1e6, 1e5, 1e4, 1e3, 1e2, 1e1]
        for col in accelerometer_gyroscope_columns:
            for factor in scale_factors:
                air_quality_data[col] = air_quality_data[col].apply(lambda x: x / factor if abs(x) > factor else x)

        # Additional latitude and longitude scaling
        air_quality_data['lat'] = air_quality_data['lat'].apply(lambda x: x / 100000000.0 if x < -2000000000.0 else x)
        air_quality_data['lon'] = air_quality_data['lon'].apply(lambda x: x / 1000000000.0 if x < -5000000000.0 else x)

        # Scale particulate matter values
        particulate_columns = ['pm1m', 'pm25m', 'pm4m', 'pm10m', 'pm1n', 'pm25n', 'pm4n', 'pm10n']
        for col in particulate_columns:
            for factor in scale_factors:
                air_quality_data[col] = air_quality_data[col].apply(lambda x: x / factor if abs(x) > factor else x)

        air_quality_data['time'] = air_quality_data['time'].astype(str)
        air_quality_data = air_quality_data.round(2)

        self.stdout.write(self.style.SUCCESS("Dataframe filtered and processed"))

        self.stdout.write(f"Initial data shape: {air_quality_data.shape[0]} rows")

        # Resource usage during data processing
        partial_memory, partial_cpu = monitor_usage()
        memory_used = partial_memory - start_memory
        cpu_used = partial_cpu - start_cpu
        self.stdout.write(f"\nMemory used: {memory_used:.2f} MB")
        self.stdout.write(f"CPU used: {cpu_used:.2f} %\n")

        # Insert data into the Django model
        for index, row in air_quality_data.iterrows():
            AirQualityData.objects.create(
                measure_time=row['time'],
                temperature=row['temp'],
                humidity=row['umi'],
                lat=row['lat'],
                lon=row['lon'],
                ax=row['ax'],
                ay=row['ay'],
                az=row['az'],
                gx=row['gx'],
                gy=row['gy'],
                gz=row['gz'],
                pm1m=row['pm1m'],
                pm25m=row['pm25m'],
                pm4m=row['pm4m'],
                pm10m=row['pm10m'],
                pm1n=row['pm1n'],
                pm25n=row['pm25n'],
                pm4n=row['pm4n'],
                pm10n=row['pm10n'],
                pts=row['pts'],
                address="Unknown"  # Assuming address is not provided in your DataFrame
            )

        self.stdout.write(self.style.SUCCESS("Data inserted into the database"))

        # Resource usage after data processing and insertion
        end_memory, end_cpu = monitor_usage()
        memory_used = end_memory - start_memory
        cpu_used = end_cpu - start_cpu
        self.stdout.write(f"Final memory used: {memory_used:.2f} MB")
        self.stdout.write(f"Final CPU used: {cpu_used:.2f} %")
        self.stdout.write(self.style.SUCCESS("Data processing completed"))
