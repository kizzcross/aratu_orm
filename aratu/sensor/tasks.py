import os
import json
import pandas as pd
import numpy as np
import psutil
import warnings
from celery import shared_task
from datetime import datetime
from collections import defaultdict
from django.db import transaction
from sensor.models import AirQualityData

# Ignore 'SettingWithCopyWarning'
warnings.filterwarnings('ignore')

# Define the source and destination folders
SOURCE_FOLDER = ''
DEST_FOLDER = ''


# Function to monitor resource usage
def monitor_usage():
    process = psutil.Process()
    memory_usage_mb = process.memory_info().rss / (1024 * 1024)  # Memory in MB
    cpu_usage_percent = process.cpu_percent(interval=None)  # CPU usage percentage
    return memory_usage_mb, cpu_usage_percent


def take_off_outliers_and_replace_with_mean(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    mean_value = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)][column].mean()
    df[column] = df[column].apply(lambda x: mean_value if x < lower_bound or x > upper_bound else x)
    return df


def threat_lat_and_lon_outliers(df, column):
    df = df[df[column] != 0]
    if column == 'lat':
        df[column] = df[column].apply(lambda x: x / 100000000.0 if x < -90 else x)
    if column == 'lon':
        df[column] = df[column].apply(lambda x: x / 100000000.0 if x < -180 else x)
    while df[column].abs().min() < 10:
        min_value = df[column].abs().min()
        df[column] = df[column].apply(lambda x: x * 10 if abs(x) == min_value else x)
    return df


@shared_task
def process_and_import_air_quality_data(file_path):
    start_memory, start_cpu = monitor_usage()

    # Load data from json
    with open(file_path) as f:
        data = json.load(f)

    # Create a DataFrame from the data
    air_quality_data = pd.DataFrame(data)

    # Pre-process the data (similar to the command)
    air_quality_data['temp'] = air_quality_data['temp'].apply(lambda x: str(x).replace(',', '.'))
    cols_to_convert_numeric = ['temp', 'umi']
    cols_to_convert_float = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'pm1m', 'pm25m', 'pm4m', 'pm10m', 'pm1n', 'pm25n',
                             'pm4n', 'pm10n', 'pts']
    for col in cols_to_convert_numeric:
        air_quality_data[col] = pd.to_numeric(air_quality_data[col], errors='coerce')
    for col in cols_to_convert_float:
        air_quality_data = air_quality_data[air_quality_data[col].apply(lambda x: isinstance(x, (int, float)))]
        air_quality_data[col] = pd.to_numeric(air_quality_data[col], errors='coerce')
    air_quality_data = air_quality_data.dropna(subset=['lat', 'lon'])

    # Fill NaN values with random normal distribution
    for column in air_quality_data.select_dtypes(include=['number']).columns:
        mean = air_quality_data[column].mean()
        std = air_quality_data[column].std()
        null_count = air_quality_data[column].isnull().sum()
        random_values = np.random.normal(mean, std, null_count)
        air_quality_data[column].fillna(pd.Series(random_values), inplace=True)

    air_quality_data.dropna(how='all', inplace=True)
    air_quality_data.reset_index(drop=True, inplace=True)
    air_quality_data.drop(columns=['vel', 'endereco'], inplace=True, errors='ignore')

    # Filter data
    air_quality_data['temp'] = air_quality_data['temp'].apply(lambda x: x / 100000000.0 if x > 50.0 else x)
    air_quality_data = air_quality_data[air_quality_data['temp'] >= 0]
    air_quality_data = air_quality_data[air_quality_data['umi'] <= 100]
    air_quality_data = air_quality_data[air_quality_data['temp'] <= 100]

    # Scale accelerometer and gyroscope values
    scale_factors = [1e12, 1e11, 1e10, 1e9, 1e8, 1e7, 1e6, 1e5, 1e4, 1e3, 1e2, 1e1]
    accelerometer_gyroscope_columns = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    for col in accelerometer_gyroscope_columns:
        for factor in scale_factors:
            air_quality_data[col] = air_quality_data[col].apply(lambda x: x / factor if abs(x) > factor else x)

    # Replace outliers
    columns_to_replace_with_mean = ['temp', 'umi', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'pm1m', 'pm25m', 'pm4m', 'pm10m',
                                    'pm1n', 'pm25n', 'pm4n', 'pm10n', 'pts']
    for column in columns_to_replace_with_mean:
        air_quality_data = take_off_outliers_and_replace_with_mean(air_quality_data, column)
    columns_to_replace_with_last_value = ['lat', 'lon']
    for column in columns_to_replace_with_last_value:
        air_quality_data = threat_lat_and_lon_outliers(air_quality_data, column)

    # Insert data into the Django model
    with transaction.atomic():
        # AirQualityData.objects.all().delete()  # Optional: Clear old data
        for _, row in air_quality_data.iterrows():
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
                address="Unknown"
            )

    end_memory, end_cpu = monitor_usage()
    memory_used = end_memory - start_memory
    cpu_used = end_cpu - start_cpu
    print(f"Memory used: {memory_used:.2f} MB, CPU used: {cpu_used:.2f} %")
    return f"Data processing completed with {len(air_quality_data)} rows inserted."
