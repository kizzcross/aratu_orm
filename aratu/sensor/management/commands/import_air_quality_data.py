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



def take_off_outliers_and_replace_with_mean(df, column):
    """
    This function takes a DataFrame and a column name as input and removes the outliers from the column.
    It replaces the outliers with the mean of the column.
    """
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    # Calculate IQR (Interquartile Range)
    IQR = Q3 - Q1
    # Define the bounds for identifying outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Calculate the mean of the non-outlier values
    mean_value = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)][column].mean()

    # Replace outliers with the mean
    df[column] = df[column].apply(lambda x: mean_value if x < lower_bound or x > upper_bound else x)

    return df

def threat_lat_and_lon_outliers(df, column):
    # take off rows that are 0
    df = df[df[column] != 0]
    # Sometimes latitude and longitude values are missing a 0 in the last digit (e.g.,-202540741 instead of -20.25407410)
    if column == 'lat':
        df[column] = df[column].apply(lambda x: x / 100000000.0 if x < -90 else x)
    if column == 'lon':
        df[column] = df[column].apply(lambda x: x / 100000000.0 if x < -180 else x)
    # if its abs is less than 10 keep multiplying by 10
    while df[column].abs().min() < 10:
        # take the minimum value
        min_value = df[column].abs().min()
        # multiply by 10
        df[column] = df[column].apply(lambda x: x * 10 if abs(x) == min_value else x)
    return df

class Command(BaseCommand):
    help = 'Process and import air quality data into the AirQualityData model'

    def handle(self, *args, **kwargs):
        #delete all
        AirQualityData.objects.all().delete()
        self.stdout.write(self.style.SUCCESS('Starting data processing...'))

        # Start monitoring resources
        start_memory, start_cpu = monitor_usage()

        # Load data from Excel
        file_path = "../../aratu/MQA_DATA.xlsx"
        air_quality_data = pd.read_excel(file_path)
        self.stdout.write(self.style.SUCCESS("Excel imported to DataFrame"))

        # change the temp to .
        air_quality_data['temp'] = air_quality_data['temp'].apply(lambda x: str(x).replace(',', '.'))
        # Convert specified columns to numeric
        cols_to_convert_numeric = ['temp','umi']
        cols_to_convert_float = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'pm1m', 'pm25m', 'pm4m',
                           'pm10m', 'pm1n', 'pm25n', 'pm4n', 'pm10n', 'pts']
        for col in cols_to_convert_numeric:
            air_quality_data[col] = pd.to_numeric(air_quality_data[col], errors='coerce')
        for col in cols_to_convert_float:
            # drop the rows that have a string value
            air_quality_data = air_quality_data[air_quality_data[col].apply(lambda x: isinstance(x, (int, np.int64, float, np.float64)))]
            air_quality_data[col] = pd.to_numeric(air_quality_data[col], errors='coerce')
        # drop the rows that have null in lat or lon
        air_quality_data = air_quality_data.dropna(subset=['lat', 'lon'])
        #print lon and lat

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
        # coloumns that outliers will be replaced with the mean
        columns_to_replace_with_mean = ['temp', 'umi', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'pm1m', 'pm25m', 'pm4m',
                           'pm10m', 'pm1n', 'pm25n', 'pm4n', 'pm10n', 'pts']
        # order the df by date
        air_quality_data = air_quality_data.sort_values(by='time')
        for column in columns_to_replace_with_mean:
            air_quality_data = take_off_outliers_and_replace_with_mean(air_quality_data, column)
        # coloumns that outliers will be replaced with the last valid value
        columns_to_replace_with_last_value = ['lat', 'lon']
        for column in columns_to_replace_with_last_value:
            air_quality_data = threat_lat_and_lon_outliers(air_quality_data, column)
        # Insert data into the Django model
        for index, row in air_quality_data.iterrows():
            print(row)
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
            # use time and acelerometer values to set a velocity value


        self.stdout.write(self.style.SUCCESS("Data inserted into the database"))

        # Resource usage after data processing and insertion
        end_memory, end_cpu = monitor_usage()
        memory_used = end_memory - start_memory
        cpu_used = end_cpu - start_cpu
        self.stdout.write(f"Final memory used: {memory_used:.2f} MB")
        self.stdout.write(f"Final CPU used: {cpu_used:.2f} %")
        self.stdout.write(self.style.SUCCESS("Data processing completed"))


