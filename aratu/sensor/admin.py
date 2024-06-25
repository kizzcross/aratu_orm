from django.contrib import admin
from .models import AirQualityMeter, MicroController, Sensor, AirQualityData

# Customize the AirQualityMeter admin
@admin.register(AirQualityMeter)
class AirQualityMeterAdmin(admin.ModelAdmin):
    list_display = ('name', 'model', 'serial_number', 'acquisition_date', 'maintenance_date', 'exchange_date')
    search_fields = ('name', 'model', 'serial_number')
    list_filter = ('acquisition_date', 'maintenance_date', 'exchange_date')
    date_hierarchy = 'acquisition_date'
    ordering = ['-acquisition_date']

# Customize the MicroController admin
@admin.register(MicroController)
class MicroControllerAdmin(admin.ModelAdmin):
    list_display = ('aratu_id', 'name', 'model', 'serial_number', 'acquisition_date', 'maintenance_date', 'exchange_date', 'air_quality_meter')
    search_fields = ('aratu_id', 'name', 'model', 'serial_number', 'air_quality_meter__name')
    list_filter = ('acquisition_date', 'maintenance_date', 'exchange_date', 'air_quality_meter')
    date_hierarchy = 'acquisition_date'
    ordering = ['-acquisition_date']

# Customize the Sensor admin
@admin.register(Sensor)
class SensorAdmin(admin.ModelAdmin):
    list_display = ('name', 'model', 'serial_number', 'acquisition_date', 'maintenance_date', 'exchange_date', 'microcontroller')
    search_fields = ('name', 'model', 'serial_number', 'microcontroller__name')
    list_filter = ('acquisition_date', 'maintenance_date', 'exchange_date', 'microcontroller')
    date_hierarchy = 'acquisition_date'
    ordering = ['-acquisition_date']

# Customize the AirQualityData admin
@admin.register(AirQualityData)
class AirQualityDataAdmin(admin.ModelAdmin):
    list_display = ('air_quality_meter', 'device', 'measure_time', 'temperature', 'humidity', 'pm1m', 'pm25m', 'pm10m', 'pm1n', 'pm25n', 'pm10n', 'address')
    search_fields = ('air_quality_meter__name', 'device', 'address')
    list_filter = ('measure_time', 'air_quality_meter', 'device')
    date_hierarchy = 'measure_time'
    ordering = ['-measure_time']


    # Optional: Fieldsets to organize fields in groups
    fieldsets = (
        ('General Info', {
            'fields': ('air_quality_meter', 'device', 'measure_time', 'address')
        }),
        ('Environmental Measurements', {
            'fields': ('temperature', 'humidity', 'lat', 'lon')
        }),
        ('Particulate Matter', {
            'fields': ('pm1m', 'pm25m', 'pm4m', 'pm10m', 'pm1n', 'pm25n', 'pm10n', 'pm4n', 'pts')
        }),
        ('Velocity', {
            'fields': ('vel',)
        }),
        ('Acceleration and Gyro', {
            'fields': ('ax', 'ay', 'az', 'gx', 'gy', 'gz')
        }),
    )

