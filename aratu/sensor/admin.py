from django.contrib import admin
from .models import AirQualityMeter, MicroController, Sensor, AirQualityData, BackupConfig

# Customize the AirQualityMeter admin
@admin.register(AirQualityMeter)
class AirQualityMeterAdmin(admin.ModelAdmin):
    list_display = ('name', 'model', 'serial_number')
    search_fields = ('name', 'model', 'serial_number')
    ordering = ['name']

# Customize the MicroController admin
@admin.register(MicroController)
class MicroControllerAdmin(admin.ModelAdmin):
    list_display = ('aratu_id', 'name', 'model', 'serial_number', 'air_quality_meter')
    search_fields = ('aratu_id', 'name', 'model', 'serial_number', 'air_quality_meter__name')
    list_filter = ('air_quality_meter',)
    ordering = ['aratu_id']

# Customize the Sensor admin
@admin.register(Sensor)
class SensorAdmin(admin.ModelAdmin):
    list_display = ('microcontroller',)
    search_fields = ('microcontroller__name',)
    ordering = ['microcontroller']

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

@admin.register(BackupConfig)
class BackupConfigAdmin(admin.ModelAdmin):
    list_display = ('backup_interval_days', 'data_retention_days', 'last_backup', 'enabled')