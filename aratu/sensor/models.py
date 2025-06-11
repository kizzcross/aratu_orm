from django.db import models
from auditlog.registry import auditlog


class AirQualityMeter(models.Model):
    name = models.CharField(max_length=100)
    model = models.CharField(max_length=100)
    serial_number = models.CharField(max_length=100)

    def __str__(self):
        return f'{self.name} - {self.model}'


class MicroController(models.Model):
    aratu_id = models.CharField(max_length=100)
    name = models.CharField(max_length=100)
    serial_number = models.CharField(max_length=100)
    model = models.CharField(max_length=100)
    air_quality_meter = models.ForeignKey(AirQualityMeter, on_delete=models.CASCADE)

    def __str__(self):
        return f'{self.aratu_id} - {self.name}'


class Sensor(models.Model):
    microcontroller = models.ForeignKey(MicroController, on_delete=models.CASCADE)

    def __str__(self):
        return f'{self.name} - {self.model}'


class AirQualityData(models.Model):
    air_quality_meter = models.ForeignKey(AirQualityMeter, on_delete=models.CASCADE, related_name='air_quality_data',
                                          null=True, blank=True)
    device = models.CharField(max_length=100)
    measure_time = models.DateTimeField()
    temperature = models.FloatField()
    humidity = models.FloatField()
    lat = models.FloatField()
    lon = models.FloatField()
    ax = models.DecimalField(max_digits=4, decimal_places=2)
    ay = models.DecimalField(max_digits=4, decimal_places=2)
    az = models.DecimalField(max_digits=4, decimal_places=2)
    gx = models.DecimalField(max_digits=4, decimal_places=2)
    gy = models.DecimalField(max_digits=4, decimal_places=2)
    gz = models.DecimalField(max_digits=4, decimal_places=2)
    pm1m = models.FloatField()
    pm25m = models.FloatField()
    pm4m = models.FloatField()
    pm10m = models.FloatField()
    pm1n = models.FloatField()
    pm25n = models.FloatField()
    pm10n = models.FloatField()
    pm4n = models.FloatField()
    pts = models.FloatField()
    vel = models.SmallIntegerField(default=0)
    address = models.CharField(max_length=150)

    def __str__(self):
        if self.air_quality_meter:
            return f'{self.air_quality_meter.name} - {self.measure_time}'
        else:
            return f'Unknown - {self.measure_time}'

    class Meta:
        ordering = ['-measure_time']
        verbose_name = 'Air Quality Data'
        verbose_name_plural = 'Air Quality Data'


class PredictedData(models.Model):
    """
    Modelo que vai quardar os dados da ultima predicao
    """
    cluster = models.IntegerField()
    data = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    end_date = models.DateTimeField()
    how_many_days = models.IntegerField()


class BackupConfig(models.Model):
    backup_interval_days = models.PositiveIntegerField(default=30)
    data_retention_days = models.PositiveIntegerField(default=90)
    last_backup = models.DateTimeField(null=True, blank=True)
    max_db_size_mb = models.PositiveIntegerField(null=True, blank=True)  # optional trigger
    enabled = models.BooleanField(default=True)


# all models must be auditables
auditlog.register(AirQualityMeter)
auditlog.register(MicroController)
auditlog.register(Sensor)
