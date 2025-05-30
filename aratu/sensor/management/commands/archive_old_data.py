from django.core.management.base import BaseCommand
from django.utils.timezone import now, timedelta
import pandas as pd
from sensor.models import AirQualityData, BackupConfig
import os

class Command(BaseCommand):
    help = 'Archive and delete old records to Parquet'

    def handle(self, *args, **options):
        config = BackupConfig.objects.filter(enabled=True).first()
        if not config:
            self.stdout.write("No active backup config found.")
            return

        cutoff = now() - timedelta(days=config.data_retention_days)
        qs = AirQualityData.objects.filter(measure_time__lt=cutoff)
        count = qs.count()

        if count == 0:
            self.stdout.write("No data to archive.")
            return

        df = pd.DataFrame(list(qs.values()))
        filename = f"archive/AirQualityData_backup_{now().strftime('%Y%m%d_%H%M%S')}.parquet"
        os.makedirs("archive", exist_ok=True)
        df.to_parquet(filename, index=False)
        self.stdout.write(f"Archived {count} records to {filename}")

        qs.delete()
        self.stdout.write("Deleted old records from DB.")

        config.last_backup = now()
        config.save()