import os
import json
from celery import shared_task
from datetime import datetime
from collections import defaultdict

# Define the source and destination folders
SOURCE_FOLDER = ''
DEST_FOLDER = ''


@shared_task
def aggregate_json_files():
    aggregated_data = defaultdict(list)

    # Read each JSON file from the source folder
    for filename in os.listdir(SOURCE_FOLDER):
        if filename.endswith(".json"):
            filepath = os.path.join(SOURCE_FOLDER, filename)
            with open(filepath, 'r') as json_file:
                data = json.load(json_file)
                device_id = data.get("device_id")

                if device_id:
                    # Append data to the corresponding device ID entry
                    aggregated_data[device_id].append(data)

    # Store each device's aggregated data in a new file in the destination folder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for device_id, data in aggregated_data.items():
        output_filename = os.path.join(DEST_FOLDER, f"aggregated_{device_id}_{timestamp}.json")

        with open(output_filename, 'w') as output_file:
            json.dump(data, output_file)

        print(f"Aggregated JSON for device {device_id} saved as {output_filename}")
