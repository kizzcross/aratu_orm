# data_processor/views.py
from django.shortcuts import render
from sensor.models import AirQualityData
from .utils import calculate_velocity

def process_data(request):
    data = AirQualityData.objects.all().order_by('time')
    
    velocities = []
    previous_data = None
    
    for entry in data:
        #pegamos cada registros da nossa base de dados
        if previous_data:
            #definindo nossa variação total de tempo 
            delta_t = entry.timestamp - previous_data.timestamp
            velocity = calculate_velocity(
                entry.accel_x, 
                entry.accel_y, 
                entry.accel_z, 
                delta_t
            )
            
            velocities.append({
                'timestamp': entry.timestamp,
                'velocity': velocity,
            })
        previous_data = entry
    
    return render(request, 'data_processor/velocity.html', {'velocities': velocities})
