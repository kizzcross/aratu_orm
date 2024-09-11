from datetime import timedelta
import math

# neste ficara responsavel pelos calculos e retornos com dados processados

def calculate_velocity(accel_x, accel_y, accel_z, delta_t):
    """
    accel_x -> dados do acelerometro no eixo x 
    accel_y -> dados do acelerometro no eixo y 
    accel_z -> dados do acelerometro no eixo z 
    delta_t --> providos da coluna time da base de dados temos a variação total de tempo

    Calcula a velocidade atravez da integração dos dados do acelerometro e a a variação de tempo
    O calculo da magnitude tambem poderia ser feito usado a lib numpy com o metodo sqrt()
    
    Em questão de desempenho sempre sera melhor usar maneiras diretas de realizar estes calculos
    """
    # Calcular a magnitude da aceleração
    magnitude = (accel_x**2 + accel_y**2 + accel_z**2)**0.5
  

    # Multiplicar pela variação de tempo para obter a mudança de velocidade
    velocity = magnitude * delta_t.total_seconds()
    return velocity



def convert_to_meters(lat1, lon1, lat2, lon2):
    """
    lat1 -> latitudo do primeiro ponto
    lon1 -> longitude do primeiro ponto
    
    lat2 -> latitudo do segundo ponto
    lon2 -> longitude do segundo ponto

    Essa função me retorna em kms a distancia entre estes dois pontos. 
    


    
    """
    R = 6371  # Raio da Terra em km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c  # em km # precisamos tratar os dados para que seja feita a medida em metros para aratu
    return distance

if "__name__" == "__main__":
    print("Ola mundo")
 # Testando a função de calcular distancia convert_to_meters()

    
    lat1, lon1 = -23.550520, -46.633308  # São Paulo
    lat2, lon2 = -22.906847, -43.172896  # Rio de Janeiro

    # Calcular a distância -> retornada em km
    distance = convert_to_meters(lat1, lon1, lat2, lon2)


    time = 5 # Este tempo no caso é dado em horas
    velocity = distance / time
    # para converter a velocidae de km/hora para metros por segundo(importante pois no nosso acelerometro a velocidade é retornada no padrão internacional m/s)
    velocity_meter_per_second = velocity/(3.6)

    print(f"Distância: {distance:.2f} km")
    print(f"Velocidade: {velocity:.2f} km/h")
    # tambem seria interessante ver essa velocidade em metros por segundo de acordo com o sistema internacional
