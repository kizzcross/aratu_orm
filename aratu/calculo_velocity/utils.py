from datetime import timedelta

# neste ficara responsavel pelos calculos e retornos com dados processados

def calculate_velocity(accel_x, accel_y, accel_z, delta_t):
    """
    accel_x -> dados do acelerometro no eixo x 
    accel_y -> dados do acelerometro no eixo y 
    accel_z -> dados do acelerometro no eixo z 
    delta_t --> providos da coluna time da base de dados temos a variação total de tempo

    Calcula a velocidade atravez da integração dos dados do acelerometro e a a variação de tempo
    """
    # Calcular a magnitude da aceleração
    magnitude = (accel_x**2 + accel_y**2 + accel_z**2)**0.5
    # Podemos tambem usar o numpy para calcular a raiz quadrada da da soma dos quadrados desta função , foi feito a escolha de elevar a meio que é a mesma coisa que tirar a raiz quadrada pela eficiencia do código em vez de usar metodos de libs externas

    # Multiplicar pela variação de tempo para obter a mudança de velocidade
    velocity = magnitude * delta_t.total_seconds()
    return velocity