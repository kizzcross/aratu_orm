# aratu_orm
# Projeto: Cálculo de Velocidade Instantânea com Base em Coordenadas Geográficas

## Descrição

Esta função foi desenvolvido com o objetivo de calcular a velocidade instantânea de um corpo utilizando dados de latitude, longitude e tempo de uma base de dados . A velocidade é calculada com base nas diferenças de posição e tempo entre medições consecutivas, Foi usado a fórmula de Haversine para medir a distância entre dois pontos geográficos. Também foi necessário realizar um pré-processamento dos dados temporais, convertendo formatos inconsistentes para um padrão datetime. 

## Funcionalidades Implementadas

### 1. Conversão de Coordenadas em Distância
A função `convert_to_meters(lat1, lon1, lat2, lon2)` foi criada para calcular a distância entre dois pontos geográficos utilizando a **fórmula de Haversine**. Essa fórmula retorna a distância entre os dois pontos em quilômetros, que pode ser usada posteriormente para calcular a velocidade. A integração dos dados do acelerometro é dada em metros por segundo , sera necessario converter posteriormente estes dados de km para metros para comparação.

#### Parâmetros da função:
- `lat1`, `lon1`: Latitude e longitude do primeiro ponto.
- `lat2`, `lon2`: Latitude e longitude do segundo ponto.
o conjunto de lat e log me da uma posição geografica do corpo que vamos medir a velocidade

#### Retorno:
- A distância em quilômetros entre os dois pontos. 

### 2. Tratamento de Dados Temporais
Foi identificado que a coluna `time` da base de dados apresentava problemas de padronização, com alguns valores já no formato `datetime` e outros em outro padrão . Para resolver isso, foi implementado um processo que tenta converter todos os valores, identificando e contando quantos dados estão fora do padrão datime.

#### Detalhes do processamento:
 
- A função diff do pandas é usada para somar dados consecutivos e foi utilizada tanto para a coluna time quanto para a coluna 
- Os dados fora do padrão foram contabilizados e removidos.
- Foi gerada uma nova lista com os dados no formato correto de `datetime`.
- Uma nova coluna `time_diff` foi criada para armazenar a diferença entre os tempos consecutivos, calculada utilizando a função `diff()` do pandas.
- A diferença de tempo foi convertida para segundos utilizando `total_seconds()`. e formatada para inteiro utilizando `int()`

### 3. Agrupamento por Data
Os dados foram agrupados por data para calcular a soma dos intervalos de tempo em segundos. Isso permite que se tenha uma visualização mais clara sobre qual dia se trata a variação de te.

### 4. Cálculo da Velocidade
- O Desenvolvimento da velocidade foi baseado em 2 principais conceitos , que seria a a velocidade media entre dois pontos consecutivos e integração dos dados do acelerometro
### 4.1 Integração dos dados dos acelerometros





## Pré-requisitos

- **Python 3.12.6**
- Bibliotecas utilizadas:
  - `pandas`
  - `numpy`
  - `math`
  - `datetime`


