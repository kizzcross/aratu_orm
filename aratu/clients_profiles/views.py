from django.shortcuts import render
from django.http import JsonResponse
from django.db.models import Min, Max
import pandas as pd
import folium
from folium.plugins import HeatMap
from sensor.models import AirQualityData  # Certifique-se de que o modelo está correto
from django.contrib.auth.decorators import permission_required
from django.core.exceptions import PermissionDenied

#------------------------------------------------------------------------
#inclusao do cluste geografico
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import networkx as nx
#import yfinance as yf

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

import math
import plotly.express as px
import itertools as it
from numba import jit

from scipy import stats
import seaborn as sns

import gmaps
from IPython.display import display

from datetime import datetime
import logging
import json
import plotly.io as pio
from plotly.io import to_html

db_heatmap = pd.DataFrame()
#-------------------------------------------------------------------------

# Lista de tipos de PM válidos
VALID_PM_TYPES = ['pm1m', 'pm25m', 'pm4m', 'pm10m']

def home(request):
    return render(request, 'clients_profiles/home.html')

def previsao(request):
    return render(request, 'clients_profiles/previsao.html')

def mapadecalor(request):
    return render(request, 'clients_profiles/mapadecalor.html')

def relatorio(request):
    return render(request, 'clients_profiles/relatorio.html')

def data(request):
    return render(request, 'clients_profiles/data.html')

# Endpoint para obter limites de datas, cluisterizar e criar modelo
def get_date_limits(request):
    try:
        date_limits = AirQualityData.objects.aggregate(
            min_date=Min('measure_time'),
            max_date=Max('measure_time')
        )

        min_date = date_limits.get('min_date')
        max_date = date_limits.get('max_date')

        return JsonResponse({'start_date': str(min_date), 'end_date': str(max_date)})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def create_cluster(request):
    global db_heatmap
    if request.method == 'POST':
        try:
            print("Recebendo requisição POST")  # Debug inicial
            # Extraindo as datas do corpo da requisição
            body = json.loads(request.body)
            print(f"Body recebido: {body}")  # Debug do body recebido
            start_date = body.get('start_date')
            end_date = body.get('end_date')
            print(f"Datas extraídas: start_date={start_date}, end_date={end_date}")  # Debug das datas

            # Validando as datas recebidas
            if not start_date or not end_date:
                print("Erro: Datas de início ou fim ausentes")
                return JsonResponse({'error': 'Datas de início e fim são obrigatórias'}, status=400)

            # Convertendo as datas para objetos datetime
            try:
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
                print(f"Datas convertidas: start_date={start_date}, end_date={end_date}")  # Debug das datas convertidas
            except ValueError:
                print("Erro: Formato de data inválido")
                return JsonResponse({'error': 'Formato de data inválido'}, status=400)

            # Filtrar os dados do banco com base nas datas
            print("Filtrando dados do banco...")
            data = AirQualityData.objects.filter(measure_time__range=[start_date, end_date]).values()

            print(f"Dados filtrados: {list(data)[:5]}")  # Mostrando os primeiros 5 registros para debug
            
            db = pd.DataFrame.from_records(data)
            print(f"DataFrame criado com {len(db)} registros")  # Debug do DataFrame criado
            
            db.rename(columns={
                'measure_time': 'date',
                'temperature': 'temp',
                'humidity': 'umi',
                'pm1n': 'pm1',
                'pm25n': 'pm25',
                'pm10n': 'pm10'
            }, inplace=True)

            db_heatmap = db[['date','temp', 'umi', 'lat', 'lon', 'pm1', 'pm25', 'pm10', 'pts']]
            print(f"DataFrame para heatmap criado com {len(db_heatmap)} registros")  # Debug do db_heatmap

            # Verificando se o DataFrame está vazio
            if db_heatmap.empty:
                print("DataFrame db_heatmap está vazio")
                logging.warning("DataFrame db_heatmap está vazio")
                return JsonResponse({'error': 'Nenhum dado encontrado para o intervalo fornecido'}, status=404)

            # Gerando HTML do cabeçalho e rodapé
            head_html = db_heatmap.head().to_html(index=False)
            tail_html = db_heatmap.tail().to_html(index=False)
            print("HTML gerado com sucesso")  # Debug da geração do HTML

            return JsonResponse({
                'message': 'Cluster geográfico criado com sucesso!',
                'head': head_html,
                'tail': tail_html
            })
        except Exception as e:
            print(f"Erro inesperado: {str(e)}")  # Debug de erros
            return JsonResponse({'error': f'Ocorreu um erro: {str(e)}'}, status=500)
    print("Método não permitido")  # Debug do método inválido
    return JsonResponse({'error': 'Método não permitido'}, status=405)
"""
def get_plot(request):
    global db_heatmap
    if db_heatmap is not None:
        xk_heatmap = db_heatmap[['lat', 'lon']].values
        y = db_heatmap['cluster'].values
        latlon = db_heatmap[['lat', 'lon']].values

        # Geração do gráfico
        colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
        df_plot = {
            'latitude': latlon[:, 1],
            'longitude': latlon[:, 0],
            'cluster': y
        }
        fig = px.scatter(
            df_plot,
            x='latitude',
            y='longitude',
            color='cluster',
            color_discrete_sequence=colors,
            title='Clusterização',
            width=800,
            height=600
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black'),
            xaxis=dict(gridcolor='gray'),
            yaxis=dict(gridcolor='gray')
        )

        # Serializa o gráfico para JSON e retorna ao frontend
        graph_json = pio.to_json(fig)
        return JsonResponse({'plot': graph_json})

    return JsonResponse({'error': 'Nenhum dado disponível para plotar.'}, status=400)
"""

# Endpoint para definir regiões
def define_regions(request):
    global db_heatmap
    if request.method == 'POST':
        # Lógica para definir regiões
        xk_heatmap = db_heatmap[['lat', 'lon']].astype(np.float64).values
        latlon = db_heatmap[['lat', 'lon']].astype(np.float64).values
        r0 = 5 #parametro da clusterização - limite de variancia
        y = clusters_maia(xk_heatmap, r0)
        db_heatmap["cluster"] = y
        fig = clusters_plot(xk_heatmap, y, latlon)
        #graph_json = pio.to_json(fig)
        #return JsonResponse({'plot': fig})
        return JsonResponse({'message': 'Regiões definidas com sucesso!'})
    return JsonResponse({'error': 'Método não permitido'}, status=405)

# Endpoint para treinar modelo
def train_model(request):
    if request.method == 'POST':
        # Lógica para treinar o modelo

        return JsonResponse({'message': 'Modelo treinado com sucesso!'})
    return JsonResponse({'error': 'Método não permitido'}, status=405)

@permission_required('clients_profiles.view_airqualitydata', raise_exception=True)
def generate_heatmap(request):
    try:
        pm_type = request.GET.get('pm_type', 'pm25m')  # Padrão: 'pm25m'

        # Verificar se o tipo de PM é válido
        if pm_type not in VALID_PM_TYPES:
            return JsonResponse({'error': f'Invalid PM type. Valid types are: {", ".join(VALID_PM_TYPES)}'})

        # Buscar os dados do banco de dados
        data = AirQualityData.objects.values('lat', 'lon', pm_type)

        # Converter os dados para um DataFrame
        df = pd.DataFrame.from_records(data)

        # Remover valores NaN em PM, latitude e longitude
        df = df.dropna(subset=[pm_type, 'lat', 'lon'])

        # Verificar se há dados válidos
        if df.empty:
            return JsonResponse({'error': 'No valid data to display. All selected PM values are NaN.'})

        # Selecionar os dados de localização e valores
        locations = df[['lat', 'lon']].values.tolist()
        pm_values = df[pm_type].values.tolist()

        # Configurar o mapa
        center = [-20.27931919525688, -40.287294463682024]  # Ajuste para a localização desejada
        m = folium.Map(
            location=center,
            zoom_start=12,
            width='100%',
            height='70%'
        )

        # Adicionar camada de calor
        heat_data = [[loc[0], loc[1], pm] for loc, pm in zip(locations, pm_values)]
        HeatMap(heat_data, max_intensity=100, radius=8).add_to(m)

        # Renderizar o mapa como HTML
        map_html = m._repr_html_()

    except PermissionDenied:
        # Se o usuário não tiver permissão, retornamos um erro personalizado em JSON
        return JsonResponse({'error': 'Você não tem permissão para gerar o heatmap!'}, status=403)

    # Retornar o HTML do mapa como parte da resposta JSON
    return JsonResponse({'map_html': map_html})


#---------------------------------------------------------------------------------------------------------------------------------------------

class EvolvingClustering:
    def __init__(self, macro_cluster_update=1,
                 verbose=0, variance_limit=0.001, decay = 100, debug=False, plot_graph=False):
        self.verbose = verbose
        self.total_num_samples = 0
        self.micro_clusters = []
        self.macro_clusters = []
        self.active_macro_clusters = []
        self.graph = nx.Graph()
        self.active_graph = nx.Graph()
        self.variance_limit = variance_limit
        self.macro_cluster_update = macro_cluster_update
        self.debug = debug
        self.plot_graph = plot_graph
        self.changed_micro_clusters = []
        self.fading_factor = 1 / decay

    @staticmethod
    def get_micro_cluster(id, num_samples, mean, variance, density, life):
        return {"id": id,"num_samples": num_samples, "mean": mean, "variance": variance,
                "density": density, "active": True, "changed": True, "life": life}

    def create_new_micro_cluster(self, x):

        if len(self.micro_clusters) == 0:
            id = 1
        else:
            id = max([m["id"] for m in self.micro_clusters]) + 1

        num_samples = 1
        mean = x
        variance = 0
        density = 0
        life = 1
        new_mc = EvolvingClustering.get_micro_cluster(id, num_samples, mean, variance, density, life)
        self.micro_clusters.append(new_mc)
        self.changed_micro_clusters.append(new_mc)
        self.graph.add_node(id)

    def is_outlier(self, s_ik, var_ik, norm_ecc):

        if s_ik < 3:
            outlier = (var_ik > self.variance_limit)
        else:
            mik_sik = 3 / (1 + math.exp(-0.007 * (s_ik - 100)))
            outlier_limit = ((mik_sik ** 2) + 1) / (2 * s_ik)
            outlier = (norm_ecc > outlier_limit)

        return outlier

    def update_micro_cluster(self, xk, micro_cluster, num_samples, mean, variance, norm_ecc):

        self.update_life(xk, micro_cluster)

        micro_cluster["num_samples"] = num_samples
        micro_cluster["mean"] = mean
        micro_cluster["variance"] = variance
        micro_cluster["density"] = 1 / norm_ecc
        micro_cluster["changed"] = True

        if micro_cluster not in self.changed_micro_clusters:
            self.changed_micro_clusters.append(micro_cluster)

    def update_life(self, xk, micro_cluster):
        previous_mean = micro_cluster["mean"]
        previous_var = micro_cluster["variance"]
        if previous_var > 0:
            d = EvolvingClustering.get_euclidean_distance(xk, previous_mean)
            dist = np.sqrt(np.sum(d))
            rt = np.sqrt(previous_var)
            micro_cluster["life"] = micro_cluster["life"] + (((rt - dist) / rt) * self.fading_factor)
        else:
            micro_cluster["life"] = 1

    @staticmethod
    def get_updated_micro_cluster_values(x, s_ik, mu_ik, var_ik):

        s_ik += 1
        mean = ((s_ik - 1) / s_ik) * mu_ik + (x / s_ik)

        # Codigo dissertacao
        delta = x - mean
        variance = EvolvingClustering.update_variance(delta, s_ik, var_ik)

        norm_ecc = EvolvingClustering.get_normalized_eccentricity(x, s_ik, mean, variance)
        return (s_ik, mean, variance, norm_ecc)

    @staticmethod
    @jit(nopython=True)
    def update_variance(delta, s_ik, var_ik):
        variance = ((s_ik - 1) / s_ik) * var_ik + (((np.linalg.norm(delta) * 2 / len(delta))** 2 / (s_ik - 1)))
        return variance

    @staticmethod
    def get_normalized_eccentricity(x, num_samples, mean, var):
        ecc = EvolvingClustering.get_eccentricity(x, num_samples, mean, var)
        return ecc / 2

    @staticmethod
    @jit(nopython=True)
    def get_eccentricity(x, num_samples, mean, var):
        if var == 0 and num_samples > 1:
            result = (1/num_samples)
        else:
            a = mean - x
            result = ((1 / num_samples) + ((np.linalg.norm(a) * 2 / len(a)) ** 2 / (num_samples * var)))

        return result

    def fit(self, X, update_macro_clusters=True, prune_micro_clusters=True):

        lenx = len(X)

        if self.debug:
            print("Training...")

        for xk in X:
            self.update_micro_clusters(xk)

            if prune_micro_clusters:
                self.prune_micro_clusters()

            self.total_num_samples += 1

            if self.debug:
                print('Training %d of %d' %(self.total_num_samples, lenx))

        if update_macro_clusters:
            if self.debug:
                print('Updating Macro_clusters')
            self.update_macro_clusters()

        if self.plot_graph:
            self.plot_micro_clusters(X)


    def update_micro_clusters(self, xk):
        # First sample
        if self.total_num_samples == 0:
            self.create_new_micro_cluster(xk)
        else:
            new_micro_cluster = True

            for mi in self.micro_clusters:
                mi["changed"] = False
                s_ik = mi["num_samples"]
                mu_ik = mi["mean"]
                var_ik = mi["variance"]

                (num_samples, mean, variance, norm_ecc) = EvolvingClustering.get_updated_micro_cluster_values(xk, s_ik, mu_ik, var_ik)

                if not self.is_outlier(num_samples, variance, norm_ecc):
                    self.update_micro_cluster(xk, mi, num_samples, mean, variance, norm_ecc)
                    new_micro_cluster = False

            if new_micro_cluster:
                self.create_new_micro_cluster(xk)

    def update_macro_clusters(self):
        self.define_macro_clusters()
        self.define_activations()

    def predict(self, X):
        self.labels_ = np.zeros(len(X), dtype=int)
        index = 0
        lenx = len(X)

        if self.debug:
            print('Predicting...')

        for xk in X:
            memberships = []
            for mg in self.active_macro_clusters:
                active_micro_clusters = self.get_active_micro_clusters(mg)

                memberships.append(EvolvingClustering.calculate_membership(xk, active_micro_clusters))

            self.labels_[index] = np.argmax(memberships)
            index += 1

            if self.debug:
                print('Predicting %d of %d' % (index, lenx))

        return self.labels_

    @staticmethod
    def calculate_membership(x, active_micro_clusters):
        total_density = 0
        for m in active_micro_clusters:
            total_density += m["density"]

        mb = 0
        for m in active_micro_clusters:
            d = m["density"]

            t = 1 - EvolvingClustering.get_normalized_eccentricity(x, m["num_samples"], m["mean"], m["variance"])
            mb += (d / total_density) * t
        return mb


    @staticmethod
    def calculate_micro_membership(x, params):

        micro_cluster = params[0]
        total_density = params[1]

        d = micro_cluster["density"]

        t = 1 - EvolvingClustering.get_normalized_eccentricity(x, micro_cluster["num_samples"], micro_cluster["mean"], micro_cluster["variance"])
        return (d / total_density) * t

    def get_total_density(self):
        active_mcs = self.get_all_active_micro_clusters()
        total_density = 0

        for m in active_mcs:
            total_density += m["density"]

        return total_density

    def get_active_micro_clusters(self, mg):
        active_micro_clusters = []
        for mi_ind in mg:
            mi = next(item for item in self.micro_clusters if item["id"] == mi_ind)
            if mi["active"]:
                active_micro_clusters.append(mi)
        return active_micro_clusters

    def get_all_active_micro_clusters(self):
        active_micro_clusters = []

        for m in self.micro_clusters:
            if m["active"]:
                active_micro_clusters.append(m)
        return active_micro_clusters

    def get_changed_micro_clusters(self):
        changed_micro_clusters = []

        for m in self.micro_clusters:
            if m["changed"]:
                changed_micro_clusters.append(m)
        return changed_micro_clusters

    def get_changed_active_micro_clusters(self):
        changed_micro_clusters = []

        for m in self.changed_micro_clusters:
            if m["active"]:
                changed_micro_clusters.append(m)
        return changed_micro_clusters


    def define_macro_clusters(self):

        for mi in self.changed_micro_clusters:
            for mj in self.micro_clusters:
                if mi["id"] != mj["id"]:
                    edge = (mi["id"],mj["id"])
                    if EvolvingClustering.has_intersection(mi, mj):
                        self.graph.add_edge(*edge)
                    elif EvolvingClustering.nodes_connected(mi["id"],mj["id"], self.graph):
                        self.graph.remove_edge(*edge)

        self.macro_clusters = list(nx.connected_components(self.graph))
        self.changed_micro_clusters.clear()



    @staticmethod
    def nodes_connected(u, v, G):
        return u in G.neighbors(v)

    def define_activations(self):

        self.active_graph = self.graph.copy()

        for mg in self.macro_clusters:
            num_micro = len(mg)
            total_density = 0

            for i in mg:
                dens = next(item["density"] for item in self.micro_clusters if item["id"] == i)
                total_density += dens

            mean_density = total_density / num_micro

            for i in mg:
                mi = next(item for item in self.micro_clusters if item["id"] == i)
                mi["active"] = (mi["num_samples"] > 2) and (mi["density"] >= mean_density)

                if not mi["active"]:
                    self.active_graph.remove_node(mi["id"])

        self.active_macro_clusters = list(nx.connected_components(self.active_graph))


    @staticmethod
    def has_intersection(mi, mj):
        mu_i = mi["mean"]
        mu_j = mj["mean"]
        var_i = mi["variance"]
        var_j = mj["variance"]

        d = EvolvingClustering.get_euclidean_distance(mu_i, mu_j)
        dist = np.sqrt(np.sum(d))

        deviation = EvolvingClustering.get_deviation(var_i, var_j)

        return dist <= deviation

    @staticmethod
    @jit(nopython=True)
    def get_deviation(var_i, var_j):
        deviation = 2 * (np.sqrt(var_i) + np.sqrt(var_j))
        return deviation

    @staticmethod
    @jit(nopython=True)
    def get_euclidean_distance(mu_i, mu_j):
        dist = [(a - b) ** 2 for a, b in zip(mu_i, mu_j)]
        return dist

    def plot_micro_clusters(self, X):

        micro_clusters = self.get_all_active_micro_clusters()
        ax = plt.gca()

        ax.scatter(X[:, 0], X[:, 1], s=1, color='b')

        for m in micro_clusters:
            mean = m["mean"]
            std = math.sqrt(m["variance"])

            circle = plt.Circle(mean, std, color='r', fill=False)

            ax.add_artist(circle)
        plt.draw()

    def prune_micro_clusters(self):
        for mc in self.micro_clusters:

            if not mc["active"]:
                mc["life"] = mc["life"] - self.fading_factor

                if mc["life"] < 0:
                    self.micro_clusters.remove(mc)

                    if mc in self.changed_micro_clusters:
                        self.changed_micro_clusters.remove(mc)

                    self.graph.remove_node(mc["id"])


def clusters_maia(x, r0):

    print(f'r0 = {r0}')

    evol = EvolvingClustering(macro_cluster_update=1, variance_limit=r0, debug=False)
    evol.fit(x)

    y = evol.predict(x)

    return y
#-----------------------------------------------------------------------------------------------------
def clusters_plot(x, y, ll):
    global db_heatmap
    if db_heatmap is not None:
        latlon = db_heatmap[['lat', 'lon']].values
        y = db_heatmap['cluster'].values

        # Configuração do gráfico
        colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
        df_plot = {
            'latitude': latlon[:, 1],
            'longitude': latlon[:, 0],
            'cluster': y
        }
        fig = px.scatter(
            df_plot,
            x='latitude',
            y='longitude',
            color='cluster',
            color_discrete_sequence=colors,
            title='Clusterização',
            width=800,
            height=600
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black'),
            xaxis=dict(gridcolor='gray'),
            yaxis=dict(gridcolor='gray')
        )
        return fig.show()


def model_singh(X,y):

    # Entrada -> X (dados de treino) / Saidas -> y (saidas de treino)

    # Clusterização dos dados de saída - Dados Y

    # Encontra a raiz da arvore

    Rg = y.max() - y.min()              # Calcula a faixa de valor dos dados
    SD = y.std()                        # Calcula do desvio padrão
    W = Rg / (SD * (len(y)))            # Calcula a amplitude
    Lbound = y.min() - W                # Intervalos minimo
    Ubound = y.max() + W                # Intervalos maximo
    Xmid = ((Lbound + Ubound) / 2)*0.5  # Calculo do ponto medio
    Troot = Xmid                        # Atribui a raiz da arvore o valor do ponto medio

    tree = initialize_tree(Troot)       # Inicia o processo de criação da arvore, criando a matriz e inserindo o valor da raiz
    for value in y:                     # Loop for cada valor da lista de dados
        insert(tree, value)             # Insere cada valor na arvore

    # Clusterização dos dados #

    cluster = [0] * len(tree)           # Cria uma lista para cluster com o tamanho da arvore com valores 0
    cluster[0] = 1                      # O primeiro cluster recebe o valor 1

    for i in range(len(tree)): # loop para analisar todos os valores da arvore

        if (tree[i][1] != None) and (tree[i][2] != None):           # Analisa se o nó tem tanto o filho esquerdo quanto o filho direito
            dist_left = abs(tree[i][0] - tree[tree[i][1]][0])       # Calcula a distancia do filho esquerdo com o pai
            dist_right = abs(tree[i][0] - tree[tree[i][2]][0])      # Calcula a distancia do filho direita com o pai
            if dist_left <= dist_right:                             # Verifica se a distancia do filho esquerdo é menor do que da direita
                left_idx = tree[i][1]                               # Salva a posição do filho da esquerda
                right_idx = tree[i][2]                              # Salva a posição do filho da direita
                cluster[left_idx] = cluster[i]                      # Insere o valor do cluster analisado na posição do filho esquerdo
                if (tree[right_idx][1] != None) or (tree[right_idx][2] != None): # Verifica se o nó direito tem filhos
                    cluster[right_idx] = max(cluster) + 1           # Nesse caso, é criado um novo cluster na posição do filho direito (utilizando o maior valor da lista e somando 1)
                else:                                               # Caso não tenha nenhum filho
                    cluster[right_idx] = cluster[i]                 # O filho direito recebe o mesmo cluster do esquerdo
            else:                                                   # Caso a distancia do filho direito seja menor do que a da equerda
                left_idx = tree[i][1]                               # Salva a posição do filho da esquerda
                right_idx = tree[i][2]                              # Salva a posição do filho da direita
                cluster[right_idx] = cluster[i]                     # Insere o valor do cluster analisado na posição do filho direito
                if (tree[left_idx][1] != None) or tree[left_idx][2] != None: # Verifica se o nó direito tem filhos
                    cluster[left_idx] = max(cluster) + 1            # Nesse caso, é criado um novo cluster na posição do filho esquerdo (utilizando o maior valor da lista e somando 1)
                else:                                               # Caso não tenha nenhum filho
                    cluster[left_idx] = cluster[i]                  # O filho esquerdo recebe o mesmo cluster do direito
        elif (tree[i][1] != None) and (tree[i][2] == None):         # Analisa se o nó possui somente o filho esquerdo
            left_idx = tree[i][1]                                   # Salva a posição do filho da esquerda
            cluster[left_idx] = cluster[i]                          # Insere o valor do cluster analisado na posição do filho esquerdo
        elif (tree[i][1] == None) and (tree[i][2] != None):         # Analisa se o nó possui somente o filho direito
            right_idx = tree[i][2]                                  # Salva a posição do filho da direita
            cluster[right_idx] = cluster[i]                         # Insere o valor do cluster analisado na posição do filho direito

    for i in range(len(tree)):                                      # Apos a clusterizção dos dados o loop for é feito para rotular todos os valores da arvore
        tree[i][3] = cluster[i]                                     # Adiciona o cluster relacionado ao nó na ultima posição da linha


    # Criação de uma matriz dos valores de y com seu determinado cluster #

    Data = [[None, None] for _ in range(len(y))] # cria a matriz "Data" com duas colunas vazias e do mesmo tamanho de y

    for i in range(len(y)):                 # loop for para cada valor de y
        Data[i][0] = y[i]                   # Insere o valor de y_i na primeira posição da matriz na linha i
        for j in y:                         # loop for onde j recebe todos os valores de y 
            if j == tree[i+1][0]:           # se o valor de j for igual ao do nó correspondente na arvore (i+1, para pular a raiz da arvore)
                Data[i][1] = tree[i+1][3]   # Insere o numero do clusters na segunda posição da matriz na linha i

    # Criação da matriz "FLRy" informando o cluster e o cluster seguinte #

    FLRy = [[None, None] for _ in range(len(tree))] # cria a matriz "FLRy" com duas colunas vazias e do mesmo tamanho de tree

    for i in range(len(tree)):              # loop for para analisar cada valor de tree
        FLRy[i][0] = tree[i][3]             # Insere o valor do cluster na primeira posição da matriz na linha i
        if i + 1 < len(tree):               # Verifica se existe um proximo valor, para evitar erros no programa
            FLRy[i][1] = tree[i + 1][3]     # Insere o valor do proximo cluster na segunda posição da matriz na linha i
        if FLRy[i][1] == None:              # Verifica se existe um proximo cluster, vendo se o cluster adicionado na segunda posição é vazio ou não
            FLRy.pop()                      # Se for vazio, a linha é removida

    ###############################################################

    # Clusterização dos dados de entrada - Dados X

    # Analisa quantas variaveis de entrada existem
    if len(X.shape) == 1:                   # Olha o tamanho da varivel de entrada, se for 1 gera o modelo para apenas uma variavel de entrada
        
        # Encontra a raiz da arvore
            
        Rg = X.max() - X.min()              # Calcula a faixa de valor dos dados
        SD = X.std()                        # Calcula do desvio padrão
        W = Rg / (SD * (len(X)))            # Calcula a amplitude
        Lbound = X.min() - W                # Intervalos minimo
        Ubound = X.max() + W                # Intervalos maximo
        U = [Lbound, Ubound]                # Intervalos da amostra
        Xmid = ((Lbound + Ubound) / 2)*0.5  # Calculo do ponto medio
        Troot = Xmid                        # Inicia o processo de criação da arvore, criando a matriz e inserindo o valor da raiz
        
        tree = initialize_tree(Troot)       # Inicia o processo de criação da arvore, criando a matriz e inserindo o valor da raiz
        for value in X:                     # Loop for cada valor da lista de dados
            insert(tree, value)             # Insere cada valor na arvore

        # Clusterização dos dados #

        cluster = [0] * len(tree)           # Cria uma lista para cluster com o tamanho da arvore com valores 0
        cluster[0] = 1                      # O primeiro cluster recebe o valor 1

        for i in range(len(tree)): # loop para analisar todos os valores da arvore

            if (tree[i][1] != None) and (tree[i][2] != None):           # Analisa se o nó tem tanto o filho esquerdo quanto o filho direito
                dist_left = abs(tree[i][0] - tree[tree[i][1]][0])       # Calcula a distancia do filho esquerdo com o pai
                dist_right = abs(tree[i][0] - tree[tree[i][2]][0])      # Calcula a distancia do filho direita com o pai
                if dist_left <= dist_right:                             # Verifica se a distancia do filho esquerdo é menor do que da direita
                    left_idx = tree[i][1]                               # Salva a posição do filho da esquerda
                    right_idx = tree[i][2]                              # Salva a posição do filho da direita
                    cluster[left_idx] = cluster[i]                      # Insere o valor do cluster analisado na posição do filho esquerdo
                    if (tree[right_idx][1] != None) or (tree[right_idx][2] != None): # Verifica se o nó direito tem filhos
                        cluster[right_idx] = max(cluster) + 1           # Nesse caso, é criado um novo cluster na posição do filho direito (utilizando o maior valor da lista e somando 1)
                    else:                                               # Caso não tenha nenhum filho
                        cluster[right_idx] = cluster[i]                 # O filho direito recebe o mesmo cluster do esquerdo
                else:                                                   # Caso a distancia do filho direito seja menor do que a da equerda
                    left_idx = tree[i][1]                               # Salva a posição do filho da esquerda
                    right_idx = tree[i][2]                              # Salva a posição do filho da direita
                    cluster[right_idx] = cluster[i]                     # Insere o valor do cluster analisado na posição do filho direito
                    if (tree[left_idx][1] != None) or tree[left_idx][2] != None: # Verifica se o nó direito tem filhos
                        cluster[left_idx] = max(cluster) + 1            # Nesse caso, é criado um novo cluster na posição do filho esquerdo (utilizando o maior valor da lista e somando 1)
                    else:                                               # Caso não tenha nenhum filho
                        cluster[left_idx] = cluster[i]                  # O filho esquerdo recebe o mesmo cluster do direito
            elif (tree[i][1] != None) and (tree[i][2] == None):         # Analisa se o nó possui somente o filho esquerdo
                left_idx = tree[i][1]                                   # Salva a posição do filho da esquerda
                cluster[left_idx] = cluster[i]                          # Insere o valor do cluster analisado na posição do filho esquerdo
            elif (tree[i][1] == None) and (tree[i][2] != None):         # Analisa se o nó possui somente o filho direito
                right_idx = tree[i][2]                                  # Salva a posição do filho da direita
                cluster[right_idx] = cluster[i]                         # Insere o valor do cluster analisado na posição do filho direito

        for i in range(len(tree)):                                      # Apos a clusterizção dos dados o loop for é feito para rotular todos os valores da arvore
            tree[i][3] = cluster[i]                                     # Adiciona o cluster relacionado ao nó na ultima posição da linha

        # Criação de uma matriz dos valores de x com seu determinado cluster #

        Data = [[None, None] for _ in range(len(X))]    # cria a matriz "Data" com duas colunas vazias e do mesmo tamanho de x
        for i in range(len(X)):                         # loop for para cada valor de x
            Data[i][0] = X[i]                           # Insere o valor de x_i na primeira posição da matriz na linha i
            for j in X:                                 # loop for onde j recebe todos os valores de x
                if j == tree[i+1][0]:                   # se o valor de j for igual ao do nó correspondente na arvore (i+1, para pular a raiz da arvore)
                    Data[i][1] = tree[i+1][3]           # Insere o numero do clusters na segunda posição da matriz na linha i

        # FLR de X #

        # Criação da matriz "FLRx" informando o cluster e o cluster seguinte #

        FLRx = [[None, None] for _ in range(len(tree))] # cria a matriz "FLRx" com duas colunas vazias e do mesmo tamanho de tree
        
        for i in range(len(tree)):                      # loop for para analisar cada valor de tree
            FLRx[i][0] = tree[i][3]                     # Insere o valor do cluster na primeira posição da matriz na linha i
            if i + 1 < len(tree):                       # Verifica se existe um proximo valor, para evitar erros no programa
                FLRx[i][1] = tree[i + 1][3]             # Insere o valor do proximo cluster na segunda posição da matriz na linha i
            if FLRx[i][1] == None:                      # Verifica se existe um proximo cluster, vendo se o cluster adicionado na segunda posição é vazio ou não
                FLRx.pop()                              # Se for vazio, a linha é removida

        # FLR para x e y #

        FLRxy = []                                      # Cria uma lista vazia para armazenar os valores dos clusters de exntrada e o seguinte cluster da saída
        
        for i in range(0,len(FLRy)):                    # Passa por todos os valores de FLRx e FLRy 
            FLRxy.append([FLRx[i][0],FLRy[i][1]])       # Adiciona uma linha nova na lista FLRx, o primeiro é o cluster de FLRx e o segundo é o proximo cluster de FLRy
            
        # FLRG #

        ValueAssociation = {}                           # Cria um dicionario vazio para armazenar as associações temporais

        for pair in FLRxy:                              # Loop para analisar todos os elementos da matriz FLRxy
            key, value = pair                           # Separa os valores de cada linha, sendo o primeiro a key e o segundo em value
            if key not in ValueAssociation:             # Confere se valor já não está no dicionario ou se não foi adicionado ainda
                ValueAssociation[key] = set()           # Inclui a key no dicionario e evita linhas duplicadas
            if value is not None:                       # Se o valor não for vazio
                ValueAssociation[key].add(value)        # Insere o value na key analisada

        # Converte o dicionário em lista um lista "FLRG" #
        
        FLRG = [[key] + sorted(list(values)) for key, values in ValueAssociation.items()]   # Adiciona os itens na lista sendo o primeiro da linha a key e os seguintes o value 
        FLRG.sort(key=lambda x: x[0])                                                       # Ordena a lista por chaves em ordem crescente para garantir a ordem correta

        # Criação do dicionario FLRG_ para armazenar os clusters e os valores contidos em cada um deles #

        FLRG_ = {}                                      # Cria um dicionario vazio 

        for pair in Data:                               # Loop para analisar todos os elementos da matriz Data
            value, cluster = pair                       # Separa os valores de cada linha, sendo o primeiro a key e o segundo em value
            if cluster not in FLRG_:                    # Confere se o cluster já está no dicionario ou se não foi adicionado ainda
                FLRG_[cluster] = set()                  # Inclui o cluster no dicionario e evita linhas duplicadas
            if value is not None:                       # Se o valor não for vazio
                FLRG_[cluster].add(value)               # Insere o value no cluster analisado
        
        # Cria um dicionario (FLRG_sorted) onde ordena o dicinario FLRG com os clusters em ordem crescente
        FLRG_sorted = {key: FLRG_[key] for key in sorted(FLRG_)}
        
        # Pontos medio de cada cluster #

        midPoint = []                                   # Cria uma lista vazia para armazenar os pontos medios de cada cluster
        
        for cluster, values in FLRG_sorted.items():     # Loop for dos items do diconario divididos em clusters e uma lista de valores
            if cluster:                                 # Verifica se o cluster não está vazio
                associated_values = []                  # Cria uma lista auxiliar vazia para armazenar cada valor separadamente e cada loop reinicia a lista
                for val in values:                      # Loop for para separar os valores da lista values
                    associated_values.append(val)       # Adicona os valores separados na lista auxiliar
                if associated_values:                   # Verifica se a lista não está vazia 
                    average = (max(associated_values) + min(associated_values))/2 # Calcula a media dos valores do cluster, calculado pelo valor minimo + valor maximo dividido por 2
                    midPoint.append(average)            # Adiciona a media de cada clusters na lista dos pontos medios

        # Variaveis Fuzzy #

        A = []                                          # Cria uma lista vazia para armazenar o intervalo das variaveis fuzzy
        
        for i in FLRG_sorted:                                   # loop for para analisar cada valor do dicionario FLRG
            Ai = [min(FLRG_sorted[i]), max(FLRG_sorted[i]), i]  # Cria uma linha para cada intervalo, 1° - inicio do intervalo/2° - fim do intervalo/3° - cluster relacionado
            A.append(Ai)                                        # Adiciona as infromaçoes na lista A

        A.sort()                                        # Organiza os intervalos em ordem crescente

        # Preencher os espaços entre os intervalos para evitar erros em codigos #
        
        for i in range(0, len(A)-1):                    # Loop for para analisar os espaços entre o os intervalos, por conta disso não analisa o ultimo da lista

            A_diff = (A[i+1][0] - A[i][1])/2            # Calcula a diferença entre o fim do primeiro intervalo e incio do segundo, depois divide por 2
            A[i][1] = A[i][1] + A_diff                  # o final do primeiro intervalo é modificado somando a metade da distancia
            A[i+1][0] = A[i+1][0] - A_diff              # o inicio do segundo intervalo é modificado subtraindo a metade da distancia

        # Ajustes nos intervalos para evitar erros #
        
        A[0][0] = -100000                               # O inicio do primeiro intervalo é modificado para um valor muito baixo, para evitar algum valor fora dos intervalos
        A[-1][1] = A[-1][1]*10                          # O final do ultimo intervalo é modificado para um valor muito altp, para evitar algum valor fora dos intervalos

        # Criação do modelo #

        model = []                                      # Cria uma lista vazia para armazenar as informações do modelo

        for i in range(0, len(FLRG)):                   # Loop for passa por todos os FLRG
            fore_mid = 0                                # Defina o valor inicial para a media, reinicia a cada loop
            n = (len(FLRG[i])-1)                        # Valor que divide na media dos valores, len(FLRG[i])-1 -> quantos elementos estão relacionados com o cluster, retira o primeiro
            for j in range(1, len(FLRG[i])):            # Loop for para analisar cada valor relacionado com o cluster
                fore_mid += midPoint[(FLRG[i][j])-1]    # Somatorio de todos os pontos medios dos cluster relacionados no FLRG
            fore_mid = fore_mid/n                       # Faz a divisão pela quantidade de valores no somatorio
            model.append([fore_mid, i+1])               # Adicona esse valor a lista do modelo e adiciona o numero do cluster (i se inicia no 0, por isso i+1)

        # Junção das informações e saída da função #
        
        model_singh = [A, model]                        # Coloca ambas as nformações principais do modelo em uma unica variavel

        return model_singh, tree                        # Retorna a saida da função

    else:                               # No caso de possuir mais de uma variavel de entrada
        
        model = []                      # Cria uma lista vazia para armazenar o modelo de cada entrada x
        A = []                          # Cria uma lista vazia para armazenar os intervalos de fuzzy de cada entrada x
        
        for mi in range(0, len(X)):             # Loop for para criar o modelo de cada variavel de entrada x
            
            x = X[mi]                           # Salva na variavel o valor da entrada analisada

            Rg = x.max() - x.min()              # Calcula a faixa de valor dos dados
            SD = x.std()                        # Calcula do desvio padrão
            W = Rg / (SD * (len(x)))            # Calcula a amplitude
            Lbound = x.min() - W                # Intervalos minimo
            Ubound = x.max() + W                # Intervalos maximo
            U = [Lbound, Ubound]                # Intervalos da amostra
            Xmid = ((Lbound + Ubound) / 2)*0.5  # Calculo do ponto medio
            Troot = Xmid                        # Atribui a raiz da arvore o valor do ponto medio
            
            tree = initialize_tree(Troot)       # Inicia o processo de criação da arvore, criando a matriz e inserindo o valor da raiz
            for value in x:                     # Loop for cada valor da lista de dados
                insert(tree, value)             # Insere cada valor na arvore
            
            # Clusterização dos dados #

            cluster = [0] * len(tree)           # Cria uma lista para cluster com o tamanho da arvore com valores 0
            cluster[0] = 1                      # O primeiro cluster recebe o valor 1

            for i in range(len(tree)): # loop para analisar todos os valores da arvore

                if (tree[i][1] != None) and (tree[i][2] != None):           # Analisa se o nó tem tanto o filho esquerdo quanto o filho direito
                    dist_left = abs(tree[i][0] - tree[tree[i][1]][0])       # Calcula a distancia do filho esquerdo com o pai
                    dist_right = abs(tree[i][0] - tree[tree[i][2]][0])      # Calcula a distancia do filho direita com o pai
                    if dist_left <= dist_right:                             # Verifica se a distancia do filho esquerdo é menor do que da direita
                        left_idx = tree[i][1]                               # Salva a posição do filho da esquerda
                        right_idx = tree[i][2]                              # Salva a posição do filho da direita
                        cluster[left_idx] = cluster[i]                      # Insere o valor do cluster analisado na posição do filho esquerdo
                        if (tree[right_idx][1] != None) or (tree[right_idx][2] != None): # Verifica se o nó direito tem filhos
                            cluster[right_idx] = max(cluster) + 1           # Nesse caso, é criado um novo cluster na posição do filho direito (utilizando o maior valor da lista e somando 1)
                        else:                                               # Caso não tenha nenhum filho
                            cluster[right_idx] = cluster[i]                 # O filho direito recebe o mesmo cluster do esquerdo
                    else:                                                   # Caso a distancia do filho direito seja menor do que a da equerda
                        left_idx = tree[i][1]                               # Salva a posição do filho da esquerda
                        right_idx = tree[i][2]                              # Salva a posição do filho da direita
                        cluster[right_idx] = cluster[i]                     # Insere o valor do cluster analisado na posição do filho direito
                        if (tree[left_idx][1] != None) or tree[left_idx][2] != None: # Verifica se o nó direito tem filhos
                            cluster[left_idx] = max(cluster) + 1            # Nesse caso, é criado um novo cluster na posição do filho esquerdo (utilizando o maior valor da lista e somando 1)
                        else:                                               # Caso não tenha nenhum filho
                            cluster[left_idx] = cluster[i]                  # O filho esquerdo recebe o mesmo cluster do direito
                elif (tree[i][1] != None) and (tree[i][2] == None):         # Analisa se o nó possui somente o filho esquerdo
                    left_idx = tree[i][1]                                   # Salva a posição do filho da esquerda
                    cluster[left_idx] = cluster[i]                          # Insere o valor do cluster analisado na posição do filho esquerdo
                elif (tree[i][1] == None) and (tree[i][2] != None):         # Analisa se o nó possui somente o filho direito
                    right_idx = tree[i][2]                                  # Salva a posição do filho da direita
                    cluster[right_idx] = cluster[i]                         # Insere o valor do cluster analisado na posição do filho direito

            for i in range(len(tree)):                                      # Apos a clusterizção dos dados o loop for é feito para rotular todos os valores da arvore
                tree[i][3] = cluster[i]                                     # Adiciona o cluster relacionado ao nó na ultima posição da linha
            
            # Criação de uma matriz dos valores de x com seu determinado cluster #

            Data = [[None, None] for _ in range(len(x))]    # cria a matriz "Data" com duas colunas vazias e do mesmo tamanho de x
            for i in range(len(x)):                         # loop for para cada valor de x
                Data[i][0] = x[i]                           # Insere o valor de x_i na primeira posição da matriz na linha i
                for j in x:                                 # loop for onde j recebe todos os valores de x
                    if j == tree[i+1][0]:                   # se o valor de j for igual ao do nó correspondente na arvore (i+1, para pular a raiz da arvore)
                        Data[i][1] = tree[i+1][3]           # Insere o numero do clusters na segunda posição da matriz na linha i

            # FLR de X #

            # Criação da matriz "FLRx" informando o cluster e o cluster seguinte #

            FLRx = [[None, None] for _ in range(len(tree))] # cria a matriz "FLRx" com duas colunas vazias e do mesmo tamanho de tree
            
            for i in range(len(tree)):                      # loop for para analisar cada valor de tree
                FLRx[i][0] = tree[i][3]                     # Insere o valor do cluster na primeira posição da matriz na linha i
                if i + 1 < len(tree):                       # Verifica se existe um proximo valor, para evitar erros no programa
                    FLRx[i][1] = tree[i + 1][3]             # Insere o valor do proximo cluster na segunda posição da matriz na linha i
                if FLRx[i][1] == None:                      # Verifica se existe um proximo cluster, vendo se o cluster adicionado na segunda posição é vazio ou não
                    FLRx.pop()                              # Se for vazio, a linha é removida
                    
            # FLR para x e y #

            FLRxy = []                                      # Cria uma lista vazia para armazenar os valores dos clusters de exntrada e o seguinte cluster da saída
            
            for i in range(0,len(FLRy)):                    # Passa por todos os valores de FLRx e FLRy 
                FLRxy.append([FLRx[i][0],FLRy[i][1]])       # Adiciona uma linha nova na lista FLRx, o primeiro é o cluster de FLRx e o segundo é o proximo cluster de FLRy
            
            # FLRG #
            
            ValueAssociation = {}                           # Cria um dicionario vazio para armazenar as associações temporais

            for pair in FLRxy:                              # Loop para analisar todos os elementos da matriz FLRxy
                key, value = pair                           # Separa os valores de cada linha, sendo o primeiro a key e o segundo em value
                if key not in ValueAssociation:             # Confere se valor já não está no dicionario ou se não foi adicionado ainda
                    ValueAssociation[key] = set()           # Inclui a key no dicionario e evita linhas duplicadas
                if value is not None:                       # Se o valor não for vazio
                    ValueAssociation[key].add(value)        # Insere o value na key analisada
            
            # Converte o dicionário em lista um lista "FLRG" #
        
            FLRG = [[key] + sorted(list(values)) for key, values in ValueAssociation.items()]   # Adiciona os itens na lista sendo o primeiro da linha a key e os seguintes o value 
            FLRG.sort(key=lambda x: x[0])                                                       # Ordena a lista por chaves em ordem crescente para garantir a ordem correta

            # Criação do dicionario FLRG_ para armazenar os clusters e os valores contidos em cada um deles #

            FLRG_ = {}                                      # Cria um dicionario vazio 

            for pair in Data:                               # Loop para analisar todos os elementos da matriz Data
                value, cluster = pair                       # Separa os valores de cada linha, sendo o primeiro a key e o segundo em value
                if cluster not in FLRG_:                    # Confere se o cluster já está no dicionario ou se não foi adicionado ainda
                    FLRG_[cluster] = set()                  # Inclui o cluster no dicionario e evita linhas duplicadas
                if value is not None:                       # Se o valor não for vazio
                    FLRG_[cluster].add(value)               # Insere o value no cluster analisado
                
            # Cria um dicionario (FLRG_sorted) onde ordena o dicinario FLRG com os clusters em ordem crescente
            FLRG_sorted = {key: FLRG_[key] for key in sorted(FLRG_)}

            # Pontos medio de cada cluster #

            midPoint = []                                   # Cria uma lista vazia para armazenar os pontos medios de cada cluster
            
            for cluster, values in FLRG_sorted.items():     # Loop for dos items do diconario divididos em clusters e uma lista de valores
                if cluster:                                 # Verifica se o cluster não está vazio
                    associated_values = []                  # Cria uma lista auxiliar vazia para armazenar cada valor separadamente e cada loop reinicia a lista
                    for val in values:                      # Loop for para separar os valores da lista values
                        associated_values.append(val)       # Adicona os valores separados na lista auxiliar
                    if associated_values:                   # Verifica se a lista não está vazia 
                        average = (max(associated_values) + min(associated_values))/2 # Calcula a media dos valores do cluster, calculado pelo valor minimo + valor maximo dividido por 2
                        midPoint.append(average)            # Adiciona a media de cada clusters na lista dos pontos medios

            # Variaveis Fuzzy #
            
            A_i = []                                        # Cria uma lista vazia para armazenar o intervalo das variaveis fuzzy

            for i in FLRG_sorted:                                   # loop for para analisar cada valor do dicionario FLRG
                Ai = [min(FLRG_sorted[i]), max(FLRG_sorted[i]), i]  # Cria uma linha para cada intervalo, 1° - inicio do intervalo/2° - fim do intervalo/3° - cluster relacionado
                A_i.append(Ai)                                      # Adiciona as infromaçoes na lista A

            A_i.sort()                                      # Organiza os intervalos em ordem crescente

            # Preencher os espaços entre os intervalos para evitar erros em codigos #
            
            for i in range(0, len(A_i)-1):                  # Loop for para analisar os espaços entre o os intervalos, por conta disso não analisa o ultimo da lista
                
                A_diff = (A_i[i+1][0] - A_i[i][1])/2        # Calcula a diferença entre o fim do primeiro intervalo e incio do segundo, depois divide por 2
                A_i[i][1] = A_i[i][1] + A_diff              # O final do primeiro intervalo é modificado somando a metade da distancia
                A_i[i+1][0] = A_i[i+1][0] - A_diff          # o inicio do segundo intervalo é modificado subtraindo a metade da distancia
            
            # Ajustes nos intervalos para evitar erros #
            
            A_i[0][0] = -100000                                 # O inicio do primeiro intervalo é modificado para um valor muito baixo, para evitar algum valor fora dos intervalos
            A_i[-1][1] = A_i[-1][1]*10                          # O final do ultimo intervalo é modificado para um valor muito altp, para evitar algum valor fora dos intervalos

            # Criação do modelo #

            model_i = []                            # Cria uma lista vazia para armazenar as informações do modelo

            for i in range(0, len(FLRG)):                   # Loop for passa por todos os FLRG
                fore_mid = 0                                # Defina o valor inicial para a media, reinicia a cada loop
                n = (len(FLRG[i])-1)                        # Valor que divide na media dos valores, len(FLRG[i])-1 -> quantos elementos estão relacionados com o cluster, retira o primeiro
                for j in range(1, len(FLRG[i])):            # Loop for para analisar cada valor relacionado com o cluster
                    fore_mid += midPoint[(FLRG[i][j])-1]    # Somatorio de todos os pontos medios dos cluster relacionados no FLRG
                fore_mid = fore_mid/n                       # Faz a divisão pela quantidade de valores no somatorio
                model_i.append([fore_mid, i+1])             # Adicona esse valor a lista do modelo e adiciona o numero do cluster (i se inicia no 0, por isso i+1)

            # Salva o modelo de cada variavel de entrada #
            
            model.append(model_i)                           # Adiciona a o modelo na lista cria para todos os modelos
            A.append(A_i)                                   # Adiciona a os intervalos na lista cria para todos os intervalos

            # Junção das informações e saída da função #

            model_singh = [A, model]                        # Coloca ambas as nformações principais do modelo em uma unica variavel
            
            return model_singh, tree                        # Retorna a saida da função

def predict(X, model_singh):
    
    # Entrada -> X (dados a serem previstos) e model_singh (modelo treinado) / Saidas -> y (saidas previstas)

    if len(X.shape) == 1:   # Olha o tamanho da varivel de entrada, se for 1 gera o modelo para apenas uma variavel de entrada
        
        # Separação das variaveis dentro da variavel model_singh #
        
        A = model_singh[0]                          # Salva as informações dos intervalos na variavel A
        model = model_singh[1]                      # Salva as informações do modelo na variavel model

        X_A = []                                    # Cria uma lista vazia para armazenar e rotular os dados da previsão em seus devidos intevalos

        for i in X:                                 # Loop for para analisar todos os dados
            for j in range(0, len(A)):              # Loop for para analisar todos os intervalos
                if i > A[j][0] and A[j][1] >= i:    # Verifica se o valor está dentro do intervalo
                    X_A.append([i, A[j][2]])        # Se estiver, adiciona uma nova linha na lista com o primeiro sendo o valor e o segundo o intervalo

        NewData = []                                # Cria uma lista vazia para armazenar as previsões

        for i in range(0, len(X)):                  # Loop for para realizar as previsões

            forecast = model[X_A[i][1]-1][0]        # Analisa o intervalo do dado analisado com o modelo e salva a resposta na variavel
            NewData.append(forecast)                # Adiciona na lista o resultado da previsão

        return NewData                              # Retorna a saida da função
        
    else:   # No caso de possuir mais de uma variavel de entrada
        
        A_m = model_singh[0]                # Salva as informações dos intervalos na variavel A
        model_m = model_singh[1]            # Salva as informações do modelo na variavel model
        NewData_i = []                      # Cria uma lista vazia para armazenar as previsões de cada entrada
        
        for mi in range(0,len(model_m)):    # Loop for para analisar todas as variaveis
            
            # Salva cada variavel de entrada #
            
            A = A_m[mi]                                 # Salva as informações dos intervalos da variavel
            model = model_m[mi]                         # Salva as informações do modelo da variavel
            X_i = X[mi]                                 # Entrada especifica

            X_A = []                                    # Cria uma lista vazia para armazenar e rotular os dados da previsão em seus devidos intevalos

            for i in X_i:                               # Loop for para analisar todos os dados
                for j in range(0, len(A)):              # Loop for para analisar todos os intervalos
                    if i > A[j][0] and A[j][1] >= i:    # Verifica se o valor está dentro do intervalo
                        X_A.append([i, A[j][2]])        # Se estiver, adiciona uma nova linha na lista com o primeiro sendo o valor e o segundo o intervalo

            NewData_mi = []                             # Cria uma lista vazia para armazenar as previsões da variavel

            for i in range(0, len(X_i)):                # Loop for para realizar as previsões
                forecast = model[X_A[i][1]-1][0]        # Analisa o intervalo do dado analisado com o modelo e salva a resposta na variavel
                NewData_mi.append(forecast)             # Adiciona na lista o resultado da previsão
                
            NewData_i.append(NewData_mi)                # Adiciona as previsões de cada variavel na lista
            
        NewData = np.zeros(len(NewData_i[0]))       # Cria uma lista para receber a media das previsões
        n = len(NewData_i)                          # Numero de variaveis utilizadas
        
        for i in range(0, len(NewData_i[0])):       # Loop for para analisar cada valor previsto
            for j in range(0,len(NewData_i)):       # Loop for para analisar a previsão de cada variavel
                NewData[i] += NewData_i[j][i]       # Realiza a soma de todas as previsões
            NewData[i] = NewData[i]/n               # Faz a media do valor e salva na previsão final

        return NewData                              # Retorna a saida da função

def rmse(predictions, targets):                                     # Função para calculo do RMSE
    rmse_result = np.sqrt(np.mean((predictions - targets) ** 2))    # Reliza o calculo da função e salva na variavel rmse
    return rmse_result                                              # Retorna como valor final da função o resultado do rmse

def NYVE(targets):                  # Função para calculo do NYVE
    nyve = targets[:-1]             # Retira o ultimo valor da lista e cria uma lista onde a previsão é o resultado do dia anterior
    y = targets[1:]                 # Retira o primeiro valor da lista e cria a lista dos valores reais
    result = rmse(nyve, y)          # É realizado o rmse dessa previsão e salvo na variavel result
    return result                   # Retorna como valor final da função o resultado do NYVE

def graph(real, fore):

  fig = plt.figure(figsize=(10,8))      # Tamanho da figura
  ax1 = fig.add_subplot()               # Adiciona o grafico

  ax1.plot(real, label = 'Real', linestyle = '-', color = 'blue')         # Plotagem do sinal
  ax1.plot(fore, label = 'Predito', linestyle = '--', color = 'orange')  # Plotagem do sinal

  ax1.set_xlabel('Amostras')                   # Nome do eixo x
  ax1.set_ylabel('Particulado')                   # Nome do eixo y
  plt.grid()                            # Adiciona uma grade ao grafico
  plt.title('Real x Predito')          # Titulo do grafico
  plt.legend(loc = 4)                   # Legenda do grafico





"""
# View para a página web
def cluster_geografico(request):
    # Gere o gráfico chamando a função
    fig = clusters_plot(None, None, None)  # Substitua os argumentos conforme necessário

    # Converta o gráfico para HTML
    graph_html = to_html(fig, full_html=False)

    # Renderize o template com o gráfico
    return render(request, 'cluster_geografico.html', {'graph': graph_html})


def clusters_plot(x, y, ll):
    global db_heatmap
    if db_heatmap is not None:
        xk_heatmap = db_heatmap[['lat', 'lon']].values
        y = db_heatmap['cluster'].values
        latlon = db_heatmap[['lat', 'lon']].values

        # Geração do gráfico
        colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
        df_plot = {
            'latitude': latlon[:, 1],
            'longitude': latlon[:, 0],
            'cluster': y
        }
        fig = px.scatter(
            df_plot,
            x='latitude',
            y='longitude',
            color='cluster',
            color_discrete_sequence=colors,
            title='Clusterização',
            width=800,
            height=600
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black'),
            xaxis=dict(gridcolor='gray'),
            yaxis=dict(gridcolor='gray')
        )

        # Serializa o gráfico para JSON e retorna ao frontend
        #graph_json = pio.to_json(fig)
    return fig.show()#graph_json


#-----------------------------------------------------------------------------------------------------
def clusters_test_plot(x, ll, r0):

  print(f'r0 = {r0}')

  evol = EvolvingClustering(macro_cluster_update=1, variance_limit=r0, debug=False)
  evol.fit(x)

  y = evol.predict(x)

  print(f'clusters: {max(y) + 1}')

  # Definindo a paleta de cores
  colors = list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                              '#f781bf', '#a65628', '#984ea3',
                              '#999999', '#e41a1c', '#dede00']),
                      int(max(y) + 1)))

  # Criando um DataFrame para o Plotly
  df = {
      'lat': ll[:, 1],
      'lon': ll[:, 0],
      'cluster': y
  }
  
  # Criando o gráfico de dispersão com Plotly Express
  fig = px.scatter(
      df,
      x='lat',
      y='lon',
      color='cluster',
      color_discrete_sequence=colors,  # Usando a paleta de cores personalizada
      title='Evolving',
      width=800,
      height=600
  )

  # Atualizar o layout para deixar o fundo preto
  fig.update_layout(
      plot_bgcolor='white',     # Cor de fundo do gráfico
      paper_bgcolor='white',    # Cor de fundo ao redor do gráfico
      font=dict(color='black')  # Alterar a cor da fonte para branco (para contraste)
  )
  # Mostrando o gráfico
  return fig.show()
"""