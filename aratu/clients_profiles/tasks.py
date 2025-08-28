# app/tasks.py
from celery import shared_task
from .models import AirQualityData  # ajuste pro seu modelo real
from .ml_utils import model_singh, predict, clusters_maia, convert_ndarray_to_list
from sensor.models import PredictedFile  # ajuste pro seu modelo real
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
from django.core.cache import cache
from .models import RegionResult
import json
import logging

logger = logging.getLogger(__name__)

#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------


@shared_task
def train_models_task(selected_clusters, forecast_period, user_id):
    
    #Treina modelos para os clusters selecionados usando dados armazenados no cache.
    #Retorna o ID do arquivo CSV gerado (PredictedFile).
    

    # Recupera dados do cache
    db_json = cache.get(f'db_heatmap_{user_id}')
    if not db_json:
        raise ValueError("Dados expiraram. Refaça a criação de clusters/regiões")

    db_heatmap = pd.read_json(db_json)

    if db_heatmap.empty:
        raise ValueError("Nenhum dado encontrado")

    models_results = []
    df_result = pd.DataFrame()
    last_date = db_heatmap['date'].max()

    for cluster in selected_clusters:
        print(f"Processando o cluster: {cluster}")
        cluster_db = db_heatmap[db_heatmap['cluster'] == cluster]

        if cluster_db.empty:
            print(f"Nenhum dado encontrado para o cluster {cluster}")
            continue

        # Verifica se colunas essenciais existem
        required_columns = ['total_pm', 'temp']
        missing_columns = [col for col in required_columns if col not in cluster_db.columns]
        if missing_columns:
            print(f"Colunas ausentes para o cluster {cluster}: {missing_columns}")
            continue

        # Converte para numpy arrays
        X_pm_array = np.array(cluster_db['total_pm'], dtype=float)
        X_temp_array = np.array(cluster_db['temp'], dtype=float)

        if len(X_pm_array) < 2:
            print(f"Dados insuficientes para o cluster {cluster}")
            continue

        # Criação dos modelos usando versão compatível com arrays
        model_pm, _ = model_singh(X_pm_array, X_pm_array)
        model_temp, _ = model_singh(X_temp_array, X_temp_array)

        # Previsão para o período solicitado
        yp_pm = []
        yp_temp = []

        cur_pm, cur_temp = X_pm_array[-1], X_temp_array[-1]

        for i in range(forecast_period):
            # ✅ Entrada para predict deve ser uma lista de arrays
            pred_temp = predict([np.array([cur_temp])], model_temp)[0]
            pred_pm = predict([np.array([cur_pm]), np.array([pred_temp])], model_pm)[0]

            yp_pm.append(pred_pm)
            yp_temp.append(pred_temp)
            cur_pm, cur_temp = pred_pm, pred_temp

            # Adiciona linha ao DataFrame final
            df_result = pd.concat([
                df_result,
                pd.DataFrame([{
                    'cluster': cluster,
                    'pm': pred_pm,
                    'temp': pred_temp,
                    'date': last_date + pd.DateOffset(days=i + 1)
                }])
            ], ignore_index=True)

        # Armazena resultados em lista (opcional, pode ser usado para debug)
        models_results.append({
            'cluster': cluster,
            'forecast': yp_pm,
            'temp': yp_temp
        })

    # Converte arrays em listas para consistência (se necessário)
    models_results = convert_ndarray_to_list(models_results)

    # Gera CSV em memória e salva como PredictedFile
    csv_buffer = io.StringIO()
    df_result.to_csv(csv_buffer, index=False)
    pf = PredictedFile.objects.create()
    pf.file.save("trained_models.csv", ContentFile(csv_buffer.getvalue()))

    # Retorna o ID do arquivo para o front-end baixar
    return {"file_id": pf.id}


"""
@shared_task
def train_models_task(selected_clusters, forecast_period, user_id):
    # Carrega dados do banco
    #qs = AirQualityData.objects.filter(cluster__in=selected_clusters).values(
    #    'cluster', 'date', 'total_pm', 'temp'
    #).order_by('date')
    #df_all = pd.DataFrame(list(qs))

    db_json = cache.get(f'db_heatmap_{user_id}')
    if not db_json:
        raise ValueError("Dados expiraram. Refaça a criação de clusters/regiões")

    df_all = pd.read_json(db_json)

    # Filtrar apenas os clusters selecionados pelo usuário

    if df_all.empty:
        raise ValueError("Nenhum dado encontrado")

    df_result = pd.DataFrame()
    last_date = df_all['date'].max()

    for cluster in selected_clusters:
        cluster_db = df_all[df_all['cluster'] == cluster]
        if cluster_db.empty:
            continue

        X_pm = np.array(cluster_db['total_pm'], dtype=float)
        X_temp = np.array(cluster_db['temp'], dtype=float)
        #X_pm = cluster_db['total_pm'].astype(float).values
        #X_temp = cluster_db['temp'].astype(float).values
        model_pm, _ = model_singh(X_pm, X_pm)
        model_temp, _ = model_singh(X_temp, X_temp)

        yp_pm = []
        yp_temp = []
        cur_pm, cur_temp = X_pm[-1], X_temp[-1]

        for i in range(forecast_period):
            pred_temp = predict(cur_temp, model_temp)[0]
            pred_pm = predict(cur_pm, pred_temp, model_pm)[0]
            yp_pm.append(pred_pm)
            yp_temp.append(pred_temp)
            cur_pm, cur_temp = pred_pm, pred_temp

            df_result = pd.concat([
                df_result,
                pd.DataFrame([{
                    'cluster': cluster,
                    'pm': pred_pm,
                    'temp': pred_temp,
                    'date': last_date + pd.DateOffset(days=i + 1)
                }])
            ])

    # gerar CSV em memória
    csv_buffer = io.StringIO()
    df_result.to_csv(csv_buffer, index=False)

    pf = PredictedFile.objects.create()
    pf.file.save("trained_models.csv", ContentFile(csv_buffer.getvalue()))
    return  {"file_id": pf.id}
"""

# app/tasks.py
import pandas as pd
import numpy as np
from celery import shared_task
from django.core.cache import cache
import io, json
from django.core.files.base import ContentFile

# substitua a implementação atual por esta (na parte onde está a versão "original")
@shared_task(bind=True)
def define_regions_task(self, user_id):
    task_id = self.request.id
    rr = RegionResult.objects.create(user_id=user_id, task_id=task_id, status="STARTED")
    cache_key = f'db_heatmap_{user_id}'
    db_json = cache.get(cache_key)
    if not db_json:
        # cria um RegionResult com erro (ajuda no troubleshooting via frontend)
        rr = RegionResult.objects.create(user_id=user_id, status="FAILURE", error="Dados expiraram. Refaça a criação de Clusters Geográficos.")
        return {"status": "FAILURE", "region_id": rr.id, "error": "Dados expiraram. Refaça a criação de Clusters Geográficos."}

    # reconstrói dataframe
    db_heatmap = pd.read_json(db_json)

    # lógica original de clusterização
    xk_heatmap = db_heatmap[['lat', 'lon']].astype(np.float64).values
    r0 = 5
    y = clusters_maia(xk_heatmap, r0)

    db_heatmap['total_pm'] = db_heatmap.get('pm1', 0) + db_heatmap.get('pm25', 0) + db_heatmap.get('pm10', 0)
    db_heatmap['cluster'] = y

    cache_key = f'db_heatmap_{user_id}'
    cache.set(cache_key, db_heatmap.to_json(orient='records'), timeout=900)
    # salvar csv em PredictedFile (opcional, mantenho comportamento anterior)
    csv_buffer = io.StringIO()
    db_heatmap.to_csv(csv_buffer, index=False)
    pf = PredictedFile.objects.create()
    pf.file.save("define_regions.csv", ContentFile(csv_buffer.getvalue()))

    # monta coordinates no formato esperado pelo front
    coordinates = [
        {
            "latitude": float(lat),
            "longitude": float(lon),
            "cluster": int(cluster)
        }
        for lat, lon, cluster in zip(xk_heatmap[:, 0], xk_heatmap[:, 1], y)
    ]

    # **cria um RegionResult e grava as coordinates** para que a view /region-result/<id>/ consiga recuperar
    #rr = RegionResult.objects.create(user_id=user_id, status="SUCCESS")
    rr.set_coordinates(coordinates)
    rr.status = "SUCCESS"
    rr.save(update_fields=["status"])

    # retorna o region_id (o polling / task_status espera isso)
    return {"status": "SUCCESS", "region_id": rr.id}