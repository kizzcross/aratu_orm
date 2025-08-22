# app/tasks.py
from celery import shared_task
from .models import AirQualityData  # ajuste pro seu modelo real
from .ml_utils import model_singh, predict, clusters_maia
from sensor.models import PredictedFile  # ajuste pro seu modelo real
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
from django.core.cache import cache
from .models import RegionResult
import json
import logging

logger = logging.getLogger(__name__)

@shared_task(bind=True)
def define_regions_task(self, user_id):
    """
    Pega o cache db_heatmap_{user_id}, roda o processamento (clustering),
    salva o resultado em RegionResult e retorna {"region_id": id}.
    """
    cache_key = f"db_heatmap_{user_id}"
    try:
        raw = cache.get(cache_key)
        if raw is None:
            msg = "Dados não encontrados no cache. Certifique-se de usar cache compartilhado (redis)."
            logger.error(msg)
            # criar registro com erro para permitir troubleshooting via front
            rr = RegionResult.objects.create(user_id=user_id, status="FAILURE", error=msg)
            return {"status": "FAILURE", "region_id": rr.id, "error": msg}

        # se você armazenou JSON no cache:
        import pandas as pd
        db_heatmap = pd.read_json(raw) if isinstance(raw, str) else pd.DataFrame.from_records(raw)

        # --- Aqui vai sua lógica de clustering / define regions ---
        # Preferível delegar para uma função já existente (ml_utils.define_regions)
        # Exemplo genérico (substitua pela sua lógica):
        def run_clustering(df):
            # retornar lista de polígonos/centroids/ex.: [{"id":1,"coords":[...]}]
            # -> substitua pela função real
            coords = [{"id": 1, "coords": [[-43.0, -19.9], [-43.1, -19.9], [-43.1, -20.0]]}]
            return coords

        coordinates = run_clustering(db_heatmap)

        rr = RegionResult.objects.create(user_id=user_id, status="SUCCESS")
        rr.set_coordinates(coordinates)

        # Retorna identificador pro front se quiser exibir logo no polling
        return {"status": "SUCCESS", "region_id": rr.id}
    except Exception as e:
        logger.exception("Erro ao definir regiões")
        rr = RegionResult.objects.create(user_id=user_id, status="FAILURE", error=str(e))
        # opcional: re-raise para Celery registrar stacktrace
        raise
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------

@shared_task
def train_models_task(selected_clusters, forecast_period):
    # Carrega dados do banco
    qs = AirQualityData.objects.filter(cluster__in=selected_clusters).values(
        'cluster', 'date', 'total_pm', 'temp'
    ).order_by('date')
    df_all = pd.DataFrame(list(qs))

    if df_all.empty:
        raise ValueError("Nenhum dado encontrado")

    df_result = pd.DataFrame()
    last_date = df_all['date'].max()

    for cluster in selected_clusters:
        cluster_db = df_all[df_all['cluster'] == cluster]
        if cluster_db.empty:
            continue

        X_pm = cluster_db['total_pm'].astype(float).values
        X_temp = cluster_db['temp'].astype(float).values
        model_pm, _ = model_singh(X_pm, X_pm)
        model_temp, _ = model_singh(X_temp, X_temp)

        yp_pm = []
        yp_temp = []
        cur_pm, cur_temp = X_pm[-1], X_temp[-1]

        for i in range(forecast_period):
            pred_temp = predict([cur_temp], model_temp)[0]
            pred_pm = predict([cur_pm, pred_temp], model_pm)[0]
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
    return pf.id


# app/tasks.py
import pandas as pd
import numpy as np
from celery import shared_task
from django.core.cache import cache
import io, json
from django.core.files.base import ContentFile


@shared_task
def define_regions_task(user_id):
    cache_key = f'db_heatmap_{user_id}'
    db_json = cache.get(cache_key)
    if not db_json:
        raise ValueError("Dados expiraram. Refaça a criação de Clusters Geográficos.")

    db_heatmap = pd.read_json(db_json)

    # Lógica original
    xk_heatmap = db_heatmap[['lat', 'lon']].astype(np.float64).values
    r0 = 5
    y = clusters_maia(xk_heatmap, r0)

    db_heatmap['total_pm'] = db_heatmap['pm1'] + db_heatmap['pm25'] + db_heatmap['pm10']
    db_heatmap["cluster"] = y

    # Salvar resultado como CSV (em PredictedFile)
    csv_buffer = io.StringIO()
    db_heatmap.to_csv(csv_buffer, index=False)

    pf = PredictedFile.objects.create()
    pf.file.save("define_regions.csv", ContentFile(csv_buffer.getvalue()))

    # Retorno simplificado para o front-end
    coordinates = [
        {
            "latitude": float(lat),
            "longitude": float(lon),
            "cluster": int(cluster)
        }
        for lat, lon, cluster in zip(xk_heatmap[:, 0], xk_heatmap[:, 1], y)
    ]

    return {"file_id": pf.id, "coordinates": coordinates}
