import os
import calendar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def cargar_datos(path):
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"Error al cargar datos: {e}")
        return None
    return df

def limpiar_y_agrupar(df):
    df['orden_compra_timestamp'] = pd.to_datetime(df['orden_compra_timestamp'])
    df['fecha'] = df['orden_compra_timestamp'].dt.normalize()
    df_diario = df.groupby('fecha')['precio_final'].sum().reset_index()
    df_diario.columns = ['ds', 'y']
    df_diario['ds'] = pd.to_datetime(df_diario['ds'])
    return df_diario

def limpiar_y_agrupar_por(df, columna):
    df['orden_compra_timestamp'] = pd.to_datetime(df['orden_compra_timestamp'])
    df['fecha'] = df['orden_compra_timestamp'].dt.normalize()
    df_diario = df.groupby(['fecha', columna])['precio_final'].sum().reset_index()
    df_diario = df_diario.rename(columns={'fecha': 'ds', columna: 'grupo', 'precio_final': 'y'})
    df_diario['ds'] = pd.to_datetime(df_diario['ds'])
    return df_diario

def tratar_outliers(df_diario):
    media = df_diario['y'].mean()
    std = df_diario['y'].std()
    z_score = (df_diario['y'] - media) / std
    outliers = np.abs(z_score) > 3
    df_diario.loc[outliers, 'y'] = df_diario['y'].rolling(7, min_periods=1, center=True).mean()[outliers]
    return df_diario

def agregar_variables_externas(df_diario):
    np.random.seed(42)
    df_diario['tipo_cambio'] = 18.0 + np.cumsum(np.random.normal(0, 0.02, len(df_diario)))
    df_diario['inflacion_mensual'] = 4.0 + np.sin(np.linspace(0, 6, len(df_diario))) + np.random.normal(0, 0.1, len(df_diario))
    return df_diario

def marcar_eventos(df):
    eventos = pd.DataFrame({
        'evento': ['Hot Sale', 'Buen Fin'],
        'inicio': [pd.Timestamp('2018-05-28'), pd.Timestamp('2018-11-16')],
        'fin':    [pd.Timestamp('2018-06-01'), pd.Timestamp('2018-11-19')]
    })
    df['evento_especial'] = 0
    for _, row in eventos.iterrows():
        mask = (df['ds'] >= row['inicio']) & (df['ds'] <= row['fin'])
        df.loc[mask, 'evento_especial'] = 1
    return df

def crear_features_temporales(df):
    df = df.copy()
    for i in range(1, 22):
        df[f'lag_{i}'] = df['y'].shift(i)
    df['dia_semana'] = df['ds'].dt.dayofweek
    df['dia_mes'] = df['ds'].dt.day
    df['mes'] = df['ds'].dt.month
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
    df['dia_semana_sin'] = np.sin(2 * np.pi * df['dia_semana'] / 7)
    df['dia_semana_cos'] = np.cos(2 * np.pi * df['dia_semana'] / 7)
    df['semana_mes'] = (df['ds'].dt.day - 1) // 7 + 1
    df['trimestre'] = df['ds'].dt.quarter
    df['anio'] = df['ds'].dt.year
    df['media_movil_7'] = df['y'].rolling(7).mean().shift(1)
    df['media_movil_15'] = df['y'].rolling(15).mean().shift(1)
    df['tendencia_lineal'] = np.arange(len(df))
    df['tendencia_cuadratica'] = df['tendencia_lineal']**2
    festivos = [
        '2017-01-01', '2018-01-01',
        '2017-02-05', '2018-02-05',
        '2017-03-21', '2018-03-21',
        '2017-04-14', '2018-03-30',
        '2017-05-01', '2018-05-01',
        '2017-09-16', '2018-09-16',
        '2017-11-02', '2018-11-02',
        '2017-11-20', '2018-11-20',
        '2017-12-25', '2018-12-25'
    ]
    df['festivo'] = df['ds'].isin(pd.to_datetime(festivos)).astype(int)
    df['quincena'] = df['ds'].dt.day.apply(lambda x: 1 if x <= 15 else 2)
    return df.dropna().reset_index(drop=True)

def predecir_mes_con_tendencia_ensemble(modelos, ultimos_datos, feature_names, fechas_pred, df_diario):
    predicciones = []
    pred_min_list = []
    pred_max_list = []
    lags = ultimos_datos.copy()
    ultimo_dia_conocido = df_diario['ds'].max()
    ingresos_mensuales = []
    for i in range(3, 0, -1):
        fecha = ultimo_dia_conocido - pd.DateOffset(months=i)
        primer_dia_mes = fecha.replace(day=1)
        ultimo_dia_mes = fecha.replace(day=calendar.monthrange(fecha.year, fecha.month)[1])
        mask = (df_diario['ds'] >= primer_dia_mes) & (df_diario['ds'] <= ultimo_dia_mes)
        total_mes = df_diario.loc[mask, 'y'].sum()
        ingresos_mensuales.append(total_mes)
    x = np.arange(len(ingresos_mensuales))
    y = np.array(ingresos_mensuales)
    pendiente, _ = np.polyfit(x, y, 1)
    factor_ajuste = 1
    if ingresos_mensuales[-1] != 0:
        factor_ajuste = 1 + (pendiente / abs(ingresos_mensuales[-1]))
    if pendiente < 0:
        factor_ajuste = max(0.7, min(factor_ajuste, 1))
    else:
        factor_ajuste = min(1.3, max(factor_ajuste, 1))
    for fecha_pred in fechas_pred:
        dia_semana = fecha_pred.dayofweek
        dia_mes = fecha_pred.day
        mes = fecha_pred.month
        mes_sin = np.sin(2 * np.pi * mes / 12)
        mes_cos = np.cos(2 * np.pi * mes / 12)
        dia_semana_sin = np.sin(2 * np.pi * dia_semana / 7)
        dia_semana_cos = np.cos(2 * np.pi * dia_semana / 7)
        semana_mes = (dia_mes - 1) // 7 + 1
        trimestre = ((fecha_pred.month - 1) // 3) + 1
        anio = fecha_pred.year
        media_movil_7 = np.mean(lags[-7:])
        media_movil_15 = np.mean(lags[-15:])
        tendencia_lineal = len(lags)
        tendencia_cuadratica = tendencia_lineal ** 2
        festivos_mx = [
            datetime(anio, 1, 1), datetime(anio, 2, 5), datetime(anio, 3, 21),
            datetime(anio, 5, 1), datetime(anio, 9, 16), datetime(anio, 11, 2),
            datetime(anio, 11, 20), datetime(anio, 12, 25)
        ]
        if anio == 2018:
            festivos_mx.append(datetime(2018, 3, 30))
        festivo = int(fecha_pred in festivos_mx)
        quincena = 1 if dia_mes <= 15 else 2
        tipo_cambio = 18.0
        inflacion_mensual = 4.0
        evento_especial = int(
            (fecha_pred >= datetime(2018, 5, 28) and fecha_pred <= datetime(2018, 6, 1)) or
            (fecha_pred >= datetime(2018, 11, 16) and fecha_pred <= datetime(2018, 11, 19))
        )
        features = (
            lags[-21:] +
            [dia_semana, dia_mes, mes, mes_sin, mes_cos, dia_semana_sin, dia_semana_cos, semana_mes, trimestre, anio,
            media_movil_7, media_movil_15, tendencia_lineal, tendencia_cuadratica,
            festivo, quincena, tipo_cambio, inflacion_mensual, evento_especial]
        )
        features_df = pd.DataFrame([features], columns=feature_names)
        pred_min = modelos[0].predict(features_df)[0]
        pred_max = modelos[1].predict(features_df)[0]
        pred = (pred_min + pred_max) / 2
        pred_ajustado = float(pred) * 1.2
        if pendiente > 0 and pred_ajustado < (np.mean(lags[-7:]) * 0.8):
            pred_ajustado = np.mean(lags[-7:]) * 1.2
        fecha_pasada = fecha_pred.replace(year=fecha_pred.year - 1)
        mes_pasado = fecha_pasada.month
        anio_pasado = fecha_pasada.year
        mask = (df_diario['ds'].dt.month == mes_pasado) & (df_diario['ds'].dt.year == anio_pasado)
        std_mes_pasado = df_diario.loc[mask, 'y'].std()
        if np.isnan(std_mes_pasado) or std_mes_pasado == 0:
            std_mes_pasado = 0.02 * pred_ajustado
        ruido = np.random.normal(loc=0, scale=std_mes_pasado)
        pred_ajustado = pred_ajustado + ruido

def predecir_mes_con_tendencia_ensemble(modelos, ultimos_datos, feature_names, fechas_pred, df_diario):
    predicciones = []
    pred_min_list = []
    pred_max_list = []
    lags = ultimos_datos.copy()
    ultimo_dia_conocido = df_diario['ds'].max()
    ingresos_mensuales = []
    for i in range(3, 0, -1):
        fecha = ultimo_dia_conocido - pd.DateOffset(months=i)
        primer_dia_mes = fecha.replace(day=1)
        ultimo_dia_mes = fecha.replace(day=calendar.monthrange(fecha.year, fecha.month)[1])
        mask = (df_diario['ds'] >= primer_dia_mes) & (df_diario['ds'] <= ultimo_dia_mes)
        total_mes = df_diario.loc[mask, 'y'].sum()
        ingresos_mensuales.append(total_mes)
    x = np.arange(len(ingresos_mensuales))
    y = np.array(ingresos_mensuales)
    pendiente, _ = np.polyfit(x, y, 1)
    factor_ajuste = 1
    if ingresos_mensuales[-1] != 0:
        factor_ajuste = 1 + (pendiente / abs(ingresos_mensuales[-1]))
    if pendiente < 0:
        factor_ajuste = max(0.7, min(factor_ajuste, 1))
    else:
        factor_ajuste = min(1.3, max(factor_ajuste, 1))
    for fecha_pred in fechas_pred:
        dia_semana = fecha_pred.dayofweek
        dia_mes = fecha_pred.day
        mes = fecha_pred.month
        mes_sin = np.sin(2 * np.pi * mes / 12)
        mes_cos = np.cos(2 * np.pi * mes / 12)
        dia_semana_sin = np.sin(2 * np.pi * dia_semana / 7)
        dia_semana_cos = np.cos(2 * np.pi * dia_semana / 7)
        semana_mes = (dia_mes - 1) // 7 + 1
        trimestre = ((fecha_pred.month - 1) // 3) + 1
        anio = fecha_pred.year
        media_movil_7 = np.mean(lags[-7:])
        media_movil_15 = np.mean(lags[-15:])
        tendencia_lineal = len(lags)
        tendencia_cuadratica = tendencia_lineal ** 2
        festivos_mx = [
            datetime(anio, 1, 1), datetime(anio, 2, 5), datetime(anio, 3, 21),
            datetime(anio, 5, 1), datetime(anio, 9, 16), datetime(anio, 11, 2),
            datetime(anio, 11, 20), datetime(anio, 12, 25)
        ]
        if anio == 2018:
            festivos_mx.append(datetime(2018, 3, 30))
        festivo = int(fecha_pred in festivos_mx)
        quincena = 1 if dia_mes <= 15 else 2
        tipo_cambio = 18.0
        inflacion_mensual = 4.0
        evento_especial = int(
            (fecha_pred >= datetime(2018, 5, 28) and fecha_pred <= datetime(2018, 6, 1)) or
            (fecha_pred >= datetime(2018, 11, 16) and fecha_pred <= datetime(2018, 11, 19))
        )
        features = (
            lags[-21:] +
            [dia_semana, dia_mes, mes, mes_sin, mes_cos, dia_semana_sin, dia_semana_cos, semana_mes, trimestre, anio,
            media_movil_7, media_movil_15, tendencia_lineal, tendencia_cuadratica,
            festivo, quincena, tipo_cambio, inflacion_mensual, evento_especial]
        )
        features_df = pd.DataFrame([features], columns=feature_names)
        pred_min = modelos[0].predict(features_df)[0]
        pred_max = modelos[1].predict(features_df)[0]
        pred = (pred_min + pred_max) / 2
        pred_ajustado = float(pred) * 1.2
        if pendiente > 0 and pred_ajustado < (np.mean(lags[-7:]) * 0.8):
            pred_ajustado = np.mean(lags[-7:]) * 1.2
        fecha_pasada = fecha_pred.replace(year=fecha_pred.year - 1)
        mes_pasado = fecha_pasada.month
        anio_pasado = fecha_pasada.year
        mask = (df_diario['ds'].dt.month == mes_pasado) & (df_diario['ds'].dt.year == anio_pasado)
        std_mes_pasado = df_diario.loc[mask, 'y'].std()
        if np.isnan(std_mes_pasado) or std_mes_pasado == 0:
            std_mes_pasado = 0.02 * pred_ajustado
        ruido = np.random.normal(loc=0, scale=std_mes_pasado)
        pred_ajustado = pred_ajustado + ruido

        pred_min_ajustado = float(pred_min) * 1.25 + ruido
        pred_max_ajustado = float(pred_max) * 1.15 + ruido

        predicciones.append(pred_ajustado)
        pred_min_list.append(pred_min_ajustado)
        pred_max_list.append(pred_max_ajustado)
        lags.append(pred_ajustado)
    return predicciones, pred_min_list, pred_max_list

def pipeline_prediccion_por_grupo_mes_a_mes(df, columna, nombre_archivo, df_diario_pred):
    grupos = df[columna].dropna().unique()
    datos_grupo = {}
    modelos_grupo = {}
    lags_grupo = {}
    feature_names_grupo = {}
    fechas_pred_meses = []

    for grupo in grupos:
        df_grupo = df[df[columna] == grupo].copy()
        if df_grupo.empty or df_grupo['precio_final'].sum() == 0:
            continue
        df_diario = limpiar_y_agrupar_por(df_grupo, columna)
        df_diario = tratar_outliers(df_diario)
        df_diario = agregar_variables_externas(df_diario)
        df_diario = marcar_eventos(df_diario)
        df_temporal = crear_features_temporales(df_diario)
        if len(df_temporal) < 30:
            continue
        X_train_final = df_temporal.drop(['ds', 'y', 'grupo'], axis=1)
        y_train_final = df_temporal['y']
        feature_names = X_train_final.columns
        rf_final = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_split=2, max_features='sqrt', random_state=42)
        rf_final.fit(X_train_final, y_train_final)
        xgb_final = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.1, random_state=42)
        xgb_final.fit(X_train_final, y_train_final)
        ultimo_dia = df_diario['ds'].max()
        datos_grupo[grupo] = {
            "df_diario": df_diario,
            "ultimo_dia": ultimo_dia
        }
        modelos_grupo[grupo] = [rf_final, xgb_final]
        lags_grupo[grupo] = df_temporal['y'].values[-21:].tolist()
        feature_names_grupo[grupo] = feature_names

    if datos_grupo:
        ultimo_dia_global = max([datos_grupo[grupo]["ultimo_dia"] for grupo in datos_grupo])
        for i in range(1, 3):
            mes = (ultimo_dia_global.month + i - 1) % 12 + 1
            anio = ultimo_dia_global.year + ((ultimo_dia_global.month + i - 1) // 12)
            primer_dia = datetime(anio, mes, 1)
            ultimo_dia_mes = datetime(anio, mes, calendar.monthrange(anio, mes)[1])
            fechas_pred_meses.append(list(pd.date_range(primer_dia, ultimo_dia_mes, freq='D')))

    resultados_todos = []
    for mes_idx, fechas_pred in enumerate(fechas_pred_meses):
        for grupo in datos_grupo:
            predicciones, _, _ = predecir_mes_con_tendencia_ensemble(
                modelos_grupo[grupo],
                lags_grupo[grupo],
                feature_names_grupo[grupo],
                fechas_pred,
                datos_grupo[grupo]["df_diario"]
            )
            lags_grupo[grupo].extend(predicciones)
            resultados = pd.DataFrame({
                'fecha': [f.date() for f in fechas_pred],
                'prediccion': predicciones,
                columna: grupo
            })
            resultados_todos.append(resultados)

    if resultados_todos:
        df_final = pd.concat(resultados_todos, ignore_index=True)
        df_total_diario = df_diario_pred[df_diario_pred['fecha'].isin(df_final['fecha'])]
        df_merged = df_final.merge(df_total_diario, on='fecha', suffixes=('', '_total'))
        df_sum = df_merged.groupby('fecha')['prediccion'].sum().reset_index()
        df_merged = df_merged.merge(df_sum, on='fecha', suffixes=('', '_sum'))
        df_merged['factor_ajuste'] = df_merged['prediccion_total'] / df_merged['prediccion_sum']
        df_merged['prediccion'] = df_merged['prediccion'] * df_merged['factor_ajuste']
        df_final = df_merged[['fecha', columna, 'prediccion']]
        output_path = os.path.abspath(nombre_archivo)
        df_final.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)

def generar_alertas_con_predicciones(df_real, df_pred_diaria, df_pred_region, df_pred_categoria):
    fecha_inicio_pred = pd.to_datetime(df_pred_diaria['fecha']).min()
    fecha_fin_pred = pd.to_datetime(df_pred_diaria['fecha']).max()
    fecha_inicio_4m = fecha_inicio_pred - pd.DateOffset(months=4)
    fecha_fin_4m = fecha_inicio_pred - pd.DateOffset(days=1)

    df_real['orden_compra_timestamp'] = pd.to_datetime(df_real['orden_compra_timestamp'])
    df_4m_real = df_real[(df_real['orden_compra_timestamp'] >= fecha_inicio_4m) &
                         (df_real['orden_compra_timestamp'] <= fecha_fin_4m)]

    cat_group = df_4m_real.groupby('categoria_simplificada').agg(
        ingreso_real_4m=('precio_final', 'sum'),
        pedidos_4m=('precio_final', 'count')
    ).reset_index()
    pred_cat_group = df_pred_categoria.groupby('categoria_simplificada').agg(
        ingreso_pred_2m=('prediccion', 'sum'),
        pedidos_2m=('prediccion', 'count')
    ).reset_index()

    alertas_caida_cat = []
    for _, row_real in cat_group.iterrows():
        cat = row_real['categoria_simplificada']
        ingreso_real_mensual = row_real['ingreso_real_4m'] / 4 if row_real['pedidos_4m'] > 0 else 0
        pred_row = pred_cat_group[pred_cat_group['categoria_simplificada'] == cat]
        if pred_row.empty:
            continue
        ingreso_pred_mensual = float(pred_row['ingreso_pred_2m'].values[0]) / 2 if float(pred_row['pedidos_2m'].values[0]) > 0 else 0
        if ingreso_real_mensual > 0:
            cambio_pct = ((ingreso_pred_mensual - ingreso_real_mensual) / ingreso_real_mensual) * 100
            if cambio_pct < -15:
                alertas_caida_cat.append(
                    f"{cat} bajó {abs(cambio_pct):.2f}%"
                )
    if alertas_caida_cat:
        alerta_caida_cat = "Categorías con caída de más del 15% en ingreso mensual:\n" + "\n".join(alertas_caida_cat)
    else:
        alerta_caida_cat = "Sin alerta de caídas mayores al 15% en ingreso mensual por categoría."

    alertas_cat = []
    for _, row_real in cat_group.iterrows():
        cat = row_real['categoria_simplificada']
        ingreso_real_mensual = row_real['ingreso_real_4m'] / 4 if row_real['pedidos_4m'] > 0 else 0
        volumen_cat_mensual = row_real['pedidos_4m'] / 4 if row_real['pedidos_4m'] > 0 else 0
        pred_row = pred_cat_group[pred_cat_group['categoria_simplificada'] == cat]
        if pred_row.empty or volumen_cat_mensual == 0:
            continue
        ingreso_pred_mensual = float(pred_row['ingreso_pred_2m'].values[0]) / 2 if float(pred_row['pedidos_2m'].values[0]) > 0 else 0
        perdida_cat = (ingreso_real_mensual - ingreso_pred_mensual) * volumen_cat_mensual * 2
        if ingreso_pred_mensual < ingreso_real_mensual and perdida_cat > 0:
            alertas_cat.append(
                f"{cat} ({ingreso_real_mensual:.2f} -> {ingreso_pred_mensual:.2f}, pérdida aprox: ${perdida_cat:.2f})"
            )
    if alertas_cat:
        alerta_cat = "Categorías con disminución de ingreso promedio mensual:\n" + "\n".join(alertas_cat)
    else:
        alerta_cat = "Sin alerta de disminución por categoría."

    reg_group = df_4m_real.groupby('region').agg(
        ingreso_real_4m=('precio_final', 'sum'),
        pedidos_4m=('precio_final', 'count')
    ).reset_index()
    pred_reg_group = df_pred_region.groupby('region').agg(
        ingreso_pred_2m=('prediccion', 'sum'),
        pedidos_2m=('prediccion', 'count')
    ).reset_index()

    alertas_reg = []
    for _, row_real in reg_group.iterrows():
        reg = row_real['region']
        ingreso_real_mensual = row_real['ingreso_real_4m'] / 4 if row_real['pedidos_4m'] > 0 else 0
        volumen_reg_mensual = row_real['pedidos_4m'] / 4 if row_real['pedidos_4m'] > 0 else 0
        pred_row = pred_reg_group[pred_reg_group['region'] == reg]
        if pred_row.empty or volumen_reg_mensual == 0:
            continue
        ingreso_pred_mensual = float(pred_row['ingreso_pred_2m'].values[0]) / 2 if float(pred_row['pedidos_2m'].values[0]) > 0 else 0
        perdida_reg = (ingreso_real_mensual - ingreso_pred_mensual) * volumen_reg_mensual * 2
        if ingreso_pred_mensual < ingreso_real_mensual and perdida_reg > 0:
            alertas_reg.append(
                f"{reg} ({ingreso_real_mensual:.2f} -> {ingreso_pred_mensual:.2f}, pérdida: ${perdida_reg:.2f})"
            )
    if alertas_reg:
        alerta_reg = "Regiones con disminución de ingreso promedio mensual:\n" + "\n".join(alertas_reg)
    else:
        alerta_reg = "Sin alerta de disminución por región."

    with open("alertas.txt", "w", encoding="utf-8") as f:
        f.write(alerta_caida_cat + "\n\n")
        f.write(alerta_cat + "\n\n")
        f.write(alerta_reg + "\n")

def main():
    df = cargar_datos("df_DataBridgeConsulting.parquet")
    if df is None:
        return

    df_diario = limpiar_y_agrupar(df)
    df_diario = tratar_outliers(df_diario)
    df_diario = agregar_variables_externas(df_diario)
    df_diario = marcar_eventos(df_diario)
    df_temporal = crear_features_temporales(df_diario)
    
    X_train_final = df_temporal.drop(['ds', 'y'], axis=1)
    y_train_final = df_temporal['y']
    feature_names = X_train_final.columns
    
    rf_final = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_split=2, max_features='sqrt', random_state=42)
    rf_final.fit(X_train_final, y_train_final)
    
    xgb_final = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.1, random_state=42)
    xgb_final.fit(X_train_final, y_train_final)
    
    ultimo_dia = df_diario['ds'].max()
    fechas_pred_todos = []
    for i in range(1, 3):
        mes = (ultimo_dia.month + i - 1) % 12 + 1
        anio = ultimo_dia.year + ((ultimo_dia.month + i - 1) // 12)
        primer_dia = datetime(anio, mes, 1)
        ultimo_dia_mes = datetime(anio, mes, calendar.monthrange(anio, mes)[1])
        fechas_pred_todos.extend(list(pd.date_range(primer_dia, ultimo_dia_mes, freq='D')))
    
    ultimos_lags = df_temporal['y'].values[-21:].tolist()
    predicciones_todas = []
    rf_preds_todas = []
    xgb_preds_todas = []
    for i in range(2):
        mes = (ultimo_dia.month + i + 1 - 1) % 12 + 1
        anio = ultimo_dia.year + ((ultimo_dia.month + i + 1 - 1) // 12)
        primer_dia = datetime(anio, mes, 1)
        ultimo_dia_mes = datetime(anio, mes, calendar.monthrange(anio, mes)[1])
        fechas_pred = pd.date_range(primer_dia, ultimo_dia_mes, freq='D')
        predicciones, rf_preds, xgb_preds = predecir_mes_con_tendencia_ensemble(
            [rf_final, xgb_final],
            ultimos_lags,
            feature_names,
            fechas_pred,
            df_diario
        )
        predicciones_todas.extend(predicciones)
        rf_preds_todas.extend(rf_preds)
        xgb_preds_todas.extend(xgb_preds)
        ultimos_lags.extend(predicciones)
    
    resultados = pd.DataFrame({
        'fecha': [f.date() for f in fechas_pred_todos],
        'prediccion': predicciones_todas,
        'pred_min': rf_preds_todas,
        'pred_max': xgb_preds_todas
    })
    
    output_path = os.path.abspath("prediccion_diaria.parquet")
    resultados.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)

    pipeline_prediccion_por_grupo_mes_a_mes(df, 'region', 'prediccion_region.parquet', resultados)
    pipeline_prediccion_por_grupo_mes_a_mes(df, 'categoria_simplificada', 'prediccion_categoria.parquet', resultados)

    df_pred_diaria = pd.read_parquet("prediccion_diaria.parquet")
    df_pred_region = pd.read_parquet("prediccion_region.parquet")
    df_pred_categoria = pd.read_parquet("prediccion_categoria.parquet")

    generar_alertas_con_predicciones(df, df_pred_diaria, df_pred_region, df_pred_categoria)

if __name__ == "__main__":
    main()
