import os
import calendar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    df = pd.read_excel("df_DataBridgeConsulting.xlsx")
except Exception as e:
    print("Error:", e)
df['orden_compra_timestamp'] = pd.to_datetime(df['orden_compra_timestamp'])
df['fecha'] = df['orden_compra_timestamp'].dt.normalize()

df_diario = df.groupby('fecha')['precio_final'].sum().reset_index()
df_diario.columns = ['ds', 'y']
df_diario['ds'] = pd.to_datetime(df_diario['ds'])

media = df_diario['y'].mean()
std = df_diario['y'].std()
z_score = (df_diario['y'] - media) / std
outliers = np.abs(z_score) > 3
df_diario.loc[outliers, 'y'] = df_diario['y'].rolling(7, min_periods=1, center=True).mean()[outliers]

np.random.seed(42)
df_diario['tipo_cambio'] = 18.0 + np.cumsum(np.random.normal(0, 0.02, len(df_diario)))
df_diario['inflacion_mensual'] = 4.0 + np.sin(np.linspace(0, 6, len(df_diario))) + np.random.normal(0, 0.1, len(df_diario))

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

df_diario = marcar_eventos(df_diario)

def crear_features_temporales_mejoradas(df_diario):
    df = df_diario.copy()
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
    df['año'] = df['ds'].dt.year
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

df_temporal = crear_features_temporales_mejoradas(df_diario)

def walk_forward_validation_ensemble(df_temporal, n_months_val=3):
    scores = []
    preds = []
    trues = []
    fechas = []
    maes = []
    rmses = []
    for i in range(n_months_val, 0, -1):
        ultima_fecha = df_temporal['ds'].max() - pd.DateOffset(months=i-1)
        primer_dia_val = ultima_fecha.replace(day=1)
        ultimo_dia_val = ultima_fecha.replace(day=calendar.monthrange(ultima_fecha.year, ultima_fecha.month)[1])
        train = df_temporal[df_temporal['ds'] < primer_dia_val]
        val = df_temporal[(df_temporal['ds'] >= primer_dia_val) & (df_temporal['ds'] <= ultimo_dia_val)]
        if len(val) == 0 or len(train) < 50: continue
        X_train = train.drop(['ds', 'y'], axis=1)
        y_train = train['y']
        X_val = val.drop(['ds', 'y'], axis=1)
        y_val = val['y']
        rf = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_split=2, max_features='sqrt', random_state=42)
        xgb = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.1, random_state=42)
        rf.fit(X_train, y_train)
        xgb.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_val)
        y_pred_xgb = xgb.predict(X_val)
        y_pred = (y_pred_rf + y_pred_xgb) / 2
        preds.extend(y_pred)
        trues.extend(y_val)
        fechas.extend(val['ds'])
        scores.append(r2_score(y_val, y_pred))
        maes.append(mean_absolute_error(y_val, y_pred))
        rmses.append(np.sqrt(mean_squared_error(y_val, y_pred)))
    # print(f"MAE promedio walk-forward ensemble: {np.mean(maes):.2f}")
    # print(f"RMSE promedio walk-forward ensemble: {np.mean(rmses):.2f}")
    return fechas, trues, preds

fechas, y_true, y_pred = walk_forward_validation_ensemble(df_temporal, n_months_val=2)

# plt.figure(figsize=(14,6))
# plt.plot(fechas, y_true, label='Real')
# plt.plot(fechas, y_pred, label='Predicción Ensemble', color='orange')
# plt.title('Validación walk-forward (Ensemble): Real vs Predicción')
# plt.xlabel('Fecha')
# plt.ylabel('Ventas diarias')
# plt.legend()
# plt.tight_layout()
# plt.show()

def predecir_mes_con_tendencia_ensemble(modelos, ultimos_datos, feature_names, fechas_pred, df_diario):
    predicciones = []
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
    if ingresos_mensuales[-1] != 0:
        factor_ajuste = 1 + (pendiente / abs(ingresos_mensuales[-1]))
    else:
        factor_ajuste = 1
    if pendiente < 0:
        factor_ajuste = max(0.7, min(factor_ajuste, 1))
        tendencia_txt = "↓ BAJADA"
    else:
        factor_ajuste = min(1.3, max(factor_ajuste, 1))
        tendencia_txt = "↑ SUBIDA"
    print(f"\nTendencia detectada ultimos 3 meses: {tendencia_txt}")
    print(f"Factor de ajuste aplicado: {factor_ajuste:.2f}x")
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
        año = fecha_pred.year
        media_movil_7 = np.mean(lags[-7:])
        media_movil_15 = np.mean(lags[-15:])
        tendencia_lineal = len(lags)
        tendencia_cuadratica = tendencia_lineal ** 2
        festivos_mx = [
            datetime(año, 1, 1), datetime(año, 2, 5), datetime(año, 3, 21),
            datetime(año, 5, 1), datetime(año, 9, 16), datetime(año, 11, 2),
            datetime(año, 11, 20), datetime(año, 12, 25)
        ]
        if año == 2018:
            festivos_mx.append(datetime(2018, 3, 30))
        festivo = int(fecha_pred in festivos_mx)
        quincena = 1 if dia_mes <= 15 else 2
        tipo_cambio = 18.0
        inflacion_mensual = 4.0
        evento_especial = 1 if (
            (fecha_pred >= datetime(2018, 5, 28) and fecha_pred <= datetime(2018, 6, 1)) or
            (fecha_pred >= datetime(2018, 11, 16) and fecha_pred <= datetime(2018, 11, 19))
        ) else 0

        features = (
            lags[-21:] +
            [dia_semana, dia_mes, mes, mes_sin, mes_cos, dia_semana_sin, dia_semana_cos, semana_mes, trimestre, año,
            media_movil_7, media_movil_15, tendencia_lineal, tendencia_cuadratica,
            festivo, quincena, tipo_cambio, inflacion_mensual, evento_especial]
        )
        features_df = pd.DataFrame([features], columns=feature_names)
        pred_rf = modelos[0].predict(features_df)[0]
        pred_xgb = modelos[1].predict(features_df)[0]
        pred = (pred_rf + pred_xgb) / 2
        pred_ajustado = float(pred) * 1.2
        if pendiente > 0 and pred_ajustado < (np.mean(lags[-7:]) * 0.8):
            pred_ajustado = np.mean(lags[-7:]) * 1.2
        predicciones.append(pred_ajustado)
        lags.append(pred_ajustado)
    return predicciones

ultima_fecha = df_temporal['ds'].max()
primer_dia_ultimo_mes = ultima_fecha.replace(day=1)
train_final = df_temporal.copy()
X_train_final = train_final.drop(['ds', 'y'], axis=1)
y_train_final = train_final['y']
feature_names = X_train_final.columns

rf_final = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_split=2, max_features='sqrt', random_state=42)
rf_final.fit(X_train_final, y_train_final)
xgb_final = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.1, random_state=42)
xgb_final.fit(X_train_final, y_train_final)

ultimo_dia = df_diario['ds'].max()

fechas_pred_todos = []
for i in range(1, 3):
    if ultimo_dia.month + i > 12:
        mes = (ultimo_dia.month + i) % 12
        anio = ultimo_dia.year + ((ultimo_dia.month + i - 1) // 12)
    else:
        mes = ultimo_dia.month + i
        anio = ultimo_dia.year
    primer_dia = datetime(anio, mes, 1)
    ultimo_dia_mes = datetime(anio, mes, calendar.monthrange(anio, mes)[1])
    fechas_pred_todos.extend(list(pd.date_range(primer_dia, ultimo_dia_mes, freq='D')))

ultimos_lags = df_temporal['y'].values[-21:].tolist()
predicciones_todas = []

for i in range(2):
    if ultimo_dia.month + i + 1 > 12:
        mes = (ultimo_dia.month + i + 1) % 12
        anio = ultimo_dia.year + ((ultimo_dia.month + i) // 12)
    else:
        mes = ultimo_dia.month + i + 1
        anio = ultimo_dia.year
    primer_dia = datetime(anio, mes, 1)
    ultimo_dia_mes = datetime(anio, mes, calendar.monthrange(anio, mes)[1])
    fechas_pred = pd.date_range(primer_dia, ultimo_dia_mes, freq='D')
    
    predicciones = predecir_mes_con_tendencia_ensemble(
        [rf_final, xgb_final],
        ultimos_lags,
        feature_names,
        fechas_pred,
        df_diario
    )
    predicciones_todas.extend(predicciones)
    ultimos_lags.extend(predicciones)

resultados = pd.DataFrame({
    'Fecha': [f.date() for f in fechas_pred_todos],
    'Predicción': predicciones_todas
})

print("\nPredicción para el mes siguiente:")
print(resultados.round(2).to_string(index=False))

resultados['año_anterior'] = resultados['Fecha'].apply(lambda x: x.replace(year=x.year-1))
resultados['año_anterior'] = pd.to_datetime(resultados['año_anterior'])
historico_mes_anterior = df_diario[df_diario['ds'].isin(resultados['año_anterior'])]
total_predicho = resultados['Predicción'].sum()
total_historico = historico_mes_anterior['y'].sum() if not historico_mes_anterior.empty else 0
diferencia = total_predicho - total_historico
tendencia = "↑ MÁS INGRESOS" if diferencia > 0 else "↓ MENOS INGRESOS"

precision = None
if total_historico > 0:
    precision = (1 - abs(total_predicho - total_historico) / total_historico) * 100

print("\nHistórico año anterior:")
if not historico_mes_anterior.empty:
    print(historico_mes_anterior.rename(columns={'ds':'Fecha', 'y':'Real'}).round(2).to_string(index=False))
else:
    print("No hay datos históricos para comparar.")

print(f"\nTotal predicho: ${total_predicho:,.2f}")
print(f"\nTotal año anterior: ${total_historico:,.2f}")
# print(f"\nTendencia: {tendencia}")

ultimo_real = df_diario[['ds', 'y']].iloc[-1]
prediccion_conectada = pd.concat([
    pd.DataFrame({'Fecha': [ultimo_real['ds'].date()], 'Predicción': [ultimo_real['y']]}),
    resultados
], ignore_index=True)

# plt.figure(figsize=(14,6))
# plt.plot(df_diario['ds'], df_diario['y'], label='Histórico real')
# plt.plot(prediccion_conectada['Fecha'], prediccion_conectada['Predicción'], label='Predicción mes siguiente (Ensemble)', color='orange')
# plt.axvline(resultados['Fecha'].iloc[0], color='red', linestyle='--', label='Inicio predicción')
# plt.legend()
# plt.title('Ventas diarias: Real vs Predicción (Ensemble)')
# plt.xlabel('Fecha')
# plt.ylabel('Ventas')
# plt.tight_layout()
# plt.show()

resultados_mes = resultados.copy()
resultados_mes["Mes"] = pd.to_datetime(resultados_mes["Fecha"]).dt.month
resultados_mes["Año"] = pd.to_datetime(resultados_mes["Fecha"]).dt.year
df_pred_mes = resultados_mes.groupby(["Año", "Mes"]).agg({"Predicción": "sum"}).reset_index()
df_pred_mes.rename(columns={"Predicción": "precio_final"}, inplace=True)
df_pred_mes["Tipo"] = "pred"

output_path = os.path.abspath("prediccion_mes_siguiente.csv")

if df_pred_mes.empty:
    print("Advertencia: df_pred_mes está vacío. No se guardará el archivo CSV.")
else:
    try:
        df_pred_mes.to_csv(output_path, index=False, encoding="utf-8")
    except Exception as e:
        print(f"Error al guardar el archivo CSV: {e}")