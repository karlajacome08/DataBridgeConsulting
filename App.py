import os
import sys
import calendar
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
import requests
from streamlit_folium import st_folium
import streamlit.components.v1 as components  # <-- para incrustar HTML puro

COLOR_PRIMARY = "#001A57"
COLOR_SECUNDARY = "#000033"
COLOR_ACCENT = "#009944"
COLOR_NEGATIVE = "#E14B64"
COLOR_BG = "#e0e0e0"

# --------------------------------------
# Funciones auxiliares para manejar colores (mapa)
# --------------------------------------
def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    )

def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

def blend_with_white(base_rgb: tuple[int, int, int], pct: float) -> str:
    pct = max(min(pct, 100.0), 0.0)
    white_rgb = (255, 255, 255)
    factor = pct / 100.0
    blended = (
        int(white_rgb[0] + (base_rgb[0] - white_rgb[0]) * factor),
        int(white_rgb[1] + (base_rgb[1] - white_rgb[1]) * factor),
        int(white_rgb[2] + (base_rgb[2] - white_rgb[2]) * factor),
    )
    return rgb_to_hex(blended)

# --------------------------------------
# 1. Definir un único color base (púrpura) para el mapa
# --------------------------------------
COLOR_BASE_HEX = "#3E08A9"  # Púrpura intenso
BASE_RGB = hex_to_rgb(COLOR_BASE_HEX)

# --------------------------------------
# 2. Diálogos (sin cambios)
# --------------------------------------
@st.dialog("Optimizar rutas de entrega", width="large")
def dialog_optimizar_rutas():
    st.write("### Costo estimado por ruta")
    categorias1 = ["Ruta A", "Ruta B", "Ruta C"]
    valores1 = [120, 95, 130]
    fig1 = px.bar(x=categorias1, y=valores1, labels={'x': 'Ruta', 'y': 'Costo estimado'})
    fig1.update_layout(plot_bgcolor="#FFF", paper_bgcolor="#FFF", margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig1, use_container_width=True)
    st.write("Este gráfico ilustra el costo estimado para cada ruta. "
             "Ajusta las listas 'categorias1' y 'valores1' en el código "
             "para modificar los datos mostrados.")
    if st.button("Cerrar"):
        st.rerun()

@st.dialog("Mejorar gestión de stock", width="large")
def dialog_mejorar_stock():
    st.write("### Unidades en inventario por producto")
    categorias2 = ["Producto X", "Producto Y", "Producto Z"]
    valores2 = [450, 320, 275]
    fig2 = px.bar(x=categorias2, y=valores2, labels={'x': 'Producto', 'y': 'Unidades en Stock'})
    fig2.update_layout(plot_bgcolor="#FFF", paper_bgcolor="#FFF", margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig2, use_container_width=True)
    st.write("Este gráfico muestra la cantidad de unidades en stock para cada producto. "
             "Modifica 'categorias2' y 'valores2' en el código para actualizar los datos.")
    if st.button("Cerrar"):
        st.rerun()

@st.dialog("Ofertas segmentadas", width="large")
def dialog_ofertas_segmentadas():
    st.write("### Tasa de conversión por segmento")
    segmentos = ["Segmento A", "Segmento B", "Segmento C"]
    conversiones = [0.12, 0.08, 0.15]
    fig3 = px.bar(x=segmentos, y=conversiones, labels={'x': 'Segmento', 'y': 'Tasa de conversión'})
    fig3.update_layout(plot_bgcolor="#FFF", paper_bgcolor="#FFF", margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig3, use_container_width=True)
    st.write("Este gráfico representa la tasa de conversión por segmento. "
             "Cambia las listas 'segmentos' y 'conversiones' en el código "
             "para reflejar tus propios datos.")
    if st.button("Cerrar"):
        st.rerun()

# --------------------------------------
# 3. Configuración de la página y estilos
# --------------------------------------
st.set_page_config(page_title="Ingresos y Proyecciones", layout="wide", page_icon="🚚", initial_sidebar_state="expanded")

st.markdown(f"""
    <style>
    /* ========== GENERAL ========== */
           
    div.block-container {{
        padding-top: 0rem !important;
    }}
    div.stApp {{
        padding-top: 0rem !important;
    }}
            
    /* ========== SIDEBAR GENERAL ========== */
    section[data-testid="stSidebar"] {{
        background-color: {COLOR_BG};
        width: 400px !important;
    }}

    /* ========== TEXTO MÁS GRANDE EN TÍTULOS DE FILTROS ========== */
    section[data-testid="stSidebar"] label {{
        font-size: 3.5rem !important;
        font-weight: 600 !important;
        color: {COLOR_PRIMARY} !important;
        margin-bottom: 6px !important;
    }}

    section[data-testid="stSidebar"] .stMarkdown {{
        font-size: 1.2rem !important;
    }}

    section[data-testid="stSidebar"] h3 {{
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: {COLOR_PRIMARY} !important;
    }}

    section[data-testid="stSidebar"] h4 {{
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }}

    /* ========== FILTROS CON BORDE Y VISIBILIDAD ========== */
    section[data-testid="stSidebar"] div[data-baseweb="select"] {{
        border: 2px solid {COLOR_PRIMARY} !important;
        border-radius: 8px !important;
        background-color: #FFFFFF !important;
        box-shadow: 0 2px 8px rgba(123, 63, 242, 0.15) !important;
        min-height: 45px !important;
    }}

    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
        font-size: 1.18rem !important;
        font-weight: 400 !important;
        color: #222 !important;
        padding: 14px 18px !important;
        min-height: 55px !important;
        display: flex;
        align-items: center;
    }}

    section[data-testid="stSidebar"] div[data-baseweb="select"] svg {{
        width: 20px !important;
        height: 20px !important;
        color: {COLOR_PRIMARY} !important;
    }}

    /* ========== RECOMENDACIONES ========== */
    section[data-testid="stSidebar"] .stCheckbox {{
        transform: scale(1.3) !important;
    }}

    section[data-testid="stSidebar"] .stButton button {{
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        padding: 8px 16px !important;
        border: 2px solid {COLOR_PRIMARY} !important;
        border-radius: 8px !important;
        background-color: #FFFFFF !important;
        color: {COLOR_PRIMARY} !important;
        box-shadow: 0 2px 8px rgba(123, 63, 242, 0.10) !important;
    }}

    section[data-testid="stSidebar"] .stButton button:hover {{
        background-color: {COLOR_PRIMARY} !important;
        color: #FFFFFF !important;
    }}

    /* ========== TABS ========== */

    div[data-testid="stTabs"] [role="tablist"] > div:last-child {{
        display: none !important;
    }}

    div[data-testid="stTabs"] [role="tab"] {{
        color: #666666 !important;           /* color inactivo */
        font-size: 1.25rem !important;
        font-weight: 500 !important;
        padding: 0.75rem 1.5rem !important;
    }}

    div[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
        color: #0E2148 !important;
        border-bottom: 3px solid #0E2148 !important;
    }}

    /* ========== FILE UPLOADER ========== */
    section[data-testid="stSidebar"] .stFileUploader {{
        border: 2.5px dashed {COLOR_PRIMARY} !important;
        border-radius: 12px !important;
        padding: 10px !important;
        background-color: #FFFFFF !important;
        box-shadow: 0 4px 14px rgba(123, 63, 242, 0.12) !important;
    }}

    section[data-testid="stSidebar"] .stFileUploader label {{
        font-size: 1.3rem !important;
        font-weight: 400 !important;
        color: {COLOR_PRIMARY} !important;
    }}

    /* ========== KPI CARDS  ========== */
    .kpi-card {{
        background: #FFF;
        border-radius: 16px;
        box-shadow: 0 4px 16px rgba(123, 63, 242, 0.3);
        min-width: 180px;
        min-height: 170px;
        height: 100%;
        padding: 16px 14px 10px 14px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        box-sizing: border-box;
        margin-top:1.5rem;
    }}

    .kpi-label {{
        color: {COLOR_PRIMARY};
        font-weight: 700;
        font-size: 2.01rem;
        margin-bottom: 4px;
    }}

    .kpi-value-row {{
        display: flex;
        align-items: flex-end;
        justify-content: center;
        gap: 8px;
        width: 100%;
        margin-bottom: 2px;
    }}

    .kpi-value {{
        color: #222;
        font-size: 2.07rem;
        font-weight: 900;
        letter-spacing: -1px;
        line-height: 1;
    }}

    .kpi-delta-pos, .kpi-delta-neg {{
        font-size: 1.45rem;
        font-weight: 900;
        margin-left: 4px;
        display: flex;
        align-items: center;
        line-height: 1;
    }}
    .kpi-delta-pos {{ color: {COLOR_ACCENT}; }}
    .kpi-delta-neg {{ color: {COLOR_NEGATIVE}; }}
    .kpi-subtext {{
        color: #222;
        font-size: 1.10rem;
        margin-top: 4px;
        letter-spacing: 0.5px;
    }}
    </style>
""", unsafe_allow_html=True)

st.markdown(
        "<h1 style='text-align:start; color:#0E2148; margin-top:1.5rem;'>"
        "Ingresos y Proyecciones"
        "</h1>",
        unsafe_allow_html=True
        )



# --------------------
# Sidebar con filtros y recomendaciones
# --------------------
with st.sidebar:
    try:
        st.image("logo_danu.png", width=180)
    except:
        st.markdown("DANU ANALÍTICA")

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Filtros")

    if 'df' in st.session_state:
        df_filtros = st.session_state['df'].copy()
        df_filtros['orden_compra_timestamp'] = pd.to_datetime(df_filtros['orden_compra_timestamp'], errors='coerce')
        df_filtros = df_filtros.dropna(subset=['orden_compra_timestamp'])

        regiones = ["Todas las regiones"] + sorted(df_filtros['region'].dropna().unique().tolist())
        categorias = ["Todas las categorías"] + sorted(df_filtros['categoria_simplificada'].dropna().unique().tolist())
    else:
        regiones = ["Todas las regiones"]
        categorias = ["Todas las categorías"]

    periodo_options = ["Último año", "Últimos 6 meses", "Últimos 3 meses"]
    periodo_habilitados = ["Último año", "Últimos 6 meses", "Últimos 3 meses"]

    periodo_sel = st.selectbox("Periodo", periodo_options)
    if periodo_sel not in periodo_habilitados:
        st.warning("Opción no válida. Por favor selecciona una opción disponible.")
        st.stop()

    region_sel = st.selectbox("Región", regiones)
    categoria_sel = st.selectbox("Categoría", categorias)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    rec_keys = ['rec1', 'rec2', 'rec3']
    rec_defaults = [st.session_state.get(k, False) for k in rec_keys]
    recomendaciones_activadas = sum(rec_defaults)
    progreso_recomendaciones = int((recomendaciones_activadas / 3) * 100)

    st.markdown(
        f"<h4 style='margin-bottom: 0.5rem; color:{COLOR_PRIMARY}; font-size: 1.4rem;'>"
        f"Recomendaciones ({progreso_recomendaciones}%)</h4>",
        unsafe_allow_html=True
    )

    with st.sidebar:
        col_check1, col_text1 = st.columns([1, 10])
        rec1 = col_check1.checkbox(" ", value=rec_defaults[0], key='rec1')
        
        if col_text1.button("Optimizar rutas de entrega", key="btn_rec1"):
            dialog_optimizar_rutas()

        col_check2, col_text2 = st.columns([1, 10])
        rec2 = col_check2.checkbox(" ", value=rec_defaults[1], key='rec2')
        if col_text2.button("Mejorar gestión de stock", key="btn_rec2"):
            dialog_mejorar_stock()

        col_check3, col_text3 = st.columns([1, 10])
        rec3 = col_check3.checkbox(" ", value=rec_defaults[2], key='rec3')
        if col_text3.button("Ofertas segmentadas", key="btn_rec3"):
            dialog_ofertas_segmentadas()

    # Título grande antes del uploader
    st.markdown(
         "<span style='font-size:1.6rem; font-weight:700; color:#001A57;'>Subir base de datos</span>",
        unsafe_allow_html=True
)
    uploaded_file = st.file_uploader(
        "",
        type=["csv", "xlsx", "xls", "txt", "parquet"],
        help="Formatos soportados: CSV, Excel (xlsx, xls), TXT, Parquet"
    )
    if uploaded_file is not None:
        try:
            file_extension = uploaded_file.name.split(".")[-1].lower()
            if file_extension == "csv":
                df = pd.read_csv(uploaded_file)
            elif file_extension in ["xlsx", "xls"]:
                df = pd.read_excel(uploaded_file)
            elif file_extension == "parquet":
                df = pd.read_parquet(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file, sep=None, engine='python')
            df.to_parquet("df_DataBridgeConsulting.parquet", index=False)
            st.session_state['df'] = df
            st.success("¡Archivo cargado exitosamente!")

            resultado = subprocess.run(
                [sys.executable, "modelo_v1.py"],
                capture_output=True,
                text=True
            )
            
            if resultado.returncode == 0:
                st.success("✅ Modelo ejecutado correctamente")
                st.text("Salida del modelo:")
                st.code(resultado.stdout, language='bash')
            else:
                #st.error("❌ Error al ejecutar el modelo")
                #st.text("Detalles del error:")
                st.code(resultado.stderr, language='bash')

            if os.path.exists("prediccion_mes_siguiente.csv"):
                df_pred = pd.read_csv("prediccion_mes_siguiente.csv")
                df_pred["Tipo"] = "pred"
                st.success("Predicción mensual generada y cargada.")
            else:
                df_pred = pd.DataFrame()
                st.warning("No se generó predicción para el mes siguiente.")

        except Exception as e:
            st.error(f"Error general: {str(e)}")
    else:
        df_pred = pd.DataFrame()

# --------------------
# Función de filtrado
# --------------------
def aplicar_filtros(df, periodo, region, categoria):
    df_filtrado = df.copy()
    df_filtrado['orden_compra_timestamp'] = pd.to_datetime(df_filtrado['orden_compra_timestamp'], errors='coerce')
    df_filtrado = df_filtrado.dropna(subset=['orden_compra_timestamp'])
    
    fecha_max = df_filtrado['orden_compra_timestamp'].max()


    # Nuevas condiciones para los periodos
    if periodo == "Último año":
        fecha_limite = fecha_max - pd.DateOffset(years=1)
    elif periodo == "Últimos 6 meses":
        fecha_limite = fecha_max - pd.DateOffset(months=6)
    elif periodo == "Últimos 3 meses":
        fecha_limite = fecha_max - pd.DateOffset(months=3)
    else:
        fecha_limite = fecha_max - pd.DateOffset(years=100)  # Todos los datos

    df_filtrado = df_filtrado[df_filtrado['orden_compra_timestamp'] >= fecha_limite]

    if region != "Todas las regiones":
        df_filtrado = df_filtrado[df_filtrado['region'] == region]

    if categoria != "Todas las categorías":
        df_filtrado = df_filtrado[df_filtrado['categoria_simplificada'] == categoria]

    return df_filtrado


# --------------------
# Lógica principal: mostrar métricas y gráficos
# --------------------
tab1, tab2 = st.tabs(["Tablero", "Predicciones"])
with tab1:
    if 'df' in st.session_state:
        df_filtrado = aplicar_filtros(
            st.session_state['df'],
            periodo_sel,
            region_sel,
            categoria_sel
        )

        if len(df_filtrado) > 0:
            df_filtrado['año'] = df_filtrado['orden_compra_timestamp'].dt.year
            df_filtrado['mes'] = df_filtrado['orden_compra_timestamp'].dt.month
            df_filtrado['trimestre'] = df_filtrado['orden_compra_timestamp'].dt.quarter

        # ►►► CÓDIGO CORREGIDO ▼▼▼ (reemplazar todo el bloque de métricas)
        # 1. Determinar fechas del periodo actual filtrado
        fecha_min = df_filtrado['orden_compra_timestamp'].min()
        fecha_max = df_filtrado['orden_compra_timestamp'].max()

        # 2. Calcular periodo equivalente del año anterior
        fecha_min_anterior = fecha_min - pd.DateOffset(years=1)
        fecha_max_anterior = fecha_max - pd.DateOffset(years=1)

        # 3. Filtrar datos del año anterior CON LOS MISMOS FILTROS
        df_anterior = st.session_state['df'].copy()
        df_anterior['orden_compra_timestamp'] = pd.to_datetime(df_anterior['orden_compra_timestamp'])
        df_anterior = df_anterior[
            (df_anterior['orden_compra_timestamp'] >= fecha_min_anterior) & 
            (df_anterior['orden_compra_timestamp'] <= fecha_max_anterior)
        ]

        # Aplicar mismos filtros de región y categoría al periodo anterior
        if region_sel != "Todas las regiones":
            df_anterior = df_anterior[df_anterior['region'] == region_sel]
        if categoria_sel != "Todas las categorías":
            df_anterior = df_anterior[df_anterior['categoria_simplificada'] == categoria_sel]

        # 4. Calcular TODAS las métricas con el nuevo método
        # Ingresos
        ingresos_totales = df_filtrado['precio_final'].sum()
        ingresos_periodo_actual = df_filtrado['precio_final'].sum()
        ingresos_periodo_anterior = df_anterior['precio_final'].sum()
        delta_ingresos = (
            (ingresos_periodo_actual - ingresos_periodo_anterior) / ingresos_periodo_anterior * 100 
            if ingresos_periodo_anterior > 0 else 0.0
        )

        # Pedidos
        pedidos_totales = df_filtrado['order_id'].nunique()
        pedidos_periodo_actual = df_filtrado['order_id'].nunique()
        pedidos_periodo_anterior = df_anterior['order_id'].nunique()
        delta_pedidos = (
            (pedidos_periodo_actual - pedidos_periodo_anterior) / pedidos_periodo_anterior * 100 
            if pedidos_periodo_anterior > 0 else 0.0
        )

        # Valor promedio
        valor_promedio_actual = (
            ingresos_periodo_actual / pedidos_periodo_actual 
            if pedidos_periodo_actual > 0 else 0
        )
        valor_promedio_anterior = (
            ingresos_periodo_anterior / pedidos_periodo_anterior 
            if pedidos_periodo_anterior > 0 else 0
        )
        delta_valor = (
            (valor_promedio_actual - valor_promedio_anterior) / valor_promedio_anterior * 100 
            if valor_promedio_anterior > 0 else 0.0
        )

        # Flete promedio
        flete_promedio_actual = df_filtrado['costo_de_flete'].mean()
        flete_promedio_anterior = df_anterior['costo_de_flete'].mean()
        delta_flete = (
            (flete_promedio_actual - flete_promedio_anterior) / flete_promedio_anterior * 100 
            if flete_promedio_anterior > 0 else 0.0
        )

        comparacion_labels = {
        "Último año": "vs año anterior",
        "Últimos 6 meses": "vs mismos 6 meses año anterior",
        "Últimos 3 meses": "vs mismos 3 meses año anterior"
}
        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            color_ingresos = "kpi-delta-pos" if delta_ingresos >= 0 else "kpi-delta-neg"
            flecha = "↑" if delta_ingresos >= 0 else "↓"
            st.markdown(
                f"""<div class="kpi-card">
                        <div class="kpi-label">🏦 Ingresos Totales</div>
                        <div class="kpi-value-row">
                            <div class="kpi-value">${ingresos_totales:,.0f}</div>
                            <div class="{color_ingresos}">{abs(delta_ingresos):.1f}% {flecha}</div>
                        </div>
                        <div class="kpi-subtext">{comparacion_labels[periodo_sel]}</div>
                    </div>""",
                unsafe_allow_html=True
            )

        with col2:
            color_pedidos = "kpi-delta-pos" if delta_pedidos >= 0 else "kpi-delta-neg"
            flecha = "↑" if delta_pedidos >= 0 else "↓"
            st.markdown(
                f"""<div class="kpi-card">
                        <div class="kpi-label">📦 Pedidos Totales</div>
                        <div class="kpi-value-row">
                            <div class="kpi-value">{pedidos_totales:,}</div>
                            <div class="{color_pedidos}">{abs(delta_pedidos):.1f}% {flecha}</div>
                        </div>
                        <div class="kpi-subtext">{comparacion_labels[periodo_sel]}</div>
                    </div>""",
                unsafe_allow_html=True
            )

        with col3:
            color_valor = "kpi-delta-pos" if delta_valor >= 0 else "kpi-delta-neg"
            flecha = "↑" if delta_valor >= 0 else "↓"
            st.markdown(
                f"""<div class="kpi-card">
                        <div class="kpi-label">💵 Valor Promedio</div>
                        <div class="kpi-value-row">
                            <div class="kpi-value">${valor_promedio_actual:,.2f}</div>
                            <div class="{color_valor}">{abs(delta_valor):.1f}% {flecha}</div>
                        </div>
                        <div class="kpi-subtext">{comparacion_labels[periodo_sel]}</div>
                    </div>""",
                unsafe_allow_html=True
            )

        with col4:
            color_flete = "kpi-delta-neg" if delta_flete >= 0 else "kpi-delta-pos"
            flecha = "↑" if delta_flete >= 0 else "↓"
            st.markdown(
                f"""<div class="kpi-card">
                        <div class="kpi-label">🚚 Flete Promedio</div>
                        <div class="kpi-value-row">
                            <div class="kpi-value">${flete_promedio_actual:,.2f}</div>
                            <div class="{color_flete}">{abs(delta_flete):.1f}% {flecha}</div>
                        </div>
                        <div class="kpi-subtext">{comparacion_labels[periodo_sel]}</div>
                    </div>""",
                unsafe_allow_html=True
            )

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            # Gráfico de Tendencia Mensual
            st.markdown(
                f"<h4 style='color:{COLOR_PRIMARY}; font-size:1.75rem; margin-bottom:0.5rem;'>"
                "Tendencia de Ingresos Mensuales</h4>",
                unsafe_allow_html=True
            )
            df_filtrado['Año'] = df_filtrado['orden_compra_timestamp'].dt.year
            df_filtrado['Mes'] = df_filtrado['orden_compra_timestamp'].dt.month
            df_filtrado['Dia'] = df_filtrado['orden_compra_timestamp'].dt.day

            dias_por_mes = df_filtrado.groupby(['Año', 'Mes'])['Dia'].nunique().reset_index()
            dias_por_mes.rename(columns={'Dia': 'DiasRegistrados'}, inplace=True)

            if (region_sel == "Todas las regiones") and (categoria_sel == "Todas las categorías"):
                MIN_DIAS_MES = 28
                meses_validos = dias_por_mes[dias_por_mes['DiasRegistrados'] >= MIN_DIAS_MES][['Año', 'Mes']]
            else:
                # Tomar todos los meses sin filtrar por días
                meses_validos = dias_por_mes[['Año', 'Mes']]

            df_mensual = df_filtrado.groupby(['Año', 'Mes'])['precio_final'].sum().reset_index()
            df_mensual = pd.merge(df_mensual, meses_validos, on=['Año', 'Mes'], how='inner')
            df_mensual['Tipo'] = "real"

            if not df_pred.empty:
                df_pred_plot_total = df_pred[["Año", "Mes", "precio_final", "Tipo"]]
                df_total = pd.concat([df_mensual, df_pred_plot_total], ignore_index=True)
            else:
                df_total = df_mensual

            df_total = df_total.sort_values(["Año", "Mes"]).reset_index(drop=True)
            df_total["MesAbrev"] = df_total["Mes"].apply(lambda x: calendar.month_abbr[x])
            df_total["MesIndex"] = (
                df_total["Año"].astype(str) + "-" + df_total["Mes"].astype(str).str.zfill(2)
            )

            fig_tendencia = go.Figure()
            df_real = df_total[df_total["Tipo"] == "real"]
            df_pred_plot = df_total[df_total["Tipo"] == "pred"]

            fig_tendencia = go.Figure()

        # Línea de datos reales
        if not df_real.empty:
            fig_tendencia.add_trace(go.Scatter(
                x=df_real["MesIndex"],
                y=df_real["precio_final"],
                mode='lines+markers',
                name='Datos reales',
                line=dict(color="#001A57", width=3),
                marker=dict(size=8, color="#3B82F6")
            ))

            # Línea de predicción solo si hay predicción y datos reales
            # Solo mostrar la línea de predicción si los filtros están en "Todas"
            mostrar_prediccion = (region_sel == "Todas las regiones") and (categoria_sel == "Todas las categorías")
            
            if mostrar_prediccion and not df_pred_plot.empty and not df_real.empty:
                fig_tendencia.add_trace(go.Scatter(
                    x=[df_real["MesIndex"].iloc[-1]] + df_pred_plot["MesIndex"].tolist(),
                    y=[df_real["precio_final"].iloc[-1]] + df_pred_plot["precio_final"].tolist(),
                    mode='lines+markers',
                    name='Predicción mes siguiente',
                    line=dict(color='#555555', width=4, dash='dot'),
                    marker=dict(
                        size=[0] + [14] * len(df_pred_plot),
                        color=['#7B3FF2'] + ['#555555'] * len(df_pred_plot)
                    )
                ))

                pred_mes = df_pred_plot.iloc[0]
                fig_tendencia.add_annotation(
                    x=pred_mes["MesIndex"],
                    y=pred_mes["precio_final"],
                    text="Mes Predicho",
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-40,
                    font=dict(color="#333333", size=14, family="sans-serif"),
                    bgcolor="#FFF",
                    bordercolor="#111",
                    borderwidth=3,
                    arrowcolor="#111"
                )


            fig_tendencia.update_layout(
                height=500,
                plot_bgcolor="#FFF",
                paper_bgcolor="#FFF",
                margin=dict(l=20, r=20, t=30, b=20),
                font=dict(family="sans-serif", color="#222"),
                xaxis=dict(
                    title=None,
                    showgrid=False,
                    zeroline=False,
                    tickmode='array',
                    tickvals=df_total["MesIndex"],
                    ticktext=[row['MesAbrev'] for _, row in df_total.iterrows()],
                    tickfont=dict(
                        size=18,  # Más grande
                        family="sans-serif",
                        color="#222",
                    )
                ),
                yaxis=dict(
                    title=None,
                    showgrid=True,
                    gridcolor="#b0b0b0",
                    gridwidth=2,          # Más grueso
                    rangemode="tozero",
                    tickfont=dict(
                        size=18,  # Más grande
                        family="sans-serif",
                        color="#222",
                    )
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font = dict (
                        size = 16,
                        family="sans-serif",
                        color="#222"
                    )
                )
            )

            st.plotly_chart(fig_tendencia, use_container_width=True)

            # Gráficos por Región y Categoría
            col5, col6 = st.columns(2)
            with col5:
                st.markdown(
                    f"<h5 style='color:{COLOR_PRIMARY}; font-size:1.75rem; margin-bottom:0.5rem;'>"
                    "Ingresos por Región</h5>",
                    unsafe_allow_html=True
                )

                # ----------------------------
                # Construcción del mapa
                # ----------------------------
                ingresos_region = df_filtrado.groupby('region')['precio_final'].sum()
                total_ingresos_region = ingresos_region.sum()
                region_percent_map = (ingresos_region / total_ingresos_region * 100).round(1).to_dict()

                all_regions_map = ["Noroeste", "Noreste", "Centro", "Occidente", "Sureste"]
                for r in all_regions_map:
                    region_percent_map.setdefault(r, 0.0)

                valores_map = [region_percent_map[r] for r in all_regions_map]
                min_pct_map = min(valores_map)
                max_pct_map = max(valores_map)
                rango_map = max_pct_map - min_pct_map if max_pct_map > min_pct_map else 1.0

                geojson_url = (
                    "https://gist.githubusercontent.com/walkerke/"
                    "76cb8cc5f949432f9555/raw/"
                    "363c297ce82a4dcb9bdf003d82aa4f64bc695cf1/mx.geojson"
                )
                response = requests.get(geojson_url)
                mexico_geo = response.json()

                region_mapping_map = {
                    "Baja California": "Noroeste",
                    "Baja California Sur": "Noroeste",
                    "Sinaloa": "Noroeste",
                    "Sonora": "Noroeste",
                    "Chihuahua": "Noreste",
                    "Durango": "Noreste",
                    "Coahuila": "Noreste",
                    "Nuevo León": "Noreste",
                    "Tamaulipas": "Noreste",
                    "Hidalgo": "Centro",
                    "Puebla": "Centro",
                    "Tlaxcala": "Centro",
                    "Querétaro": "Centro",
                    "Ciudad de México": "Centro",
                    "México": "Centro",
                    "Morelos": "Centro",
                    "Aguascalientes": "Occidente",
                    "Guanajuato": "Occidente",
                    "San Luis Potosí": "Occidente",
                    "Zacatecas": "Occidente",
                    "Colima": "Occidente",
                    "Jalisco": "Occidente",
                    "Michoacán": "Occidente",
                    "Nayarit": "Occidente",
                    "Campeche": "Sureste",
                    "Quintana Roo": "Sureste",
                    "Tabasco": "Sureste",
                    "Veracruz": "Sureste",
                    "Yucatán": "Sureste",
                    "Chiapas": "Sureste",
                    "Guerrero": "Sureste",
                    "Oaxaca": "Sureste",
                }

                for feature in mexico_geo["features"]:
                    estado_geo = feature["properties"]["name"]
                    meso_geo = region_mapping_map.get(estado_geo)
                    pct_geo = region_percent_map.get(meso_geo, 0.0)
                    ingresos_geo = ingresos_region.get(meso_geo, 0.0)

                    if meso_geo is None:
                        feature["properties"]["region"] = "Sin datos"
                        feature["properties"]["percent_label"] = "0%"
                        feature["properties"]["ingresos"] = 0.0
                    else:
                        feature["properties"]["region"] = meso_geo
                        feature["properties"]["percent_label"] = f"{int(pct_geo)}%"
                        feature["properties"]["ingresos"] = ingresos_geo

                # Redefinimos el color base si lo deseas
                BASE_RGB = (0, 26, 87)

                def style_function(feature):
                    meso_feat = feature["properties"].get("region", "")
                    label_feat = feature["properties"].get("percent_label", "0%")
                    try:
                        pct_feat = float(label_feat.rstrip("%"))
                    except (ValueError, AttributeError):
                        pct_feat = 0.0

                    if meso_feat not in all_regions_map:
                        return {
                            "fillColor": "#D3D3D3",
                            "color": "black",
                            "weight": 1,
                            "fillOpacity": 0.4,
                        }

                    intensidad = (pct_feat - min_pct_map) / rango_map if rango_map > 0 else 1.0
                    intensidad = max(min(intensidad, 1.0), 0.0)
                    blend_pct = 30 + 70 * intensidad  # del 30% al 100%
                    fill_color_hex = blend_with_white(BASE_RGB, blend_pct)

                    return {
                        "fillColor": fill_color_hex,
                        "color": "black",
                        "weight": 1,
                        "fillOpacity": 0.8,
                    }

                # Creamos y mostramos el mapa
                m = folium.Map(location=[23.0, -102.0], zoom_start=5, tiles="cartodbpositron")
                folium.GeoJson(
                    data=mexico_geo,
                    style_function=style_function,
                    tooltip=folium.GeoJsonTooltip(
                        fields=["region", "ingresos", "percent_label"],
                        aliases=["Región:", "Ingresos totales:", "Porcentaje de ingresos:"],
                        localize=True,
                        sticky=False,
                    ),
                ).add_to(m)

                map_html = m.get_root().render()
                components.html(map_html, height=400, width=1200)


            with col6:
                st.markdown(
                    f"<h5 style='color:{COLOR_PRIMARY}; font-size:1.75rem; margin-bottom:0.5rem;'>"
                    "Costos Operativos</h5>",
                    unsafe_allow_html=True
                )

                # 1) Preparamos los datos
                bar_data = (
                    df_filtrado
                    .groupby('categoria_simplificada')['costo_de_flete']
                    .mean()
                    .reset_index()
                    .rename(columns={
                        'categoria_simplificada': 'Categoría',
                        'costo_de_flete': 'Costo Promedio de Flete'
                    })
                    .sort_values('Costo Promedio de Flete', ascending=False)
                    .reset_index(drop=True)
                )

                # 2) Etiqueta: primera palabra, salvo 'No Proporcionado'
                bar_data['Etiqueta'] = bar_data['Categoría'].apply(
                    lambda cat: cat if cat == "No proporcionado" else cat.split()[0]
                )

                # 3) Colores top 3 / resto
                colors = [
                    COLOR_PRIMARY if i < 3 else "#27548A"
                    for i in range(len(bar_data))
                ]

                # 4) Dibujamos la barra usando 'Etiqueta' para ticks
                fig_bar = go.Figure(go.Bar(
                    x=bar_data['Etiqueta'],
                    y=bar_data['Costo Promedio de Flete'],
                    marker=dict(color=colors),
                    text=bar_data['Costo Promedio de Flete'].map(lambda v: f"${v:.2f}"),
                    textposition='inside',
                    textfont=dict(color='white', size=16),
                    customdata=bar_data['Categoría'],  # para hover
                    hovertemplate="<b>%{customdata}</b><br>Costo: %{y:.2f}<extra></extra>"
                ))

                fig_bar.update_layout(
                    height=400,
                    plot_bgcolor="#FFF",
                    paper_bgcolor="#FFF",
                    margin=dict(l=20, r=20, t=30, b=100),
                    xaxis=dict(
                        title=None,
                        tickfont=dict(size=14, color="#222"),
                        tickangle=-45
                    ),
                    yaxis=dict(
                        title=None,
                        showgrid=True,
                        gridcolor="#b0b0b0",
                        gridwidth=1,
                        tickfont=dict(size=15, color="#222")
                    )
                )

                st.plotly_chart(fig_bar, use_container_width=True)
            
            st.markdown("""
            <style>
            #sugerencia-btn {
                position: fixed;
                bottom: 36px;
                right: 48px;
                z-index: 9999;
            }
            #sugerencia-btn button {
                background: #001A57;
                color: #fff;
                border: none;
                border-radius: 50px;
                box-shadow: 0 4px 16px rgba(0,0,0,0.15);
                padding: 18px 26px 18px 22px;
                font-size: 1.35rem;
                font-weight: 700;
                display: flex;
                align-items: center;
                gap: 10px;
                cursor: pointer;
                transition: background 0.2s;
            }
            #sugerencia-btn button:hover {
                background: #173b7b;
            }
            </style>
            <div id="sugerencia-btn">
            <button onclick="window.alert('¡Aquí irán las sugerencias del socio formador!')">
                <span style="font-size:1.65rem;">👩‍💼</span>
                <span>Sugerencias</span>
            </button>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.info("Sube tu base de datos para ver las métricas filtradas")

        col1, col2, col3, col4 = st.columns(4)
        for col, label in zip(
            [col1, col2, col3, col4],
            ["🏦 Ingresos Totales", "📦 Pedidos Totales", "💵 Valor Promedio", "🚚 Flete Promedio"]
        ):
            with col:
                st.markdown(
                    f"""<div class="kpi-card">
                            <div class="kpi-label">{label}</div>
                            <div class="kpi-value" style="color:#DDD;">---</div>
                            <div class="kpi-subtext" style="color:#DDD;">Sin datos</div>
                        </div>""",
                    unsafe_allow_html=True
                )

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        st.markdown(
            f"<h4 style='color:{COLOR_PRIMARY};font-size:1.75rem; margin-bottom:0.5rem;'>"
            "Tendencia de Ingresos Mensuales</h4>",
            unsafe_allow_html=True
        )
        fig_placeholder = px.line(pd.DataFrame({'x': [], 'y': []}), x='x', y='y')
        fig_placeholder.update_layout(
            height=350,
            plot_bgcolor="#FFF",
            paper_bgcolor="#FFF",
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            annotations=[dict(text="Sin datos", x=0.5, y=0.5, showarrow=False, font=dict(size=20))]
        )
        st.plotly_chart(fig_placeholder, use_container_width=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        col5, col6 = st.columns(2)
        with col5:
            st.markdown(
                f"<h5 style='color:{COLOR_PRIMARY}; font-size:1.75rem; margin-bottom:0.5rem;'>"
                "Ingresos por Región</h5>",
                unsafe_allow_html=True
            )
            # Mapa en estado 'sin datos'
            fig_empty_map = px.scatter_mapbox(pd.DataFrame({'lat': [], 'lon': []}), lat='lat', lon='lon')
            fig_empty_map.update_layout(
                mapbox={'style': "open-street-map", 'zoom': 4, 'center': {'lat': 23.0, 'lon': -102.0}},
                height=400,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig_empty_map, use_container_width=True)

        with col6:
            st.markdown(f"<h5 style='color:{COLOR_PRIMARY}; font-size:1.75rem; margin-bottom:0.5rem;'>Costos Operativos</h5>", unsafe_allow_html=True)
            fig_placeholder3 = px.scatter(pd.DataFrame({'x': [], 'y': []}), x='x', y='y')
            fig_placeholder3.update_layout(
                annotations=[dict(text="Sin datos", x=0.5, y=0.5, font_size=20, showarrow=False)],
                showlegend=False
            )
            st.plotly_chart(fig_placeholder3, use_container_width=True)

with tab2:
    st.write("Aquí irá el contenido del segundo tab. Por ahora, este texto.")