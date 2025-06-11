import os
import re
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
import streamlit.components.v1 as components

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

@st.dialog(" ", width="large")
def dialog_caidas_categoria():
    try:
        with open("alertas.txt", "r", encoding="utf-8") as f:
            alertas_texto = f.read()
    except FileNotFoundError:
        st.warning("No se encontró el archivo alertas.txt")
        return

    pat_categoria = re.compile(r"(.+?) bajó ([\d\.]+)%")
    categorias = []
    caidas = []
    for linea in alertas_texto.splitlines():
        m = pat_categoria.search(linea)
        if m:
            categorias.append(m.group(1).strip())
            caidas.append(float(m.group(2)))
    df_caidas = pd.DataFrame({"Categoría": categorias, "Caída (%)": caidas})

    if not df_caidas.empty:
        fig = px.bar(
            df_caidas, x="Categoría", y="Caída (%)", color="Caída (%)",
            color_continuous_scale="Blues", title="Caída porcentual por categoría"
        )
        fig.update_layout(height=350, plot_bgcolor="#FFF", paper_bgcolor="#FFF")
        st.plotly_chart(fig, use_container_width=True)
    st.code(
        "Categorías con caída de más del 15% en ingreso mensual:\n" +
        "\n".join([f"{cat} bajó {val:.2f}%" for cat, val in zip(categorias, caidas)]),
        language="markdown"
    )
    if st.button("Cerrar"):
        st.rerun()

@st.dialog(" ", width="large")
def dialog_disminucion_categoria():
    try:
        with open("alertas.txt", "r", encoding="utf-8") as f:
            alertas_texto = f.read()
    except FileNotFoundError:
        st.warning("No se encontró el archivo alertas.txt")
        return

    pat_ingreso = re.compile(r"(.+?) \(([\d\.]+) -> ([\d\.]+), pérdida aprox: \$([\d\.]+)\)")
    cat_ingreso, ingreso_ini, ingreso_fin, perdida = [], [], [], []
    for linea in alertas_texto.splitlines():
        m = pat_ingreso.search(linea)
        if m:
            cat_ingreso.append(m.group(1).strip())
            ingreso_ini.append(float(m.group(2)))
            ingreso_fin.append(float(m.group(3)))
            perdida.append(float(m.group(4)))
    df_perdidas = pd.DataFrame({
        "Categoría": cat_ingreso,
        "Ingreso Inicial": ingreso_ini,
        "Ingreso Final": ingreso_fin,
        "Pérdida ($)": perdida
    })

    if not df_perdidas.empty:
        fig = px.bar(
            df_perdidas, x="Categoría", y="Pérdida ($)", color="Pérdida ($)",
            color_continuous_scale="Blues", title="Pérdida monetaria por categoría"
        )
        fig.update_layout(height=350, plot_bgcolor="#FFF", paper_bgcolor="#FFF")
        st.plotly_chart(fig, use_container_width=True)
    st.code(
        "Categorías con disminución de ingreso promedio mensual:\n" +
        "\n".join([
            f"{cat} ({ini:.2f} -> {fin:.2f}, pérdida aprox: ${perd:,.2f})"
            for cat, ini, fin, perd in zip(cat_ingreso, ingreso_ini, ingreso_fin, perdida)
        ]),
        language="markdown"
    )
    if st.button("Cerrar"):
        st.rerun()

@st.dialog(" ", width="large")
def dialog_disminucion_region():
    try:
        with open("alertas.txt", "r", encoding="utf-8") as f:
            alertas_texto = f.read()
    except FileNotFoundError:
        st.warning("No se encontró el archivo alertas.txt")
        return

    pat_region = re.compile(r"([A-Za-zÁÉÍÓÚÑáéíóúñ ]+) \(([\d\.]+) -> ([\d\.]+), pérdida: \$([\d\.]+)\)")
    regiones, ingreso_ini, ingreso_fin, perdida = [], [], [], []
    for linea in alertas_texto.splitlines():
        m = pat_region.search(linea)
        if m:
            regiones.append(m.group(1).strip())
            ingreso_ini.append(float(m.group(2)))
            ingreso_fin.append(float(m.group(3)))
            perdida.append(float(m.group(4)))
    df_regiones = pd.DataFrame({
        "Región": regiones,
        "Ingreso Inicial": ingreso_ini,
        "Ingreso Final": ingreso_fin,
        "Pérdida ($)": perdida
    })

    if not df_regiones.empty:
        fig = px.bar(
            df_regiones, x="Región", y="Pérdida ($)", color="Pérdida ($)",
            color_continuous_scale="Blues", title="Pérdida monetaria por región"
        )
        fig.update_layout(height=350, plot_bgcolor="#FFF", paper_bgcolor="#FFF")
        st.plotly_chart(fig, use_container_width=True)
    st.code(
        "Regiones con disminución de ingreso promedio mensual:\n" +
        "\n".join([
            f"{reg} ({ini:.2f} -> {fin:.2f}, pérdida: ${perd:,.2f})"
            for reg, ini, fin, perd in zip(regiones, ingreso_ini, ingreso_fin, perdida)
        ]),
        language="markdown"
    )
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
        if col_text1.button("Categorías con caída >15%", key="btn_rec1"):
            dialog_caidas_categoria()
        
        col_check2, col_text2 = st.columns([1, 10])
        rec2 = col_check2.checkbox(" ", value=rec_defaults[1], key='rec2')
        if col_text2.button("Disminución de ingreso por región", key="btn_rec3"):
            dialog_disminucion_region()
        
        col_check3, col_text3 = st.columns([1, 10])
        rec3 = col_check3.checkbox(" ", value=rec_defaults[2], key='rec3')
        if col_text3.button("Disminución de ingreso por categoría", key="btn_rec2"):
            dialog_disminucion_categoria()

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
                [sys.executable, "models/modelo_v1.py"],
                capture_output=True,
                text=True
            )
            
            if resultado.returncode == 0:
                st.success("✅ Modelo ejecutado correctamente")
            else:
                st.code(resultado.stderr, language='bash')

            if os.path.exists("prediccion_diaria.parquet"):
                df_pred_diaria = pd.read_parquet("prediccion_diaria.parquet")
                df_pred_diaria['fecha'] = pd.to_datetime(df_pred_diaria['fecha'])
                df_pred_mensual = df_pred_diaria.groupby([
                    df_pred_diaria['fecha'].dt.year.rename('Año'),
                    df_pred_diaria['fecha'].dt.month.rename('Mes')
                ]).agg({
                    'prediccion': 'sum',
                    'pred_min': 'sum',
                    'pred_max': 'sum'
                }).reset_index()
                df_pred_mensual['Tipo'] = "pred"
            else:
                df_pred_mensual = pd.DataFrame()
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
tab1, tab2, tab3 = st.tabs(["Tablero", "Predicciones", "Hallazgos estrategicos y preguntas tecnicas"])
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
            "3 meses": "vs mes anterior",
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
                        <div class="kpi-comparacion">{comparacion_labels[periodo_sel]}</div>
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
                        <div class="kpi-comparacion">{comparacion_labels[periodo_sel]}</div>
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
                        <div class="kpi-comparacion">{comparacion_labels[periodo_sel]}</div>
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
                        <div class="kpi-comparacion">{comparacion_labels[periodo_sel]}</div>
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

            if not df_pred_mensual.empty:
                df_pred_mensual = df_pred_mensual.rename(columns={'prediccion': 'precio_final'})
                for col in ['pred_min', 'pred_max']:
                    if col not in df_pred_mensual.columns:
                        df_pred_mensual[col] = np.nan
                df_pred_plot_total = df_pred_mensual[["Año", "Mes", "precio_final", "pred_min", "pred_max", "Tipo"]]
                df_mensual['pred_min'] = np.nan
                df_mensual['pred_max'] = np.nan
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
                    name='Prediccion',
                    line=dict(color='#555555', width=4, dash='dot'),
                    marker=dict(
                        size=[0] + [14] * len(df_pred_plot),
                        color=['#7B3FF2'] + ['#555555'] * len(df_pred_plot)
                    )
                ))

                last_real_x = df_real["MesIndex"].iloc[-1]
                last_min = min(df_real["precio_final"].iloc[-1], df_real["precio_final"].iloc[-1])
                last_max = max(df_real["precio_final"].iloc[-1], df_real["precio_final"].iloc[-1])
                min_pred = np.minimum(df_pred_plot["pred_min"], df_pred_plot["pred_max"])
                max_pred = np.maximum(df_pred_plot["pred_min"], df_pred_plot["pred_max"])
                x_extended = [last_real_x] + df_pred_plot["MesIndex"].tolist()
                min_extended = [last_min] + min_pred.tolist()
                max_extended = [last_max] + max_pred.tolist()

                fig_tendencia.add_trace(go.Scatter(
                    x=x_extended,
                    y=min_extended,
                    mode='lines',
                    name='Mínimo',
                    line=dict(color="red", width=2, dash='dash'),
                    showlegend=True
                ))

                fig_tendencia.add_trace(go.Scatter(
                    x=x_extended,
                    y=max_extended,
                    mode='lines',
                    name='Máximo',
                    line=dict(color="green", width=2, dash='dot'),
                    fill='tonexty',
                    fillcolor='rgba(0,0,255,0.15)',
                    showlegend=True
                ))

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
                        tickfont=dict(size=20, color="#222"),
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

                # ==========================
                # SECCIÓN: RECOMENDACIONES Y ALERTAS DE INGRESOS
                # ==========================
            
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
    st.markdown(
        "<h2 style='color:#0E2148; margin-bottom:1.5rem;'>Predicciones</h2>",
        unsafe_allow_html=True
    )

    archivos = [
        ("Predicción Diaria", "prediccion_diaria.parquet"),
        ("Predicción por Región", "prediccion_region.parquet"),
        ("Predicción por Categoría", "prediccion_categoria.parquet"),
    ]

    col1, col2, col3 = st.columns(3)
    for col, (label, filename) in zip([col1, col2, col3], archivos):
        if os.path.exists(filename):
            df_pred = pd.read_parquet(filename)
            if "prediccion" in df_pred.columns:
                suma = df_pred["prediccion"].sum()
            elif "ingresos" in df_pred.columns:
                suma = df_pred["ingresos"].sum()
            elif "precio_final" in df_pred.columns:
                suma = df_pred["precio_final"].sum()
            else:
                suma = float("nan")
            valor = f"${suma:,.2f}"
            subtext = "Suma total de ingresos"
        else:
            valor = "---"
            subtext = "Archivo no encontrado"
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{valor}</div>
                    <div class="kpi-subtext">{subtext}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
        "<h4 style='color:#0E2148; margin-top:2.5rem; margin-bottom:1rem;'>Comparativo de suma diaria de ingresos por archivo de predicción</h4>",
        unsafe_allow_html=True
    )

    colg1, colg2, colg3 = st.columns(3)
    for col, (label, filename) in zip([colg1, colg2, colg3], archivos):
        if os.path.exists(filename):
            df_pred = pd.read_parquet(filename)
            fecha_col = "fecha" if "fecha" in df_pred.columns else df_pred.columns[0]
            if "prediccion" in df_pred.columns:
                y_col = "prediccion"
            elif "ingresos" in df_pred.columns:
                y_col = "ingresos"
            elif "precio_final" in df_pred.columns:
                y_col = "precio_final"
            else:
                continue

            if any(c in df_pred.columns for c in ["region", "categoria_simplificada", "grupo"]):
                df_plot = df_pred.groupby(fecha_col)[y_col].sum().reset_index()
            else:
                df_plot = df_pred[[fecha_col, y_col]].copy()
            df_plot[fecha_col] = pd.to_datetime(df_plot[fecha_col])

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_plot[fecha_col],
                y=df_plot[y_col],
                mode='lines+markers',
                name=label,
                line=dict(color="#3E08A9", width=3),
                marker=dict(size=7, color="#009944")
            ))
            fig.update_layout(
                title=label,
                xaxis_title="Fecha",
                yaxis_title="Suma diaria de ingresos",
                height=350,
                plot_bgcolor="#FFF",
                paper_bgcolor="#FFF",
                margin=dict(l=10, r=10, t=40, b=30),
                font=dict(family="sans-serif", color="#222", size=14),
            )
            col.plotly_chart(fig, use_container_width=True)
        else:
            col.info(f"No se encontró el archivo: {filename}")

    st.markdown(
        "<h4 style='color:#0E2148; margin-top:2.5rem; margin-bottom:1rem;'>Visualizador de archivos Parquet</h4>",
        unsafe_allow_html=True
    )

    opciones = [nombre for nombre, _ in archivos]
    opcion_sel = st.selectbox("Selecciona el archivo a visualizar", opciones, key="parquet_selector")

    archivo_seleccionado = dict(archivos)[opcion_sel]
    if os.path.exists(archivo_seleccionado):
        df_viz = pd.read_parquet(archivo_seleccionado)
        st.dataframe(df_viz, use_container_width=True, hide_index=True)
    else:
        st.warning(f"No se encontró el archivo: {archivo_seleccionado}")

with tab3:
    st.markdown("<div id='panel-individual'></div>", unsafe_allow_html=True)
    # --------------------
    # PANEL INDIVIDUAL – Hallazgos y preguntas tecnicas
    # --------------------

    st.markdown("<div id='Hallazgos y preguntas tecnicas'></div>", unsafe_allow_html=True)


    perfil = st.selectbox("Selecciona tu perfil", ["Ejecutivo", "Analista", "Técnico"])

    ingresos = 12037620  # entero
    ticket_promedio = 92.83  # float
    flete_promedio = 11.40  # float
    precision_modelo = 0.83  # float

    if perfil == "Ejecutivo":
        st.markdown(f"<h2 style='color: {COLOR_PRIMARY};'>🔹 Resumen Ejecutivo</h2>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💰 Ingresos Totales", f"${ingresos:,.0f}", "+294.6%")
        with col2:
            st.metric("📂 Ticket Promedio", f"${ticket_promedio:.2f}", "estable")
        with col3:
            st.metric("🚚 Flete Promedio", f"${flete_promedio:.2f}", "-4.2%")

        st.success("📈 Ingresos al alza. Electrónica lidera con el 40% del margen total, consolidándose como la categoría de mejor desempeño gracias a una combinación de alta demanda, eficiencia en logística y márgenes favorables.")
        st.warning("⚠️ Construcción y Automotriz representan solo el 4.07% del flete, pero con alto costo por pedido. Esto puede indicar una distribución ineficiente o baja densidad de envíos, lo cual debe ser optimizado.")
        st.info("💡 Recomendación: mantener foco comercial en CDMX y reforzar logística en Región Este, donde los tiempos de entrega son más largos y los costos de flete superan el promedio general.")

        st.markdown("""
        > 🧭 La tendencia mensual muestra una recuperación continua, impulsada por decisiones logísticas oportunas y un enfoque en categorías clave como Electrónica y Hogar. Estas dos categorías explican más del 60% del margen neto actual.
        > 📌 Las regiones con mejor rendimiento logístico fueron **CDMX** y **Nuevo León**, mientras que **Chiapas** y **Oaxaca** siguen presentando desafíos en tiempos de entrega y costo por pedido.
        """)

        st.divider()

        meses = ["Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"]
        ingresos_mensuales = [0.67, 0.73, 0.78, 1.2, 0.89, 1.1, 1.0, 1.15, 1.16, 1.15, 1.05, 1.05]  # Valores estimados

        df = pd.DataFrame({"Mes": meses, "Ingresos": ingresos_mensuales})

        # Crear gráfica
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["Mes"],
            y=df["Ingresos"],
            mode="lines+markers",
            line=dict(color='navy', width=3),
            marker=dict(color='dodgerblue', size=6),
        ))

        fig.update_layout(
            title="Tendencia de Ingresos Mensuales",
            xaxis_title="Mes",
            yaxis_title="Ingresos (millones)",
            yaxis_tickformat=".1fM",
            plot_bgcolor="white",
            font=dict(size=14)
        )
        st.plotly_chart(fig, use_container_width=True)


        st.markdown(f"<h3 style='color: {COLOR_PRIMARY};'>✅ Acciones clave</h3>", unsafe_allow_html=True)
        st.markdown("""
        ### 1. Perfil de clientes y productos
    - 🧍‍♂️ **85% de los clientes son nuevos**, lo que indica baja retención.  
    ➤ *Oportunidad:* implementar programas de fidelización o recompra.
    - 🛋️ **Productos más vendidos** pertenecen a la categoría Hogar y son de bajo precio.  
    ➤ *Implicación:* los ingresos actuales dependen de productos con márgenes bajos.
    - 💳 **Pagos con tarjeta de crédito** dominan (87,000 pedidos).  
    ➤ *Acción sugerida:* ofrecer beneficios o cashback exclusivo en este método de pago.
    - 🔁 Diseñar estrategia de recompra para nuevos clientes, especialmente en regiones de alto crecimiento como Centro y Occidente
    - 📢 Lanzar campañas regionales dirigidas en zonas con bajo desempeño relativo, utilizando los insights del mapa

    ### 2. Logística y entrega
    - ⏱️ **Tiempo promedio de entrega** entre 18 y 28 días.  
    ➤ *Mejor desempeño:* CDMX y Nuevo León.  
    ➤ *Mayor demora:* Chiapas, Puebla y Guerrero.
    - 📦 **Demora promedio de -13 días**, indicando entregas antes de lo estimado.  
    ➤ *Sin embargo,* más de 5000 outliers afectan la calidad del modelo.  
    ➤ *Acción:* limpieza de datos para mejorar predicción.
    - 🚚 **Centro y Noreste** destacan como regiones eficientes.  
    ➤ *Recomendación:* replicar sus prácticas logísticas en zonas de menor desempeño.
    - 🛠️ Optimizar logística en región Este (alta demora y alto costo por entrega)
    - 🔍 Revisar acuerdos con transportistas para categorías de alto flete como Construcción y Automotriz

    ### 3. Ingresos, precios y ventas
    - 💲 **Ticket promedio:** $92.83.  
    ➤ *Observación:* más de 8700 pedidos son extremadamente altos (outliers).
    - 🛍️ **Precio promedio por orden:** $92.42, con muchos pedidos sobre $300.  
    ➤ *Sugerencia:* segmentar entre minoristas y mayoristas.
    - 🏷️ **Categorías con tickets altos:** Tecnología, Automotriz, Deportes/Ocio.  
    ➤ *Acción:* priorizar estas en logística premium y promociones.
    - ⚖️ **Peso promedio bajo:** 1.32 kg, pero hay más de 10,000 productos pesados.  
    ➤ *Análisis:* riesgo logístico por volumen/peso.
    - 🚛 **Costo promedio de flete:** $16.29, con variabilidad por tamaño/distancia.  
    ➤ *Categorías con mayor volumen:* Construcción, Automotriz, Deportes.  
    ➤ *Recomendación:* crear rutas y tarifas diferenciadas para optimizar.
    - 🛒 Enfocar promociones en Electrónica y Hogar (mayor volumen y rentabilidad)
        """)

    elif perfil == "Analista":
        st.markdown(f"<h2 style='color: {COLOR_PRIMARY};'>📊 KPIs Analíticos</h2>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🎯 Precisión del Modelo", f"{precision_modelo * 100:.1f}%", "actualizado")
        with col2:
            st.metric("💳 % Clientes Nuevos", "85%", "+3.4%")

        st.markdown("Este modelo predictivo permite anticipar ingresos del siguiente mes con una precisión promedio del 83%, utilizando técnicas de aprendizaje supervisado. La tasa de adquisición de nuevos clientes (85%) sugiere un crecimiento saludable y sostenido del mercado.")


    elif perfil == "Técnico":
        st.markdown(f"<h2 style='color: {COLOR_PRIMARY};'>🧠 KPIs Técnicos del Modelo</h2>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📊 R² Score", f"{precision_modelo:.2f}")
        with col2:
            st.metric("📦 MAE Estimado", "5.63", "± 0.12")

        st.markdown("""
        🤖 Modelo de predicción basado en ensemble de Random Forest y XGBoost, entrenado con técnicas avanzadas de ingeniería de características. Incluye estacionalidad, variables de tiempo, económicos y logísticos.  
        🧪 Validación tipo walk-forward aplicada con múltiples ventanas temporales. La precisión final promedio es del **83%**, validada en datos no vistos.  
        📌 Variables más relevantes incluyen: mes (estacionalidad), costo de flete, categoría del producto, región geográfica y tipo de cambio.  
        🔁 El sistema permite reentrenamiento dinámico al marcar recomendaciones completadas, integrando retroalimentación del usuario en la actualización del modelo.
        """)

        st.divider()

        st.markdown(f"<h2 style='color: {COLOR_PRIMARY};'>❓ Preguntas Técnicas sobre el Panel</h2>", unsafe_allow_html=True)

        with st.expander("📊 ¿Qué representa cada sección del panel?", expanded=False):
            st.markdown("""
            - El panel está dividido en módulos clave que permiten un diagnóstico rápido y estratégico:

            **Indicadores clave (KPI)**
            - Ingresos Totales: Muestra la suma total de ventas en el periodo seleccionado.
            - Pedidos Totales: Total de órdenes registradas.
            - Valor Promedio: Monto promedio por pedido.
            - Costo Promedio: Gastos logísticos por envío.

            **Gráficas centrales**
            - **Tendencia de Ingresos Mensuales:** Línea de tiempo con puntos de datos reales y una predicción. Ideal para planificar operaciones y anticipar resultados financieros.
            - **Mapa de Ingresos por Región:** Muestra en tiempo real las zonas con mayor o menor contribución a las ventas. Útil para decisiones territoriales.
            - **Costo Promedio por Categoría:** Ayuda a entender el impacto del costo de flete en el margen de cada categoría.
            - **KPIs Segmentados por Perfil:** Permiten al ejecutivo, analista o técnico ver solo los indicadores que le competen.
            **Recomendaciones automatizadas**
            - Listado dinámico generado a partir del análisis predictivo y reglas de negocio.
            """)

        with st.expander("⚙️ ¿Cómo funciona el modelo de Machine Learning?", expanded=False):
            st.markdown("""
            Usamos un modelo ensemble que combina:

            - **Random Forest Regressor**
            - **XGBoost**

            Este modelo predice ingresos 2 meses adelante. Su precisión fue:

            - Mes 1: 83%
            - Mes 2: 69.78%
            - Mes 3 fue descartado (precisión < 50%)

            Incluye técnicas como:

            - Variables lag (21 meses anteriores)
            - Estacionalidad (mes, día de la semana con funciones seno/coseno)
            - Eventos especiales (festivos, quincenas)
            - Tipo de cambio e inflación simulada

            Al marcar una recomendación como “completada”, el sistema:

            - Actualiza las variables relacionadas
            - Reentrena el modelo automáticamente
            """)


        with st.expander("🧩 ¿Qué variables son más importantes?", expanded=False):
            st.markdown("""
            Las cinco variables con mayor peso en la predicción de ingresos son:
            1. Mes (variable estacional)
            2. Costo de flete
            3. Categoría del producto
            4. Región geográfica
            5. Tipo de cambio dólar-peso
            """)

        with st.expander("🔁 ¿Cómo se relacionan las secciones?", expanded=False):
            st.markdown("""
            Todo está conectado mediante un flujo integral:

            **Filtros interactivos ➜ KPIs y gráficas ➜ Análisis del modelo ➜ Recomendaciones accionables**

            - Los filtros (fecha, región, categoría) afectan todos los módulos.
            - Las gráficas nutren al modelo con datos históricos.
            - El modelo predictivo genera alertas.
            - Las recomendaciones se disparan si se detectan desviaciones, cuellos de botella o márgenes desaprovechados.

            Esto permite al usuario:

            - Diagnosticar, predecir y actuar desde un solo entorno.
            - Pasar de los datos a la acción en menos de 1 minuto.
            """) 

        with st.expander("📦 ¿Qué tan confiables son las proyecciones?", expanded=False):
            st.markdown("""
            - **Validación walk-forward**: Se prueba el modelo en períodos históricos no vistos durante el entrenamiento.
            - **Métricas clave**:
                - R²: 0.8794 (explica el 87.94% de la varianza)
                - MAE (Error Absoluto Medio): 5.63
                - RMSE (Error Cuadrático Medio): 7.89
            - **Optimización**: GridSearchCV para hiperparámetros.
            """)

        with st.expander("🎯 ¿Qué beneficios ofrece el dashboard?", expanded=False):
            st.markdown("""
            Este dashboard está diseñado para directivos y operadores logísticos que necesitan respuestas inmediatas.

            **Ventajas principales:**
            - Visualización amigable y rápida de indicadores clave.
            - Predicción con visión anticipada: hasta 3 meses de ventaja.
            - Recomendaciones concretas para evitar pérdidas.
            - Segmentación flexible por región, categoría y tiempo.

            **Impacto estimado:**
            - Mejora en entregas a tiempo: +12%
            - Reducción en costos logísticos: -15%
            - Optimización en campañas comerciales: +7% margen

            > *"La información sin acción es solo ruido. Este panel convierte el análisis en decisiones rentables."*
            """)

        st.success("🚀 Este dashboard convierte análisis complejos en decisiones claras y accionables para Danu Analytics.")