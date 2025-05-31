import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import calendar

# Paleta de colores
COLOR_PRIMARY = "#7B3FF2"
COLOR_ACCENT = "#23C16B"
COLOR_NEGATIVE = "#E14B64"
COLOR_BG = "#F6F6FB"

# Configuración de la página
st.set_page_config(page_title="Panel de Entregas", layout="wide", page_icon="🚚", initial_sidebar_state="expanded")

# --- CSS personalizado ---
st.markdown(f"""
    <style>
    body {{
        background-color: {COLOR_BG};
    }}
    .main {{
        background-color: {COLOR_BG};
    }}
    .kpi-card {{
        background: #FFF;
        border-radius: 14px;
        box-shadow: 0 2px 12px rgba(123, 63, 242, 0.08);
        padding: 18px 18px 10px 18px;
        margin: 0 8px 0 0;
        min-height: 90px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}
    .kpi-label {{
        color: {COLOR_PRIMARY};
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 4px;
    }}
    .kpi-value {{
        color: #222;
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 2px;
    }}
    .kpi-delta-pos {{
        color: {COLOR_ACCENT};
        font-size: 1rem;
        font-weight: 600;
    }}
    .kpi-delta-neg {{
        color: {COLOR_NEGATIVE};
        font-size: 1rem;
        font-weight: 600;
    }}
    .kpi-subtext {{
        color: #888;
        font-size: 0.95rem;
        font-weight: 400;
        margin-top: 0px;
    }}
    .divider {{
        border-bottom: 2px solid #ECEAF6;
        margin: 18px 0 18px 0;
    }}
    .sidebar-divider {{
        border-bottom: 1.5px solid #ECEAF6;
        margin: 16px 0 16px 0;
    }}
    </style>
""", unsafe_allow_html=True)

# --- Título principal centrado en la parte blanca ---
st.markdown(
    """
    <h1 style='text-align:center; margin-bottom: 0.5rem;'>
        Panel de Entregas 🚚
    </h1>
    """,
    unsafe_allow_html=True
)

# --- Sidebar ---
with st.sidebar:
    
    # Logo
    st.image("logo_danu.png", width=180)
    
    # Filtros
    st.markdown("### Filtros")
    st.selectbox("Periodo", ["Último mes"])
    st.selectbox("Región", ["Todas las regiones"])
    st.selectbox("Categoría", ["Todas las categorías"])
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    
    # Recomendaciones
    st.markdown(f"### <span style='color:{COLOR_PRIMARY};'>Recomendaciones</span>", unsafe_allow_html=True)
    st.checkbox("Optimizar rutas de entrega\nReducir tiempos en zona Este")
    st.checkbox("Aumentar capacidad logística\nAlmacenamiento en Barcelona")
    st.checkbox("Promocionar Electrónica\nMayor margen de beneficio")
    st.checkbox("Revisar proveedores\nReducir costos de envío")
    st.checkbox("Implementar seguimiento GPS\nPara entregas en tiempo real")
    
    # Línea divisoria
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    
    # Botón para subir base de datos
    uploaded_file = st.file_uploader(
        "Subir base de datos",
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
            st.session_state['df'] = df
            with st.expander("Vista previa de los datos cargados"):
                st.dataframe(df.head(10), height=300)
            st.success("¡Archivo cargado exitosamente!")
        except Exception as e:
            st.error(f"Error al cargar el archivo: {str(e)}")

# --- KPIs en tarjetas blancas ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        f"""<div class="kpi-card">
                <div class="kpi-label">Ingresos Totales</div>
                <div class="kpi-value">$48,250</div>
                <div class="kpi-delta-pos">+8.3% ↑</div>
                <div class="kpi-subtext">vs mes anterior</div>
            </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(
        f"""<div class="kpi-card">
                <div class="kpi-label">Pedidos Totales</div>
                <div class="kpi-value">248</div>
                <div class="kpi-delta-pos">+12% ↑</div>
                <div class="kpi-subtext">vs mes anterior</div>
            </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(
        f"""<div class="kpi-card">
                <div class="kpi-label">Valor Promedio</div>
                <div class="kpi-value">$194.56</div>
                <div class="kpi-delta-neg">-3.2% ↓</div>
                <div class="kpi-subtext">vs trimestre anterior</div>
            </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(
        f"""<div class="kpi-card">
                <div class="kpi-label">Flete Promedio</div>
                <div class="kpi-value">$12.75</div>
                <div class="kpi-delta-pos">+1.5% ↑</div>
                <div class="kpi-subtext">vs año anterior</div>
            </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Gráfica de tendencia de ingresos con predicción CONTINUA ---
if 'df' in st.session_state:
    df_real = st.session_state['df'].copy()
    
    # Convertir a fecha y extraer mes
    df_real['orden_pago_aprobado'] = pd.to_datetime(df_real['orden_pago_aprobado'], errors='coerce')
    df_real = df_real.dropna(subset=['orden_pago_aprobado', 'precio_final'])

    df_real['Mes'] = df_real['orden_pago_aprobado'].dt.month
    ingresos_reales = df_real.groupby('Mes')['precio_final'].sum().reset_index()
    ingresos_reales['MesNombre'] = ingresos_reales['Mes'].apply(lambda x: calendar.month_name[x])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ingresos_reales['MesNombre'],
        y=ingresos_reales['precio_final'],
        mode='lines+markers',
        name='Datos reales',
        line=dict(color=COLOR_PRIMARY, width=3),
        marker=dict(size=8, color=COLOR_PRIMARY)
    ))

    # Buscar predicciones si se suben
    pred_file = st.file_uploader("Sube archivo de predicción mensual", type=["csv", "xlsx"])
    if pred_file:
        try:
            ext = pred_file.name.split(".")[-1]
            df_pred = pd.read_csv(pred_file) if ext == "csv" else pd.read_excel(pred_file)
            if 'Mes' in df_pred.columns and 'Ingreso_Predicho' in df_pred.columns:
                df_pred['MesNombre'] = df_pred['Mes'].apply(lambda x: calendar.month_name[int(x)])
                fig.add_trace(go.Scatter(
                    x=df_pred['MesNombre'],
                    y=df_pred['Ingreso_Predicho'],
                    mode='lines+markers',
                    name='Predicción',
                    line=dict(color='#AAAAAA', width=2, dash='dot'),
                    marker=dict(symbol='x', size=8, color='#AAAAAA')
                ))
                st.success("Predicción agregada a la gráfica.")
            else:
                st.warning("El archivo de predicción debe tener columnas 'Mes' e 'Ingreso_Predicho'.")
        except Exception as e:
            st.error(f"Error al leer archivo de predicción: {e}")

    fig.update_layout(
        height=350,
        plot_bgcolor="#FFF",
        paper_bgcolor="#FFF",
        margin=dict(l=20, r=20, t=30, b=20),
        font=dict(family="sans-serif", color="#222"),
        xaxis=dict(title=None, showgrid=False, zeroline=False),
        yaxis=dict(title=None, showgrid=True, gridcolor="#F3EFFF"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Sube primero el archivo de datos reales para ver la gráfica.")

# Línea divisoria entre gráfica de tendencia y las dos gráficas de abajo
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# --- Segunda fila de visualizaciones ---
col5, col6 = st.columns(2)
with col5:
    st.markdown(f"<h5 style='color:{COLOR_PRIMARY}; margin-bottom:0.5rem;'>Ingresos por Región</h5>", unsafe_allow_html=True)
    regiones = ["Norte", "Sur", "Este", "Oeste"]
    valores = [12000, 15000, 11000, 10000]
    fig_reg = px.pie(
        names=regiones,
        values=valores,
        hole=0.5,
        color_discrete_sequence=["#2F1C6A", "#7B3FF2", "#B39DDB", "#7FC7FF"]
    )
    fig_reg.update_traces(
        textinfo='percent+label',
        textfont_size=15
    )
    fig_reg.update_layout(
        showlegend=True,
        legend=dict(orientation="v", y=0.5, x=1.1, font=dict(color="#222")),
        height=320,
        margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor="#FFF",
        paper_bgcolor="#FFF"
    )
    st.plotly_chart(fig_reg, use_container_width=True)
with col6:
    st.markdown(f"<h5 style='color:{COLOR_PRIMARY}; margin-bottom:0.5rem;'>Distribución por Categoría</h5>", unsafe_allow_html=True)
    categorias = ["Electrónica", "Ropa", "Hogar", "Alimentos"]
    margen = [25, 18, 15, 10]
    entregas = [80, 95, 90, 85]
    tamanio = [30, 20, 15, 10]
    df_burbujas = pd.DataFrame({
        "Categoría": categorias,
        "% Margen": margen,
        "% Entregas a tiempo": entregas,
        "Tamaño": tamanio
    })
    fig_bub = px.scatter(
        df_burbujas,
        x="% Entregas a tiempo",
        y="% Margen",
        size="Tamaño",
        color="Categoría",
        color_discrete_sequence=["#7B3FF2", "#B39DDB", "#2F1C6A", "#7FC7FF"],
        hover_name="Categoría"
    )
    fig_bub.update_layout(
        height=320,
        plot_bgcolor="#FFF",
        paper_bgcolor="#FFF",
        margin=dict(l=20, r=20, t=30, b=20),
        font=dict(family="sans-serif", color="#222"),
        legend=dict(orientation="v", y=0.5, x=1.1)
    )
    st.plotly_chart(fig_bub, use_container_width=True)
