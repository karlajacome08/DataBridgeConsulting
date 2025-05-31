import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go  # <-- ¡Esta línea falta!


# Paleta de colores según el mockup
COLOR_PRIMARY = "#7B3FF2"  # Morado principal
COLOR_ACCENT = "#23C16B"   # Verde para delta positivo
COLOR_NEGATIVE = "#E14B64" # Rojo para delta negativo
COLOR_BG = "#F6F6FB"       # Fondo general

# Configuración de la página
st.set_page_config(page_title="Panel de Entregas", layout="wide", page_icon="🚚", initial_sidebar_state="expanded")

# --- CSS personalizado para el dashboard y líneas divisorias ---
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

# --- Sidebar ---
with st.sidebar:
    st.markdown(f"<h2 style='color:{COLOR_PRIMARY}; margin-bottom:0;'>Panel de Entregas</h2>", unsafe_allow_html=True)
    st.caption("Métricas y análisis")
    st.markdown("### Filtros")
    st.selectbox("Periodo", ["Último mes"])
    st.selectbox("Región", ["Todas las regiones"])
    st.selectbox("Categoría", ["Todas las categorías"])
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)  # Línea entre filtros y recomendaciones
    st.markdown(f"### <span style='color:{COLOR_PRIMARY};'>Recomendaciones</span>", unsafe_allow_html=True)
    st.checkbox("Optimizar rutas de entrega\nReducir tiempos en zona Este")
    st.checkbox("Aumentar capacidad logística\nAlmacenamiento en Barcelona")
    st.checkbox("Promocionar Electrónica\nMayor margen de beneficio")
    st.checkbox("Revisar proveedores\nReducir costos de envío")
    st.checkbox("Implementar seguimiento GPS\nPara entregas en tiempo real")


# --- Logo de Danu Analítica ---
# Si tienes el archivo local
st.image("logo_danu.png", width=180)
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
st.markdown(f"<h4 style='color:{COLOR_PRIMARY}; margin-bottom:0.5rem;'>Tendencia de Ingresos Mensuales</h4>", unsafe_allow_html=True)

# Datos reales (enero a agosto) y predicción (octubre a diciembre)
meses = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
ingresos_reales = [30000, 32000, 39000, 38000, 44000, 50000, 48000, 51000]  # Hasta agosto
ingresos_pred = [52000, 54000, 56000]  # Octubre, Noviembre, Diciembre

fig = go.Figure()

# Línea de datos reales (enero a agosto)
fig.add_trace(go.Scatter(
    x=meses[:8],  # Enero a Agosto
    y=ingresos_reales,
    mode='lines+markers',
    name='Datos reales',
    line=dict(color=COLOR_PRIMARY, width=3),
    marker=dict(size=8, color=COLOR_PRIMARY)
))

# Línea de predicción (gris punteada), conectando agosto (último real) con octubre, noviembre, diciembre
fig.add_trace(go.Scatter(
    x=["Agosto", "Octubre", "Noviembre", "Diciembre"],  # Conexión directa: agosto + 3 meses de predicción
    y=[ingresos_reales[-1]] + ingresos_pred,            # Último real + predicción
    mode='lines+markers',
    name='Predicción',
    line=dict(color='#AAAAAA', width=2, dash='dot'),
    marker=dict(symbol='x', size=8, color='#AAAAAA')
))

fig.update_layout(
    height=350,
    plot_bgcolor="#FFF",
    paper_bgcolor="#FFF",
    margin=dict(l=20, r=20, t=30, b=20),
    font=dict(family="sans-serif", color="#222"),
    xaxis=dict(title=None, showgrid=False, zeroline=False),
    yaxis=dict(title=None, showgrid=True, gridcolor="#F3EFFF"),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

st.plotly_chart(fig, use_container_width=True)



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