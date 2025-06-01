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

    if 'df' in st.session_state:
        df_sidebar = st.session_state['df'].copy()
        df_sidebar['orden_pago_aprobado'] = pd.to_datetime(df_sidebar['orden_pago_aprobado'], errors='coerce')
        df_sidebar = df_sidebar.dropna(subset=['orden_pago_aprobado'])

        # Crear columna Año-Mes para filtrar
        df_sidebar['Periodo'] = df_sidebar['orden_pago_aprobado'].dt.to_period("M").astype(str)
        periodos = sorted(df_sidebar['Periodo'].unique(), reverse=True)
        regiones = sorted(df_sidebar['region'].dropna().unique())
        categorias = sorted(df_sidebar['categoria_simplificada'].dropna().unique())
    else:
        periodos = []
        regiones = []
        categorias = []

    # Filtros dinámicos
    st.markdown("### Filtros")
    periodo_sel = st.selectbox("Periodo", ["Todos los periodos"] + periodos)
    region_sel = st.selectbox("Región", ["Todas las regiones"] + regiones)
    categoria_sel = st.selectbox("Categoría", ["Todas las categorías"] + categorias)
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # Recomendaciones (las capturamos en variables rec1, rec2, rec3)
    st.markdown(f"### <span style='color:{COLOR_PRIMARY};'>Recomendaciones</span>", unsafe_allow_html=True)
    rec1 = st.checkbox("Recomendación 1")
    rec2 = st.checkbox("Recomendación 2")
    rec3 = st.checkbox("Recomendación 3")
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # Botón para subir base de datos (AL FINAL)
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

# --- Procesamiento y KPIs dinámicas ---
if 'df' in st.session_state:
    df = st.session_state['df'].copy()
    df['fecha'] = pd.to_datetime(df['orden_pago_aprobado'], errors='coerce')
    df = df.dropna(subset=['fecha'])
    df['trimestre'] = df['fecha'].dt.to_period('Q')
    df['año'] = df['fecha'].dt.year
    df['mes'] = df['fecha'].dt.month

    # Ingresos Totales
    ingresos_totales = df['precio_promedio_por_orden'].sum()
    mes_actual = df['mes'].max()
    anio_actual = df['año'].max()
    mes_anterior = mes_actual - 1 if mes_actual > 1 else 12
    ingresos_mes_actual = df[(df['mes'] == mes_actual) & (df['año'] == anio_actual)]['precio_promedio_por_orden'].sum()
    ingresos_mes_anterior = df[(df['mes'] == mes_anterior) & (df['año'] == anio_actual)]['precio_promedio_por_orden'].sum()
    delta_ingresos = ((ingresos_mes_actual - ingresos_mes_anterior) / ingresos_mes_anterior) * 100 if ingresos_mes_anterior != 0 else 0

    # Pedidos Totales
    pedidos_totales = df['order_id'].nunique()
    pedidos_mes_actual = df[(df['mes'] == mes_actual) & (df['año'] == anio_actual)]['order_id'].nunique()
    pedidos_mes_anterior = df[(df['mes'] == mes_anterior) & (df['año'] == anio_actual)]['order_id'].nunique()
    delta_pedidos = ((pedidos_mes_actual - pedidos_mes_anterior) / pedidos_mes_anterior) * 100 if pedidos_mes_anterior != 0 else 0

    # Valor Promedio
    valor_trimestre_actual = df[df['trimestre'] == df['trimestre'].max()]['precio_promedio_por_orden'].mean()
    valor_trimestre_anterior = df[df['trimestre'] == df['trimestre'].max() - 1]['precio_promedio_por_orden'].mean()
    delta_valor = ((valor_trimestre_actual - valor_trimestre_anterior) / valor_trimestre_anterior) * 100 if valor_trimestre_anterior != 0 else 0

    # Flete Promedio
    flete_actual = df[df['año'] == anio_actual]['costo_de_flete'].mean()
    flete_anterior = df[df['año'] == anio_actual - 1]['costo_de_flete'].mean()
    delta_flete = ((flete_actual - flete_anterior) / flete_anterior) * 100 if flete_anterior != 0 else 0

    # Renderizar tarjetas
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""<div class="kpi-card">
                    <div class="kpi-label">Ingresos Totales</div>
                    <div class="kpi-value">${ingresos_totales:,.0f}</div>
                    <div class="kpi-delta-pos">{delta_ingresos:+.1f}% ↑</div>
                    <div class="kpi-subtext">vs mes anterior</div>
                </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(
            f"""<div class="kpi-card">
                    <div class="kpi-label">Pedidos Totales</div>
                    <div class="kpi-value">{pedidos_totales:,}</div>
                    <div class="kpi-delta-pos">{delta_pedidos:+.1f}% ↑</div>
                    <div class="kpi-subtext">vs mes anterior</div>
                </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(
            f"""<div class="kpi-card">
                    <div class="kpi-label">Valor Promedio</div>
                    <div class="kpi-value">${valor_trimestre_actual:,.2f}</div>
                    <div class="kpi-delta-neg">{delta_valor:+.1f}% ↓</div>
                    <div class="kpi-subtext">vs trimestre anterior</div>
                </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(
            f"""<div class="kpi-card">
                    <div class="kpi-label">Flete Promedio</div>
                    <div class="kpi-value">${flete_actual:,.2f}</div>
                    <div class="kpi-delta-pos">{delta_flete:+.1f}% ↑</div>
                    <div class="kpi-subtext">vs año anterior</div>
                </div>""", unsafe_allow_html=True)
else:
    st.info("Sube primero la base de datos para ver las métricas.")


# --- Gráfica de tendencia de ingresos con predicción CONTINUA ---
if 'df' in st.session_state:
    df_real = st.session_state['df'].copy()
    
    # Convertir a fecha y extraer mes
    df_real['orden_pago_aprobado'] = pd.to_datetime(df_real['orden_pago_aprobado'], errors='coerce')
    df_real = df_real.dropna(subset=['orden_pago_aprobado', 'precio_final'])
    df_real['Mes'] = df_real['orden_pago_aprobado'].dt.month

    # Agrupar ingresos reales por mes y ordenar
    ingresos_reales = (
        df_real
        .groupby('Mes')['precio_final']
        .sum()
        .reset_index()
        .sort_values(by='Mes')
    )
    ingresos_reales['MesIndex'] = ingresos_reales['Mes']  
    ingresos_reales['MesNombre'] = ingresos_reales['Mes'].apply(lambda x: calendar.month_name[x])

    #Siguientes 3 meses
    ultimo_mes_real = ingresos_reales['Mes'].max()
    futuros_numeros = [ultimo_mes_real + i for i in range(1, 4)]              
    meses_futuros = [((ultimo_mes_real + i - 1) % 12) + 1 for i in range(1, 4)]  
    meses_futuros_nombres = [calendar.month_name[m] for m in meses_futuros]     

    #Valores (dummies) de prediccion
    pred1 = 489000
    pred2 = 570000   
    pred3 = 520600   
    valores_dummie = [pred1, pred2, pred3]

    df_dummie = pd.DataFrame({
        'MesIndex': futuros_numeros,         
        'Mes': meses_futuros,                
        'MesNombre': meses_futuros_nombres,  
        'precio_final': valores_dummie       
    })

    # Figura con datos reales
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ingresos_reales['MesIndex'],
        y=ingresos_reales['precio_final'],
        mode='lines+markers',
        name='Datos reales',
        line=dict(color=COLOR_PRIMARY, width=3),
        marker=dict(size=8, color=COLOR_PRIMARY)
    ))

    #Figura predicha (ultimo real + datos predichos)
    ultimo_indice = ingresos_reales['MesIndex'].iloc[-1]      
    ultimo_valor = ingresos_reales['precio_final'].iloc[-1]   

    # Coordenadas de cada segmento:
    seg_x = [
        [ultimo_indice, df_dummie['MesIndex'].iloc[0]],        # Segmento 1
        [df_dummie['MesIndex'].iloc[0], df_dummie['MesIndex'].iloc[1]],  # Segmento 2
        [df_dummie['MesIndex'].iloc[1], df_dummie['MesIndex'].iloc[2]]   # Segmento 3
    ]
    seg_y = [
        [ultimo_valor, df_dummie['precio_final'].iloc[0]],
        [df_dummie['precio_final'].iloc[0], df_dummie['precio_final'].iloc[1]],
        [df_dummie['precio_final'].iloc[1], df_dummie['precio_final'].iloc[2]]
    ]

    # Predicciones en gris
    for i in range(3):
        fig.add_trace(go.Scatter(
            x=seg_x[i],
            y=seg_y[i],
            mode='lines+markers',
            name=f"Segmento {i+1} (gris)",
            line=dict(color="#DDDDDD", width=2, dash='dot'),
            marker=dict(
                size=[0, 8],            
                color=['rgba(0,0,0,0)', "#DDDDDD"]
            ),
            showlegend=False
        ))

    #Marcar las predicciones en el grafico segun las recomendaciones
    n_checked = sum([rec1, rec2, rec3])  # 0, 1, 2 o 3

   
    for i in range(n_checked):
        fig.add_trace(go.Scatter(
            x=seg_x[i],
            y=seg_y[i],
            mode='lines+markers',
            name=f"Segmento {i+1} (activo)",
            line=dict(color=COLOR_ACCENT, width=2, dash='solid'),
            marker=dict(
                size=[0, 8],  
                color=['rgba(0,0,0,0)', COLOR_ACCENT]
            ),
            showlegend=False
        ))

    #Subir archivo para predicciones
    pred_file = st.file_uploader("Sube archivo de predicción mensual", type=["csv", "xlsx"])
    if pred_file:
        try:
            ext = pred_file.name.split(".")[-1]
            df_pred = pd.read_csv(pred_file) if ext == "csv" else pd.read_excel(pred_file)

            if 'Mes' in df_pred.columns and 'Ingreso_Predicho' in df_pred.columns:
                df_pred['Mes'] = df_pred['Mes'].astype(int)
                df_pred = df_pred.sort_values(by='Mes')
                df_pred['MesIndex'] = df_pred['Mes']
                df_pred['MesNombre'] = df_pred['Mes'].apply(lambda x: calendar.month_name[int(x)])
                fig.add_trace(go.Scatter(
                    x=df_pred['MesIndex'],
                    y=df_pred['Ingreso_Predicho'],
                    mode='lines+markers',
                    name='Predicción (archivo)',
                    line=dict(color='#666666', width=2, dash='dot'),
                    marker=dict(symbol='x', size=8, color='#666666')
                ))
                st.success("Predicción cargada desde archivo y agregada a la gráfica.")
            else:
                st.warning("El archivo de predicción debe tener columnas 'Mes' e 'Ingreso_Predicho'.")
        except Exception as e:
            st.error(f"Error al leer archivo de predicción: {e}")

    #Mostrar meses siguientes
    idx_reales = ingresos_reales['MesIndex'].tolist()
    idx_dummie = df_dummie['MesIndex'].tolist()
    todos_los_indices = idx_reales + idx_dummie

    labels_reales = ingresos_reales['MesNombre'].tolist()
    labels_dummie = df_dummie['MesNombre'].tolist()
    todas_las_etiquetas = labels_reales + labels_dummie

    fig.update_layout(
        height=350,
        plot_bgcolor="#FFF",
        paper_bgcolor="#FFF",
        margin=dict(l=20, r=20, t=30, b=20),
        font=dict(family="sans-serif", color="#222"),
        xaxis=dict(
            title=None,
            showgrid=False,
            zeroline=False,
            tickmode='array',
            tickvals=todos_los_indices,
            ticktext=todas_las_etiquetas
        ),
        yaxis=dict(title=None, showgrid=True, gridcolor="#F3EFFF"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Sube primero el archivo de datos reales para ver la gráfica.")

    


#Linea divisora
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