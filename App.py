import os
import sys
import calendar
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import random
import plotly.graph_objs as go

COLOR_PRIMARY = "#7B3FF2"
COLOR_ACCENT = "#23C16B"
COLOR_NEGATIVE = "#E14B64"
COLOR_BG = "#F6F6FB"

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
    if st.button("Cerrar"): st.rerun()

@st.dialog("Mejorar gestión de stock", width="large")
def dialog_mejorar_stock():
    st.write("### Unidades en inventario por producto")
    categorias2 = ["Producto X", "Producto Y", "Producto Z"]
    valores2 = [450, 320, 275]
    fig2 = px.bar(x=categorias2, y=valores2, labels={'x': 'Producto', 'y': 'Unidades en Stock'})
    fig2.update_layout(plot_bgcolor="#FFF", paper_bgcolor="#FFF", margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig2, use_container_width=True)
    st.write("Este gráfico muestra la cantidad de unidades en stock para cada producto."
            "Modifica 'categorias2' y 'valores2' en el código para actualizar los datos.")
    if st.button("Cerrar"): st.rerun()

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
    if st.button("Cerrar"): st.rerun()

st.set_page_config(page_title="Panel de Entregas", layout="wide", page_icon="🚚", initial_sidebar_state="expanded")

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
    .disabled-option {{
        color: #CCCCCC !important;
        background-color: #F5F5F5 !important;
    }}
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    try:
        st.image("logo_danu.png", width=180)
    except:
        st.markdown("DANU ANALÍTICA")

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Filtros")
    
    if 'df' in st.session_state:
        df_filtros = st.session_state['df'].copy()
        df_filtros['orden_compra_timestamp'] = pd.to_datetime(
            df_filtros['orden_compra_timestamp'], errors='coerce'
        )
        df_filtros = df_filtros.dropna(subset=['orden_compra_timestamp'])

        regiones = ["Todas las regiones"] + sorted(
            df_filtros['region'].dropna().unique().tolist()
        )
        categorias = ["Todas las categorías"] + sorted(
            df_filtros['categoria_simplificada'].dropna().unique().tolist()
        )
    else:
        regiones = ["Todas las regiones"]
        categorias = ["Todas las categorías"]

    periodo_options = ["Último año", "Últimos 6 meses (Próximamente)", "Último mes (Próximamente)"]
    periodo_habilitados = ["Último año"]  # solo esta opción está activa

    periodo_sel = st.selectbox("Periodo", periodo_options)

    if periodo_sel not in periodo_habilitados:
        st.warning("Esta opción estará disponible próximamente. Por favor selecciona 'Último año'.")
        st.stop()
    
    region_sel = st.selectbox("Región", regiones)
    
    categoria_sel = st.selectbox("Categoría", categorias)
    
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    rec_keys = ['rec1', 'rec2', 'rec3']
    rec_defaults = [st.session_state.get(k, False) for k in rec_keys]
    recomendaciones_activadas = sum(rec_defaults)
    progreso_recomendaciones = int((recomendaciones_activadas / 3) * 100)

    st.markdown(
        f"<h4 style='margin-bottom: 0.5rem; color:{COLOR_PRIMARY};'>Recomendaciones ({progreso_recomendaciones}%)</h4>",
        unsafe_allow_html=True
    )

    with st.sidebar:
        col_check1, col_text1 = st.columns([1, 10])
        rec1 = col_check1.checkbox("", value=rec_defaults[0], key='rec1')
        if col_text1.button("Optimizar rutas de entrega", key="btn_rec1"):
            dialog_optimizar_rutas()

        col_check2, col_text2 = st.columns([1, 10])
        rec2 = col_check2.checkbox("", value=rec_defaults[1], key='rec2')
        if col_text2.button("Mejorar gestión de stock", key="btn_rec2"):
            dialog_mejorar_stock()

        col_check3, col_text3 = st.columns([1, 10])
        rec3 = col_check3.checkbox("", value=rec_defaults[2], key='rec3')
        if col_text3.button("Ofertas segmentadas", key="btn_rec3"):
            dialog_ofertas_segmentadas()

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
            df.to_excel("df_DataBridgeConsulting.xlsx", index=False)
            st.session_state['df'] = df
            st.success("¡Archivo cargado exitosamente!")

            resultado = subprocess.run(
                [sys.executable, "modelo_v1.py"],
                 capture_output=True, text=True
            )

            # Mostrar logs del modelo
            if resultado.returncode == 0:
                st.success("✅ Modelo ejecutado correctamente")
                st.text("Salida del modelo:")
                st.code(resultado.stdout, language='bash')
            else:
                st.error("❌ Error al ejecutar el modelo")
                st.text("Detalles del error:")
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

def aplicar_filtros(df, periodo, region, categoria):
    """Aplica todos los filtros al dataframe"""
    df_filtrado = df.copy()
    df_filtrado['orden_compra_timestamp'] = pd.to_datetime(
        df_filtrado['orden_compra_timestamp'], errors='coerce'
    )
    df_filtrado = df_filtrado.dropna(subset=['orden_compra_timestamp'])

    if periodo == "Último año":
        fecha_limite = df_filtrado['orden_compra_timestamp'].max() - pd.DateOffset(years=1)
        df_filtrado = df_filtrado[df_filtrado['orden_compra_timestamp'] >= fecha_limite]

    if region != "Todas las regiones":
        df_filtrado = df_filtrado[df_filtrado['region'] == region]
    
    if categoria != "Todas las categorías":
        df_filtrado = df_filtrado[df_filtrado['categoria_simplificada'] == categoria]
    
    return df_filtrado

if 'df' in st.session_state:
    df_filtrado = aplicar_filtros(st.session_state['df'], periodo_sel, region_sel, categoria_sel)
    
    if len(df_filtrado) > 0:
        df_filtrado['año'] = df_filtrado['orden_compra_timestamp'].dt.year
        df_filtrado['mes'] = df_filtrado['orden_compra_timestamp'].dt.month
        df_filtrado['trimestre'] = df_filtrado['orden_compra_timestamp'].dt.quarter

        año_actual = df_filtrado['año'].max()
        mes_actual = df_filtrado['mes'].max()

        ingresos_totales = df_filtrado['precio_final'].sum()
        ingresos_año_actual = df_filtrado[df_filtrado['año'] == año_actual]['precio_final'].sum()
        ingresos_año_anterior = df_filtrado[df_filtrado['año'] == (año_actual - 1)]['precio_final'].sum()
        delta_ingresos = ((ingresos_año_actual - ingresos_año_anterior) / ingresos_año_anterior * 100) if ingresos_año_anterior > 0 else 0
        
        pedidos_totales = df_filtrado['order_id'].nunique()
        pedidos_año_actual = df_filtrado[df_filtrado['año'] == año_actual]['order_id'].nunique()
        pedidos_año_anterior = df_filtrado[df_filtrado['año'] == (año_actual - 1)]['order_id'].nunique()
        delta_pedidos = ((pedidos_año_actual - pedidos_año_anterior) / pedidos_año_anterior * 100) if pedidos_año_anterior > 0 else 0
        
        valor_promedio_actual = df_filtrado[df_filtrado['año'] == año_actual]['precio_final'].mean()
        valor_promedio_anterior = df_filtrado[df_filtrado['año'] == (año_actual - 1)]['precio_final'].mean()
        delta_valor = ((valor_promedio_actual - valor_promedio_anterior) / valor_promedio_anterior * 100) if valor_promedio_anterior > 0 else 0
        
        flete_promedio_actual = df_filtrado[df_filtrado['año'] == año_actual]['costo_de_flete'].mean()
        flete_promedio_anterior = df_filtrado[df_filtrado['año'] == (año_actual - 1)]['costo_de_flete'].mean()
        delta_flete = ((flete_promedio_actual - flete_promedio_anterior) / flete_promedio_anterior * 100) if flete_promedio_anterior > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            color_ingresos = "kpi-delta-pos" if delta_ingresos >= 0 else "kpi-delta-neg"
            st.markdown(
                f"""<div class="kpi-card">
                        <div class="kpi-label">Ingresos Totales</div>
                        <div class="kpi-value">${ingresos_totales:,.0f}</div>
                        <div class="{color_ingresos}">{delta_ingresos:.1f}% {'↑' if delta_ingresos >= 0 else '↓'}</div>
                        <div class="kpi-subtext">vs año anterior</div>
                    </div>""", unsafe_allow_html=True)
        
        with col2:
            color_pedidos = "kpi-delta-pos" if delta_pedidos >= 0 else "kpi-delta-neg"
            st.markdown(
                f"""<div class="kpi-card">
                        <div class="kpi-label">Pedidos Totales</div>
                        <div class="kpi-value">{pedidos_totales:,}</div>
                        <div class="{color_pedidos}">{delta_pedidos:.1f}% {'↑' if delta_pedidos >= 0 else '↓'}</div>
                        <div class="kpi-subtext">vs año anterior</div>
                    </div>""", unsafe_allow_html=True)
        
        with col3:
            color_valor = "kpi-delta-pos" if delta_valor >= 0 else "kpi-delta-neg"
            st.markdown(
                f"""<div class="kpi-card">
                        <div class="kpi-label">Valor Promedio</div>
                        <div class="kpi-value">${valor_promedio_actual:,.2f}</div>
                        <div class="{color_valor}">{delta_valor:.1f}% {'↑' if delta_valor >= 0 else '↓'}</div>
                        <div class="kpi-subtext">vs año anterior</div>
                    </div>""", unsafe_allow_html=True)
        
        with col4:
            color_flete = "kpi-delta-pos" if delta_flete >= 0 else "kpi-delta-neg"
            st.markdown(
                f"""<div class="kpi-card">
                        <div class="kpi-label">Flete Promedio</div>
                        <div class="kpi-value">${flete_promedio_actual:,.2f}</div>
                        <div class="{color_flete}">{delta_flete:.1f}% {'↑' if delta_flete >= 0 else '↓'}</div>
                        <div class="kpi-subtext">vs año anterior</div>
                    </div>""", unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Gráfico de Tendencia Mensual
        st.markdown(
            f"<h4 style='color:{COLOR_PRIMARY}; margin-bottom:0.5rem;'>"
            "Tendencia de Ingresos Mensuales</h4>",
            unsafe_allow_html=True
        )
        df_filtrado['Año'] = df_filtrado['orden_compra_timestamp'].dt.year
        df_filtrado['Mes'] = df_filtrado['orden_compra_timestamp'].dt.month
        df_filtrado['Dia'] = df_filtrado['orden_compra_timestamp'].dt.day

        dias_por_mes = df_filtrado.groupby(['Año', 'Mes'])['Dia'].nunique().reset_index()
        dias_por_mes.rename(columns={'Dia': 'DiasRegistrados'}, inplace=True)

        MIN_DIAS_MES = 28
        meses_validos = dias_por_mes[dias_por_mes['DiasRegistrados'] >= MIN_DIAS_MES][['Año', 'Mes']]

        df_mensual = df_filtrado.groupby(['Año', 'Mes'])['precio_final'].sum().reset_index()
        df_mensual = pd.merge(df_mensual, meses_validos, on=['Año', 'Mes'], how='inner')
        df_mensual['Tipo'] = "real"

        if not df_pred.empty:
            df_pred = df_pred[["Año", "Mes", "precio_final", "Tipo"]]
            df_total = pd.concat([df_mensual, df_pred], ignore_index=True)
        else:
            df_total = df_mensual

        df_total = df_total.sort_values(["Año", "Mes"]).reset_index(drop=True)
        df_total["MesNombre"] = df_total["Mes"].apply(lambda x: calendar.month_name[x])
        df_total["MesIndex"] = (
            df_total["Año"].astype(str) + "-" + df_total["Mes"].astype(str).str.zfill(2)
        )

        fig_tendencia = go.Figure()
        df_real = df_total[df_total["Tipo"] == "real"]
        fig_tendencia.add_trace(go.Scatter(
            x=df_real["MesIndex"],
            y=df_real["precio_final"],
            mode='lines+markers',
            name='Datos reales',
            line=dict(color="#3B82F6", width=3),
            marker=dict(size=8, color="#3B82F6")
        ))

        df_pred_plot = df_total[df_total["Tipo"] == "pred"]
        if not df_pred_plot.empty:
            # Unir último punto real + puntos predichos (solo para trazo de línea)
            df_pred_union = df_pred_plot.copy()
            df_pred_union.loc[df_pred_union.index[0], "precio_final"] = None  # Para no duplicar visualmente el último punto

            fig_tendencia.add_trace(go.Scatter(
                x=[df_real["MesIndex"].iloc[-1]] + df_pred_plot["MesIndex"].tolist(),
                y=[df_real["precio_final"].iloc[-1]] + df_pred_plot["precio_final"].tolist(),
                mode='lines+markers',
                name='Predicción mes siguiente',
                line=dict(color='#AAAAAA', width=2, dash='dot'),
                marker=dict(size=[0] + [8] * len(df_pred_plot), color=['#7B3FF2'] + ['#AAAAAA'] * (len(df_pred_plot)))
            ))

            # Solo mostrar anotación en el primer punto predicho
            pred_mes = df_pred_plot.iloc[0]
            fig_tendencia.add_annotation(
                x=pred_mes["MesIndex"],
                y=pred_mes["precio_final"],
                text="Mes Predicho",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40,
                font=dict(color="#AAAAAA", size=12, family="sans-serif"),
                bgcolor="#FFF",
                bordercolor="#AAAAAA"
            )


            pred_mes = df_pred_plot.iloc[0]
            fig_tendencia.add_annotation(
                x=pred_mes["MesIndex"],
                y=pred_mes["precio_final"],
                text="Mes Predicho",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40,
                font=dict(color="#AAAAAA", size=12, family="sans-serif"),
                bgcolor="#FFF",
                bordercolor="#AAAAAA"
            )

        fig_tendencia.update_layout(
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
                tickvals=df_total["MesIndex"],
                ticktext=[f"{row['MesNombre']} {row['Año']}" for _, row in df_total.iterrows()]
            ),
            yaxis=dict(title=None, showgrid=True, gridcolor="#F3EFFF"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig_tendencia, use_container_width=True)
        
        col5, col6 = st.columns(2)
        
        with col5:
            st.markdown(f"<h5 style='color:{COLOR_PRIMARY}; margin-bottom:0.5rem;'>Ingresos por Región</h5>", unsafe_allow_html=True)
            ingresos_por_region = df_filtrado.groupby('region')['precio_final'].sum().reset_index()
            
            fig_reg = px.pie(
                ingresos_por_region,
                names='region',
                values='precio_final',
                hole=0.5,
                color_discrete_sequence=["#2F1C6A", "#7B3FF2", "#B39DDB", "#7FC7FF"]
            )
            fig_reg.update_traces(textinfo='label+percent', textfont_size=14)
            fig_reg.update_layout(
                showlegend=True,
                legend=dict(orientation="v", y=0.5, x=-0.2, font=dict(color="#222")),
                height=320,
                margin=dict(l=40, r=20, t=20, b=20),
                plot_bgcolor="#FFF",
                paper_bgcolor="#FFF"
            )
            st.plotly_chart(fig_reg, use_container_width=True)
        
        with col6:
            st.markdown(f"<h5 style='color:{COLOR_PRIMARY}; margin-bottom:0.5rem;'>Costos Operativos</h5>", unsafe_allow_html=True)

            categorias = df_filtrado['categoria_simplificada'].unique()

            # Genera colores aleatorios en formato hexadecimal
            colores_random = [
                "#b39ddb",  # lavanda
                "#c5cae9",  # azul pastel
                "#9fa8da",  # azul violeta
                "#ce93d8",  # rosa lavanda
                "#90caf9",  # celeste pastel
                "#f48fb1",  # rosa claro
                "#a5d6a7",  # verde suave
                "#9575cd",  # violeta
                "#81d4fa",  # celeste
                "#cfd8dc",  # gris pastel
                "#e1bee7",  # rosa lavanda
                "#b3e5fc"   # azul claro pastel
            ][:len(categorias)]


            bubble_data = df_filtrado.groupby('categoria_simplificada').agg({
                'costo_de_flete': 'mean',
                'precio_final': 'mean',
                'peso_volumetrico_kg': 'mean',
                'order_id': 'count'
            }).reset_index()

            bubble_data.columns = ['Categoría', 'Costo de Flete', 'Precio Final', 'Peso Volumétrico (kg)', 'Tamaño']

            fig_bub = go.Figure()

            size_values = bubble_data['Peso Volumétrico (kg)'] * 15
            sizeref = 2. * max(size_values) / (80 ** 2)

            for idx, row in bubble_data.iterrows():
                fig_bub.add_trace(go.Scatter(
                    x=[row['Costo de Flete']],
                    y=[row['Precio Final']],
                    mode='markers',
                    marker=dict(
                        size=row['Peso Volumétrico (kg)'] * 15,
                        color=colores_random[idx],
                        sizemode='area',
                        sizeref=sizeref,
                        sizemin=8,
                        opacity=0.5,  # <- transparencia suave
                        line=dict(width=1, color="rgba(0,0,0,0.1)")  # borde sutil
                    ),

                    name=row['Categoría'],
                    hovertemplate=f"<b>{row['Categoría']}</b><br>Costo de Flete: {row['Costo de Flete']:.2f}<br>Precio Final: {row['Precio Final']:.2f}<br>Peso Volumétrico (kg): {row['Peso Volumétrico (kg)']:.2f}<extra></extra>",
                    visible=True
                ))

            visible_legendonly = ['legendonly'] * len(categorias)
            visible_true = [True] * len(categorias)

            fig_bub.update_layout(
                height=320,
                plot_bgcolor="#FFF",
                paper_bgcolor="#FFF",
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="v", y=0.5, x=1.1),
                updatemenus=[
                    dict(
                        type="buttons",
                        direction="left",
                        buttons=[
                            dict(
                                args=[{"visible": visible_legendonly}],
                                label="❌ Quitar todas",
                                method="restyle"
                            ),
                            dict(
                                args=[{"visible": visible_true}],
                                label="✅ Mostrar todas",
                                method="restyle"
                            )
                        ],
                        pad={"r": 10, "t": 10},
                        showactive=False,
                        x=0,
                        xanchor="left",
                        y=1.15,
                        yanchor="top"
                    )
                ]
            )

            st.plotly_chart(fig_bub, use_container_width=True)

else:
    st.info("Sube tu base de datos para ver las métricas filtradas")

    col1, col2, col3, col4 = st.columns(4)
    for col, label in zip(
        [col1, col2, col3, col4],
        ["Ingresos Totales", "Pedidos Totales", "Valor Promedio", "Flete Promedio"]
    ):
        with col:
            st.markdown(
                f"""<div class="kpi-card">
                        <div class="kpi-label">{label}</div>
                        <div class="kpi-value" style="color:#DDD;">---</div>
                        <div class="kpi-subtext" style="color:#DDD;">Sin datos</div>
                    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown(f"<h4 style='color:{COLOR_PRIMARY}; margin-bottom:0.5rem;'>Tendencia de Ingresos Mensuales</h4>", unsafe_allow_html=True)
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
        st.markdown(f"<h5 style='color:{COLOR_PRIMARY}; margin-bottom:0.5rem;'>Ingresos por Región</h5>", unsafe_allow_html=True)
        fig_placeholder2 = px.pie(values=[], names=[])
        fig_placeholder2.update_layout(
            annotations=[dict(text="Sin datos", x=0.5, y=0.5, font_size=20, showarrow=False)],
            showlegend=False,
            height=320
        )
        st.plotly_chart(fig_placeholder2, use_container_width=True)
    with col6:
        st.markdown(f"<h5 style='color:{COLOR_PRIMARY}; margin-bottom:0.5rem;'>Costos Operativos</h5>", unsafe_allow_html=True)
        fig_placeholder3 = px.scatter(pd.DataFrame({'x': [], 'y': []}), x='x', y='y')
        fig_placeholder3.update_layout(
            annotations=[dict(text="Sin datos", x=0.5, y=0.5, font_size=20, showarrow=False)],
            showlegend=False,
            height=320
        )
        st.plotly_chart(fig_placeholder3, use_container_width=True)