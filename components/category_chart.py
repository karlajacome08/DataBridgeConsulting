import streamlit as st
import pandas as pd
import plotly.express as px

def render_category_chart():
    _, col6 = st.columns([1, 1])
    with col6:
        st.markdown(f"<h5 style='color:#7B3FF2; margin-bottom:0.5rem;'>Distribución por Categoría</h5>", unsafe_allow_html=True)

        categorias = ["Electrónica", "Ropa", "Hogar", "Alimentos"]
        margen = [25, 18, 15, 10]
        entregas = [80, 95, 90, 85]
        tamanio = [30, 20, 15, 10]

        df = pd.DataFrame({
            "Categoría": categorias,
            "% Margen": margen,
            "% Entregas a tiempo": entregas,
            "Tamaño": tamanio
        })

        fig = px.scatter(
            df,
            x="% Entregas a tiempo",
            y="% Margen",
            size="Tamaño",
            color="Categoría",
            color_discrete_sequence=["#7B3FF2", "#B39DDB", "#2F1C6A", "#7FC7FF"],
            hover_name="Categoría"
        )
        fig.update_layout(
            height=320,
            plot_bgcolor="#FFF",
            paper_bgcolor="#FFF",
            margin=dict(l=20, r=20, t=30, b=20),
            font=dict(family="sans-serif", color="#222"),
            legend=dict(orientation="v", y=0.5, x=1.1)
        )
        st.plotly_chart(fig, use_container_width=True)
