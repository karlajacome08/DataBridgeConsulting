import streamlit as st
import pandas as pd
import plotly.express as px
from config import COLOR_PRIMARY

def render_trend_chart():
    st.markdown(f"<h4 style='color:{COLOR_PRIMARY}; margin-bottom:0.5rem;'>Tendencia de Ingresos Mensuales</h4>", unsafe_allow_html=True)

    meses = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
             "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
    ingresos = [30000, 32000, 39000, 38000, 44000, 50000,
                48000, 51000, 52000, 54000, 56000, 58000]
    df = pd.DataFrame({"Mes": meses, "Ingresos": ingresos})

    fig = px.line(df, x="Mes", y="Ingresos", markers=True)
    fig.update_traces(
        line=dict(color=COLOR_PRIMARY, width=3),
        marker=dict(size=8, color=COLOR_PRIMARY),
        hovertemplate="Mes: %{x}<br>Ingresos: %{y}€"
    )
    fig.update_layout(
        height=350,
        plot_bgcolor="#FFF",
        paper_bgcolor="#FFF",
        margin=dict(l=20, r=20, t=30, b=20),
        font=dict(family="sans-serif", color="#222"),
        xaxis=dict(title=None, showgrid=False),
        yaxis=dict(title=None, showgrid=True, gridcolor="#F3EFFF"),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
