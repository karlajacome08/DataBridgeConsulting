import streamlit as st
import plotly.express as px

def render_region_chart():
    col5, _ = st.columns([1, 1])
    with col5:
        st.markdown(f"<h5 style='color:#7B3FF2; margin-bottom:0.5rem;'>Ingresos por Región</h5>", unsafe_allow_html=True)

        regiones = ["Norte", "Sur", "Este", "Oeste"]
        valores = [12000, 15000, 11000, 10000]

        fig = px.pie(
            names=regiones,
            values=valores,
            hole=0.5,
            color_discrete_sequence=["#2F1C6A", "#7B3FF2", "#B39DDB", "#7FC7FF"]
        )
        fig.update_traces(textinfo='percent+label', textfont_size=15)
        fig.update_layout(
            showlegend=True,
            legend=dict(orientation="v", y=0.5, x=1, font=dict(color="#222")),
            height=320,
            margin=dict(l=20, r=10, t=30, b=20),
            plot_bgcolor="#FFF",
            paper_bgcolor="#FFF"
        )
        st.plotly_chart(fig, use_container_width=True)
