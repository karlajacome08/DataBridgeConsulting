import streamlit as st
from config import COLOR_PRIMARY

def render_sidebar():
    with st.sidebar:
        st.markdown(f"<h2 style='color:{COLOR_PRIMARY}; margin-bottom:0;'>Panel de Entregas</h2>", unsafe_allow_html=True)
        st.caption("Métricas y análisis")
        st.markdown("### Filtros")
        st.selectbox("Periodo", ["Último mes"])
        st.selectbox("Región", ["Todas las regiones"])
        st.selectbox("Categoría", ["Todas las categorías"])
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"### <span style='color:{COLOR_PRIMARY};'>Recomendaciones</span>", unsafe_allow_html=True)
        st.checkbox("Optimizar rutas de entrega\nReducir tiempos en zona Este")
        st.checkbox("Aumentar capacidad logística\nAlmacenamiento en Barcelona")
        st.checkbox("Promocionar Electrónica\nMayor margen de beneficio")
        st.checkbox("Revisar proveedores\nReducir costos de envío")
        st.checkbox("Implementar seguimiento GPS\nPara entregas en tiempo real")
