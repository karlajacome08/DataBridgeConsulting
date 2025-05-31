# main.py
import streamlit as st
from styles import inject_styles
from config import COLOR_PRIMARY
from components import sidebar, kpis, trend_chart, region_chart, category_chart

st.set_page_config(page_title="Panel de Entregas", layout="wide", page_icon="🚚", initial_sidebar_state="expanded")

inject_styles()
sidebar.render_sidebar()

# Logo
st.image("resources/danu_logo.png", width=180)

kpis.render_kpis()
trend_chart.render_trend_chart()
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
region_chart.render_region_chart()
category_chart.render_category_chart()
