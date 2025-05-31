import streamlit as st

def render_kpis():
    col1, col2, col3, col4 = st.columns(4)
    kpis = [
        ("Ingresos Totales", "€48,250", "+8.3% ↑", "vs mes anterior", "pos"),
        ("Pedidos Totales", "248", "+12% ↑", "vs mes anterior", "pos"),
        ("Valor Promedio", "€194.56", "-3.2% ↓", "vs trimestre anterior", "neg"),
        ("Flete Promedio", "€12.75", "+1.5% ↑", "vs año anterior", "pos")
    ]
    for col, (label, value, delta, subtext, trend) in zip([col1, col2, col3, col4], kpis):
        delta_class = "kpi-delta-pos" if trend == "pos" else "kpi-delta-neg"
        col.markdown(
            f"""<div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="{delta_class}">{delta}</div>
                    <div class="kpi-subtext">{subtext}</div>
                </div>""", unsafe_allow_html=True
        )
