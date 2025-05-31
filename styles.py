from config import COLOR_PRIMARY, COLOR_ACCENT, COLOR_NEGATIVE, COLOR_BG
import streamlit as st

def inject_styles():
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
            margin: 18px 0;
        }}
        .sidebar-divider {{
            border-bottom: 1.5px solid #ECEAF6;
            margin: 16px 0;
        }}
        </style>
    """, unsafe_allow_html=True)
