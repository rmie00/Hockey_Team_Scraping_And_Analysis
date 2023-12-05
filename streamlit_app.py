import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import sqlite3

st.set_page_config(page_title='My Analysis',
                   layout= 'wide',
                   page_icon=':bar-chart:')

st.title('Hockey Team Analysis for the time period between 1990 and 2011')
st.markdown('<style>.div.block-cotainer{padding-top:1rem;}</style>', unsafe_allow_html=True)

@st.cache_data
def load_sqlite(path: str, table: str):
    dat = sqlite3.connect(path)
    data = pd.read_sql_query(f'SELECT * FROM {table}',
                             dat)
    return data

df = load_sqlite('HockeyDataBase.sqlite', 'Performance')

with st.expander('Original Data'):
    st.dataframe(df,
                 column_config={
                     'Year_Played': st.column_config.NumberColumn(format = '%d')
                 })


