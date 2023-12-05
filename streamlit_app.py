import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title='My Analysis',
                   layout= 'wide',
                   page_icon=':bar-chart:')

st.title('Hockey Team Analysis for the time period between 1990 and 2011')
st.markdown('<style>.div.block-cotainer{padding-top:1rem;}</style>', unsafe_allow_html=True)

@st.cache_data
def load_sqlite(path: str, table: str, index_col: str):
    dat = sqlite3.connect(path)
    data = pd.read_sql_query(f'SELECT * FROM {table}',
                             dat,
                             index_col = [f'{index_col}'])
    return data

# Correlation Map
@st.cache_data
def plot_heatmmap():
    correlation = df.corr()
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    palette = sns.color_palette('coolwarm', 12)
    fig, axe = plt.subplots(figsize=(14, 6))
    sns.heatmap(data=correlation,
                mask=mask,
                square=True,
                cmap=palette,
                linewidth=0.5,
                annot=True,
                fmt='.2f')
    axe.set_title('Correlation Map')
    st.pyplot()

# Original Data Frame
df = load_sqlite('HockeyDataBase.sqlite',
                 'Performance',
                 'Team_Name')

df['OT_Losses'] = pd.to_numeric(df['OT_Losses'])
df['OT_Losses'] = (df['OT_Losses'].fillna(0)).astype(int)
df['Win_Perc'] = round((df['Wins'] / (df['Wins'] + df['Losses']))*100, 2)
df['+-'] = df['Goals_For'] - df['Goals_Against']
df = df[['Year_Played', 'Wins', 'Losses','OT_Losses', 'Win_Perc', 'Goals_For', 'Goals_Against', '+-']]
# DataFrame of Top Teams
df_top = df.reset_index()
df_top = df_top.sort_values(by=['Wins', '+-'], ascending = False)
df_top = df_top.groupby('Year_Played').first()[['Team_Name', 'Wins', 'Losses','OT_Losses', 'Win_Perc', 'Goals_For', 'Goals_Against', '+-']]
df_top.reset_index(inplace=True)
df_top.set_index('Team_Name', inplace=True)
# DataFrame of Bottom Teams
df_bottom = df.reset_index()
df_bottom = df_bottom.sort_values(['Wins','+-'])
df_bottom = df_bottom.groupby('Year_Played').first()[['Team_Name', 'Wins', 'Losses','OT_Losses', 'Win_Perc', 'Goals_For', 'Goals_Against', '+-']]
df_bottom.reset_index(inplace=True)
df_bottom.set_index('Team_Name', inplace=True)

with st.sidebar:
    st.button('Original Data')

with st.expander('Original Data'):
    st.dataframe(df,
                 column_config={
                     'Year_Played': st.column_config.NumberColumn(format = '%d')
                 })

# Bar Plots
bar_plot_wins_losses = pd.DataFrame({
    'Team Type': ['Average', 'Top Teams', 'Bottom Teams'],
    'Wins': [df['Wins'].mean(), df_top['Wins'].mean(), df_bottom['Wins'].mean()],
    'Losses': [df['Losses'].mean(), df_top['Losses'].mean(), df_bottom['Losses'].mean()]
})

bar_plot_goals = pd.DataFrame({
    'Team Type': ['Average', 'Top Teams', 'Bottom Teams'],
    'Goals_For': [df['Goals_For'].mean(), df_top['Goals_For'].mean(), df_bottom['Goals_For'].mean()],
    'Goals_Against': [df['Goals_Against'].mean(), df_top['Goals_Against'].mean(), df_bottom['Goals_Against'].mean()]
})

bar_plot_wins_losses_melted = bar_plot_wins_losses.melt(id_vars='Team Type', var_name='Metric', value_name='Average Value')
bar_plot_goals_melted = bar_plot_goals.melt(id_vars='Team Type', var_name='Metric', value_name='Average Value')

st.subheader('Bar Chart')
# Tabs
tab1, tab2 = st.tabs(['Wins and Losses', 'Goals'])

with tab1:
    fig, axe = plt.subplots(figsize=(14, 6))
    sns.barplot(x='Metric', y='Average Value', hue='Team Type', data=bar_plot_wins_losses_melted,
                palette=['grey', 'green', 'red'])
    plt.title('Bar Chart for Wins and Losses')
    st.pyplot()


with tab2:
    fig, axe = plt.subplots(figsize=(14, 6))
    sns.barplot(x='Metric', y='Average Value', hue='Team Type', data=bar_plot_goals_melted,
                palette = ['grey', 'green', 'red'])
    plt.title('Bar Chart for Goals')
    st.pyplot()

st.subheader('Heat Map')

col1, col2 = st.columns([0.7,0.3], gap='medium')

with col1:
    plot_heatmmap()

with col2:
    st.markdown('')




