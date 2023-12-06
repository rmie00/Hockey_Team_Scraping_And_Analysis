import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout='wide')

###Data Import###
@st.cache_data
def load_sqlite(path: str, table: str, index_col: str):
    dat = sqlite3.connect(path)
    data = pd.read_sql_query(f'SELECT * FROM {table}',
                             dat,
                             index_col = [f'{index_col}'])
    return data
# Original DataFrame
df = load_sqlite('HockeyDataBase.sqlite',
                 'Performance',
                 'Team_Name')
df['OT_Losses'] = pd.to_numeric(df['OT_Losses'])
df['OT_Losses'] = (df['OT_Losses'].fillna(0)).astype(int)
df['Win_Perc'] = round((df['Wins'] / (df['Wins'] + df['Losses']))*100, 2)
df['+-'] = df['Goals_For'] - df['Goals_Against']
df = df[['Year_Played', 'Wins', 'Losses','OT_Losses', 'Win_Perc', 'Goals_For', 'Goals_Against', '+-']]
# Select By Team Average DataFrame Without Year Played
df_team = round(df.groupby('Team_Name').mean(), 2)
df_team.drop('Year_Played', axis = 1, inplace= True)
df_team.reset_index(inplace= True)
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
# DataFrame with no Index
df_rst = df.reset_index(drop=False).copy()

# Side Bar Methods
def get_unique_year(df_data, year):
    unique_year = df_data[df_data['Year_Played'] == year]
    return unique_year

def get_unique_team(df_data, team):
    unique_team = df_data[df_data['Team_Name'] == team]
    return unique_team

def get_yearly_averages(year):
    yearly_avg = df[df['Year_Played'] == year].mean()
    return yearly_avg

def get_unique(df_data, team, year):
    unique = df_data.loc[(df_data['Team_Name'] == team) & (df_data['Year_Played'] == year)]
    return unique


# Visualisation Methods

def compare_win_gauge(type, avg, df):
    team_avg = avg[type].iloc[0]
    league_avg = df[type]
    colour = 'red' if team_avg < league_avg else 'green'
    fig = go.Figure(go.Indicator(
        domain={'x': [0, 1], 'y': [0, 1]},
        value=team_avg,
        delta={'reference': league_avg},
        number={'suffix': '%'},
        title={'text': type},
        mode='gauge+number+delta',
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': colour},
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': team_avg}}
    ))
    fig.update_layout(height=500,
                      paper_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True)
#######################################
def compare_num(type, avg, df):
    team_avg = avg[type].iloc[0]
    league_avg = df[type]
    colour = 'red' if team_avg < league_avg else 'green'
    fig = go.Figure(go.Indicator(
        mode='number+delta+gauge',
        value=team_avg,
        title = {'text' : type},
        delta= {'reference': league_avg},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge= {'bar': {'color': colour},
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': team_avg}}
    ))
    fig.update_layout(height=500,
                      paper_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True)


def plot_winperc_gauge(team: str, type: str):
    data = df_team.loc[df_team['Team_Name'] == team, type].values[0]
    colour = 'red' if data < 50 else 'green'
    fig = go.Figure(go.Indicator(
        domain={'x': [0, 1], 'y': [0, 1]},
        value=data,
        number= {'suffix': '%'},
        title={'text': type,
               'size': 28},
    delta={'reference': 50},
    mode='gauge+number',
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': colour},
        'threshold': {
            'line': {'color': "black", 'width': 4},
            'thickness': 0.75,
            'value': data}}
    ))
    fig.update_layout(height=500,
                      width = 500,
                      paper_bgcolor='white')
    st.plotly_chart(fig)

def plot_num(team: str, type: str):
    data = df_team.loc[df_team['Team_Name'] == team, type].values[0]
    fig = go.Figure(go.Indicator(
        mode='number+delta',
        value=data,
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    fig.update_layout(height=200,
                      paper_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True)

def plot_num_symb(team: str, type: str):
    data = df_team.loc[df_team['Team_Name'] == team, type].values[0]
    fig = go.Figure(go.Indicator(
        mode='number+delta',
        value=data,
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    fig.update_layout(height=200,
                      paper_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True)

def team_df(team: str):
    df1 = df.reset_index()
    df2 = df1[df1['Team_Name'] == team]
    df2.set_index('Team_Name', inplace=True)
    return st.dataframe(df2)


### SideBar
st.sidebar.text('')
st.sidebar.text('')
agreed_year = st.sidebar.checkbox('*Do You Want To Select A Specific Year')
###
if agreed_year:
    st.sidebar.markdown('*Please Select A Year You Want To Analyse:* 👇')
    year_select = st.sidebar.selectbox(
        'Please Select A Year',
        (range(1990, 2012)))
    year_selected = int(year_select)
    st.sidebar.markdown(f'You Have Selected {year_selected}')
###
st.sidebar.text('')
agreed_team = st.sidebar.checkbox('*Do You Want To Select A Specific Team')
st.sidebar.text('')

if agreed_team:
    st.sidebar.markdown('*Please Select A Team You Want To Analyse:* 👇')
    team_list = df_rst['Team_Name'].unique().tolist()
    team_select = st.sidebar.selectbox(
        'Please Select A Team',
        team_list)
    team_selected = str(team_select)
    st.sidebar.markdown(f'You Have Selected {team_selected}')

## Main Body ##

if agreed_year and agreed_team:
    st.header(f'{team_selected} ({year_selected})')
    df_unique = get_unique(df_rst, team_selected, year_selected)
    df_unique.set_index('Team_Name', inplace = True)
    with st.expander(f'{team_selected} DataFrame'):
        st.dataframe(df_unique)
    df_year_avg = get_yearly_averages(year_selected)
    compare_win_gauge('Win_Perc',df_unique, df_year_avg)
    compare_num('Wins', df_unique, df_year_avg)






