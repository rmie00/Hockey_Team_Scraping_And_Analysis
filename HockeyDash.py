import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sqlite3
import plotly.figure_factory as ff


st.set_page_config(layout='wide',
                   page_title= 'HockeyDash',
                   page_icon= ':bar-char:')


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
df_team.reset_index(drop=False,inplace= True)
df_team = df_team.sort_values(by='Win_Perc', ascending=True)
# DataFrame Of Mean Values For Each
df_year = round(df.groupby('Year_Played').mean(0), 2)
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
# Melted Bar Chart
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

bar_plot_wins_losses_melted = bar_plot_wins_losses.melt(id_vars='Team Type', var_name='Metric',
                                                            value_name='Average Value')
bar_plot_goals_melted = bar_plot_goals.melt(id_vars='Team Type', var_name='Metric', value_name='Average Value')

####Correlation Map
corre = round(df.corr(), 2)
mask = np.triu(np.ones_like(corre, dtype= bool))
corre_mask = corre.mask(mask)

# Side Bar Methods
def get_unique_year(year):
    unique_year = df_rst[df_rst['Year_Played'] == year]
    return unique_year

def get_unique_team(team):
    unique_team = df_rst[df_rst['Team_Name'] == team]
    return unique_team

def get_yearly_averages(year):
    yearly_avg = df[df['Year_Played'] == year].mean()
    return yearly_avg

def get_unique(df_data, team, year):
    unique = df_data.loc[(df_data['Team_Name'] == team) & (df_data['Year_Played'] == year)]
    return unique

# Visualisation Methods
### Compare To Other Teams Gauge
def compare_gauge(type, avg, df, suffix = None, prefix = None, range= [0,100]):
    team_avg = avg[type].iloc[0]
    league_avg = df[type].mean()
    colour = 'red' if team_avg < league_avg else 'green'
    fig = go.Figure(go.Indicator(
        domain={'x': [0, 1], 'y': [0, 1]},
        value=team_avg,
        delta={'reference': league_avg},
        number={'suffix': suffix,
                'prefix': prefix},
        title={'text': type,
               'font': {'size': 50}},
        mode='gauge+number+delta',
        gauge={
            'axis': {'range': range},
            'bar': {'color': colour},
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': team_avg}}
    ))
    fig.update_layout(height = 300,
                      paper_bgcolor='white',
                      margin= dict(l=0, r=0,t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)
#######################################
def compare_num(type, avg, df, prefix= None):
    team_avg = avg[type].iloc[0]
    league_avg = df[type].mean()
    colour = 'red' if team_avg < league_avg else 'green'
    fig = go.Figure(go.Indicator(
        domain= {'x': [0,1], 'y' : [0,1]},
        value=team_avg,
        mode = 'number+delta',
        delta = {'position': 'top', 'reference' : league_avg},
        title = {'text': type,
                 'font': {'size': 50}},
        number = {'font': {'color': colour}}))

    fig.update_layout(height = 300,
                      paper_bgcolor = 'white',
                      margin= dict(l=0, r=0,t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

def league_table(year):
    years = df[df['Year_Played'] == year].reset_index(drop=False).sort_values(by = 'Wins', ascending=True)
    positions = list(range(1, len(years)+1))
    fig = go.Figure(data=[go.Table(
        columnwidth= [40] + [60]*len(years.columns),
        header=dict(values=['Position'] + list(years.columns),
                    fill_color='lightgrey',
                    align='left',
                    line_color = 'black'),
        cells=dict(values=[positions] + [years.Team_Name, years.Year_Played, years.Wins, years.Losses, years.OT_Losses, years.Win_Perc, years.Goals_For, years.Goals_Against, years['+-']],
                   fill_color='white',
                   align='left',
                   line_color = 'black'))
    ])

    fig.update_layout(margin= dict(l=0, r=0,t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

def plot_line(team):
    team_sel = df.loc[team]
    leauge_avg = df_year

    fig = go.Figure()

    fig.add_trace(go.Scatter(x= team_sel['Year_Played'],
                             y = team_sel['Wins'],
                             name = team,
                             mode = 'lines',
                             connectgaps= True,
                             line = {'color': 'green',
                                     'width': 4}))
    fig.add_trace(go.Scatter(x = leauge_avg.index,
                             y = leauge_avg['Wins'],
                             name = 'League Average',
                             mode = 'lines',
                             connectgaps=True,
                             line = {'color' : 'grey',
                                     'width': 2}))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

def plot_bar(dat, metric1, metric2, title):
    fig = px.bar(
        dat,
        x='Metric',
        y='Average Value',
        color='Team Type',
        barmode='group',
        category_orders={'Metric': [metric1, metric2]},
        labels={'Average Value': 'Average Value'},
        title=title
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_winperc_gauge(team, type):
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
    st.plotly_chart(fig, use_container_width=True)

def plot_num(team, type):
    data = df_team.loc[df_team['Team_Name'] == team, type].values[0]
    fig = go.Figure(go.Indicator(
        mode='number+delta',
        value=data,
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    fig.update_layout(height=200,
                      paper_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True)

def plot_num_symb(team, type):
    data = df_team.loc[df_team['Team_Name'] == team, type].values[0]
    fig = go.Figure(go.Indicator(
        mode='number+delta',
        value=data,
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    fig.update_layout(height=200,
                      paper_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True)

def team_df(team):
    df1 = df.reset_index()
    df2 = df1[df1['Team_Name'] == team]
    df2.set_index('Team_Name', inplace=True)
    return st.dataframe(df2)


### SideBar
st.sidebar.text('')
st.sidebar.text('')
st.sidebar.markdown('How Would You Like To Visualise The DataFrame? *You Can Select To View The Whole DataFrame*')
st.sidebar.markdown('*Please Select A Year You Want To Analyse:* 👇')
year_select = st.sidebar.selectbox(
    'Please Select A Year',
    (range(1990, 2012)))
year_selected = int(year_select)
st.sidebar.markdown(f'You Have Selected {year_selected}')

st.sidebar.markdown('*Please Select A Team You Want To Analyse:* 👇')
team_list = df_rst['Team_Name'].unique().tolist()
team_select = st.sidebar.selectbox(
    'Please Select A Team',
    team_list)
team_selected = team_select
st.sidebar.markdown(f'You Have Selected {team_selected}')
all_view = st.sidebar.checkbox('View The Whole Picture')
if all_view:
    st.sidebar.markdown('*You Are Now Viewing The Whole DataFrame*')

valid_yt = (df_rst['Team_Name'] == team_selected) & (df_rst['Year_Played'] == year_selected)


## Main Body ##
if all_view:
    st.subheader('We Will Visualise With All The Data')
    with st.expander('DataFrame'):
        st.dataframe(df,
                     column_config={'Year_Played': st.column_config.NumberColumn(format='%d')})

    st.subheader('Let View The Top and Bottom Team Averages Compared To League Averages')

    col_1, col_2 = st.columns(2)

    with col_1:
        plot_bar(bar_plot_wins_losses_melted, 'Wins', 'Losses', 'Bar Chart For Wins And Losses')

    with col_2:
        plot_bar(bar_plot_goals_melted, 'Goals_For', 'Goals_Against', 'Bar Chart For Goals')

    bottom_left, bottom_right = st.columns(2)

    with bottom_left:

        fig = ff.create_annotated_heatmap(z = corre_mask.to_numpy(),
                                          x = corre_mask.columns.tolist(),
                                          y = corre_mask.columns.tolist(),
                                          colorscale='Portland',
                                          hoverinfo = 'none',
                                          showscale=True,
                                          ygap = 1,
                                          xgap= 1)

        fig.update_xaxes(side = 'bottom')

        fig.update_layout(
            title='Correlation Map',
            width=800,
            height=600,
            xaxis_title='Features',
            yaxis_title='Features',
            yaxis_autorange='reversed')
        for i in range(len(fig.layout.annotations)):
            if fig.layout.annotations[i].text == 'nan':
                fig.layout.annotations[i].text = ""

        fig.update_annotations(font= {'color': 'white'})

        st.plotly_chart(fig, use_container_width=True)

        with bottom_right:
            fig = px.bar(df_team,
                         x = 'Win_Perc',
                         y = 'Team_Name',
                         color = 'Wins',
                         color_continuous_scale= 'portland',
                         orientation= 'h',
                         text = 'Wins',
                         title='NHl Team Ranking From 1990 To 2011')
            fig.update_layout(xaxis_title= 'Win Percentage',
                              yaxis_title = 'Team Names',
                              coloraxis_colorbar_title = 'Wins',
                              font = {'size': 80},
                              height = 600)
            fig.update_traces(texttemplate='%{text:.2s}', textposition='outside', textfont_size=50)
            st.plotly_chart(fig, use_container_width=True)

elif valid_yt.any():
    df_unique = get_unique(df_rst, team_selected, year_selected)
    df_unique.set_index('Team_Name', inplace=True)
    df_year_avg = get_yearly_averages(year_selected)

    st.subheader(f'{team_selected} ({year_selected})')
    with st.expander(f'{team_selected} DataFrame'):
        st.dataframe(df_unique,
                     column_config={'Year_Played': st.column_config.NumberColumn(format='%d')})

    st.subheader(f'Comparing {team_selected} Statistics For The Year {year_selected} With League Averages')
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        compare_num('Wins', df_unique, df_year_avg)
    with col2:
        compare_gauge('Win_Perc', df_unique, df_year_avg, suffix='%')
    with col3:
        compare_num('Goals_For', df_unique, df_year_avg)

    with col4:
        if int(df_unique['+-']) < 0:
            compare_gauge('+-', df_unique, df_year_avg, prefix="-", range=[0, -300])
        else:
            compare_gauge('+-', df_unique, df_year_avg, prefix="+", range=[0, 300])

    col_1, col_2 = st.columns(2)
    with col_1:
        st.subheader(f'League Table For The Year {year_selected}')
        league_table(year_selected)

    with col_2:
        st.subheader(f'Comparing {team_selected} Wins With League Averages By Year')
        plot_line(team_selected)
else:
    st.error('No Such Combination In The DataFrame. Try Selecting Another Combination')



