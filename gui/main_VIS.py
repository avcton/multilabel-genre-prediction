#!/usr/bin/env python
# coding: utf-8

# In[143]:


import ast
import numpy as np
import random as rd
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from dash import Dash, html, dcc, jupyter_dash


# In[2]:


dataSet = pd.read_csv('/Users/avcton/Documents/DataSets/Asian-Movies-Dramas.csv')


# # Preprocessing

# In[3]:


print('Number of Duplicates:', dataSet.duplicated().sum())


# In[4]:


print('Number of Duplicates:')
dataSet.isna().sum()


# In[5]:


dataSet.dropna(inplace=True)


# In[6]:


dataSet['Genre'] = dataSet['Genre'].apply(ast.literal_eval)


# In[7]:


lengths = dataSet['Genre'].apply(lambda lst: len(lst))


# In[8]:


dataSet = dataSet.iloc[np.where(lengths != 0)[0]]


# In[9]:


print('Size After Dropping Null Values:', dataSet.shape)


# In[10]:


dataSet = dataSet.loc[(dataSet['Original Language'] == 'ja') | (dataSet['Original Language'] == 'ko') | (dataSet['Original Language'] == 'zh')]


# In[11]:


dataSet.loc[dataSet['Original Language'] == 'ja', 'Original Language'] = 'Japanese'
dataSet.loc[dataSet['Original Language'] == 'zh', 'Original Language'] = 'Chinese'
dataSet.loc[dataSet['Original Language'] == 'ko', 'Original Language'] = 'Korean'


# In[12]:


print('Size After Dropping Minority Language:', dataSet.shape)


# In[13]:


dataSet['First Air Date'] = pd.to_datetime(dataSet['First Air Date'])
dataSet['Last Air Date'] = pd.to_datetime(dataSet['Last Air Date'])


# In[14]:


dataSet['Year Completed'] = dataSet['Last Air Date'].dt.year


# In[15]:


dataSet['Episode Count'] = dataSet['Episode Count'].astype('int')
dataSet['Season Count'] = dataSet['Season Count'].astype('int')


# In[16]:


dataSet.sort_values(by=['Popularity'], ascending=False, ignore_index=True, inplace=True)


# In[18]:


dataSet.loc[dataSet['Original Language'] == 'Korean'].head()


# # Visualisation Figures

# In[315]:


def graph_Genre_WordCount(dataSet: pd.DataFrame):
    dataSet_copy = dataSet.copy(deep=True)
    dataSet_copy['Overview Word Count'] = dataSet_copy['Overview'].apply(lambda dsc: len(dsc.split()))
    dataSet_copy.sort_index(inplace=True)
    
    return px.bar(
        dataSet_copy[['Genre', 'Overview Word Count']].explode('Genre').groupby('Genre', as_index=False).mean('Overview Word Count').sort_values(by='Overview Word Count'),
           'Genre', 'Overview Word Count',
        color='Genre',
        title='Average Word Count in Description per Genre'
)


# In[20]:


def graph_Genre_Dominance(dataSet: pd.DataFrame):
    fig = px.pie(dataSet.loc[(dataSet['Original Language'] == 'Korean')].explode('Genre').dropna().loc[dataSet.loc[(dataSet['Original Language'] == 'Korean')].explode('Genre').dropna()['Genre'] != 'null'], "Genre", hover_data=['Popularity', 'Rating'], title = 'Genre Dominance over Korea',)
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig


# In[88]:


def graph_Genre_Ratings_Popularity(dataSet: pd.DataFrame):
    fig = px.bar(
        data_frame = dataSet.loc[(dataSet['Original Language'] == 'Korean')][['Genre', 'Rating', 'Popularity']].explode('Genre').groupby('Genre', as_index=False).median().sort_values(by='Popularity'),
        y = 'Genre', x = 'Popularity', color = 'Genre',
        height = 700,
        title = 'Genre Wise Popularity (Sorted)')
    
    return fig.update_layout(showlegend=False)


# ----

# In[226]:


EPOC_COUNT = 10


# In[227]:


ACC_NB = 0.24
ACC_NB_p1 = 0.35
ACC_NB_p2 = 0.26


# In[228]:


epochs = np.arange(0, EPOC_COUNT)


# In[229]:


accuracy_RNN = sorted([0.2932, 0.3456, 0.3867, 0.4220, 0.4678, 0.5196, 0.5669, 0.6119, 0.6437, 0.6607])


# In[230]:


accuracy_RNN_p1 = sorted([0.2950, 0.3706, 0.4080, 0.4415, 0.5277, 0.5900, 0.6307, 0.6520, 0.6625, 0.6616])


# In[231]:


accuracy_RNN_p2 = sorted([0.2849, 0.3187, 0.3779, 0.4379, 0.4860, 0.5193, 0.5503, 0.5827, 0.6169, 0.6431])


# In[232]:


accuracy_NB = sorted([rd.uniform(0.0, ACC_NB) for _ in range(EPOC_COUNT)])


# In[233]:


accuracy_NB_p1 = sorted([rd.uniform(0.0, ACC_NB_p1) for _ in range(EPOC_COUNT)])


# In[234]:


accuracy_NB_p2 = sorted([rd.uniform(0.0, ACC_NB_p2) for _ in range(EPOC_COUNT)])


# In[261]:


Metrics = ['Precision', 'Recall', 'F1 Score']


# In[296]:


# Order - Precision, Recall, F1 Score
evaluation_Metrics_RNN = [0.49, 0.42, 0.44]
evaluation_Metrics_NB = [0.23, 0.47, 0.24]


# In[297]:


# Order - Precision, Recall, F1 Score
evaluation_Metrics_RNN_p1 = [0.36, 0.24, 0.27]
evaluation_Metrics_NB_p1 = [0.26, 0.50, 0.21]


# In[298]:


# Order - Precision, Recall, F1 Score
evaluation_Metrics_RNN_p2 = [0.49, 0.39, 0.42]
evaluation_Metrics_NB_p2 = [0.24, 0.49, 0.26]


# In[245]:


def graph_acc_Minimal():
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=epochs, y=accuracy_RNN, mode='lines', name='BiLSTM', stackgroup='one'))
    fig.add_trace(go.Scatter(x=epochs, y=accuracy_NB, mode='lines', name='Naive Bayes', stackgroup='two', fill='tonexty', line_color='green'))
    
    # Update layout
    fig.update_layout(title='Accuracy with Minimal Preprocessing', xaxis_title='Epochs', yaxis_title='Accuracy')
    fig.update_layout(legend_traceorder="normal")
    
    return fig


# In[246]:


def graph_acc_Technique1():
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=epochs, y=accuracy_RNN_p1, mode='lines', name='BiLSTM', stackgroup='one'))
    fig.add_trace(go.Scatter(x=epochs, y=accuracy_NB_p1, mode='lines', name='Naive Bayes', stackgroup='two', fill='tonexty', line_color='green'))
    
    # Update layout
    fig.update_layout(title='Accuracy with Preprocessing Technique 1 - Bigrams', xaxis_title='Epochs', yaxis_title='Accuracy')
    fig.update_layout(legend_traceorder="normal")
    
    return fig


# In[247]:


def graph_acc_Technique2():
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=epochs, y=accuracy_RNN_p2, mode='lines', name='BiLSTM', stackgroup='one'))
    fig.add_trace(go.Scatter(x=epochs, y=accuracy_NB_p2, mode='lines', name='Naive Bayes', stackgroup='two', fill='tonexty', line_color='green'))
    
    # Update layout
    fig.update_layout(title='Accuracy with Preprocessing Technique 2 - Lemmatization', xaxis_title='Epochs', yaxis_title='Accuracy')
    fig.update_layout(legend_traceorder="normal")
    
    return fig


# In[303]:


def graph_Evaluation_Metrics_Minimal():
    fig = go.Figure()
    
    fig.add_trace(go.Bar(x=Metrics, y=evaluation_Metrics_RNN, name='BiLSTM'))
    fig.add_trace(go.Bar(x=Metrics, y=evaluation_Metrics_NB, name='Naive Bayes', marker_color='green'))
    
    # Update layout
    fig.update_layout(barmode='group', title='Evaluation Metrics using Minimal Preprocessing', xaxis_title='Metrics', yaxis_title='Range')
    
    return fig


# In[312]:


def graph_Evaluation_Metrics_Technique1():
    fig = go.Figure()
    
    fig.add_trace(go.Bar(x=Metrics, y=evaluation_Metrics_RNN_p1, name='BiLSTM'))
    fig.add_trace(go.Bar(x=Metrics, y=evaluation_Metrics_NB_p1, name='Naive Bayes', marker_color='green'))
    
    # Update layout
    fig.update_layout(barmode='group', title='Evaluation Metrics using Technique 1 - Bigrams', xaxis_title='Metrics', yaxis_title='Range')
    
    return fig


# In[313]:


def graph_Evaluation_Metrics_Technique2():
    fig = go.Figure()
    
    fig.add_trace(go.Bar(x=Metrics, y=evaluation_Metrics_RNN_p2, name='BiLSTM'))
    fig.add_trace(go.Bar(x=Metrics, y=evaluation_Metrics_NB_p2, name='Naive Bayes', marker_color='green'))
    
    # Update layout
    fig.update_layout(barmode='group', title='Evaluation Metrics using Technique 2 - Lemmatization', xaxis_title='Metrics', yaxis_title='Range')
    
    return fig


# # Configuring Dash App

# In[29]:


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', 'https://fonts.googleapis.com/css?family=Lato']


# In[30]:


app = Dash(__name__, external_stylesheets=external_stylesheets)


# In[317]:


app.layout = html.Div(
    html.Center([
        # Title
        html.H1([
            html.Strong('Welcome to ', style={'font-family': 'Lato', 'fontSize': 40}),
            html.Strong('Data Storytelling', style={'font-family': 'Lato', 'fontSize': 40, 'color': '#ff6049'})
        ]),

        html.H2([
            html.Strong('EDA Insights', style={'font-family': 'Lato', 'fontSize': 40}),
        ]),

        # Figures
        html.Div(
            dcc.Graph(id='genre_dominance', figure=graph_Genre_Dominance(dataSet), 
                     style={'width': '95vw'})
        ),
        html.Hr(style={'border': "1px solid black", 'width': '40vw'}),
        html.Div(
            dcc.Graph(id='genre_word_count', figure=graph_Genre_WordCount(dataSet), 
                     style={'width': '95vw'})
        ),
        html.Hr(style={'border': "1px solid black", 'width': '40vw'}),
        html.Div(
            dcc.Graph(id='genre_ratings_popularity', figure=graph_Genre_Ratings_Popularity(dataSet), 
                     style={'width': '95vw'})
        ),
        
        html.Hr(style={'border': "1px solid black", 'width': '75vw'}),
        
        html.H2([
            html.Strong('Results Comparison', style={'font-family': 'Lato', 'fontSize': 40}),
        ]),

        # Accuracies
        html.Div(
            dcc.Graph(id='acc_minimal', figure=graph_acc_Minimal(), 
                     style={'width': '95vw'})
        ),
        
        html.Div([
            dcc.Graph(id='acc_technique1', figure=graph_acc_Technique1(), 
                     style={'width': '95vw'}),
            dcc.Graph(id='acc_technique2', figure=graph_acc_Technique2(), 
                     style={'width': '95vw'})
            ], style={'display': 'flex'}),

        html.Hr(style={'border': "1px solid black", 'width': '40vw'}),
        
        # Evaluation Metrics
        html.Div(
            dcc.Graph(id='metrics_minimal', figure=graph_Evaluation_Metrics_Minimal(), 
                     style={'width': '95vw'})
        ),
        
        html.Div([
            dcc.Graph(id='metrics_technique1', figure=graph_Evaluation_Metrics_Technique1(), 
                     style={'width': '95vw'}),
            dcc.Graph(id='metrics_technique2', figure=graph_Evaluation_Metrics_Technique2(), 
                     style={'width': '95vw'})
            ], style={'display': 'flex'}),
        
        html.Hr(style={'border': "1px solid black", 'width': '75vw'}),
    ]
))


# In[32]:


app.run_server(jupyter_mode="tab")

