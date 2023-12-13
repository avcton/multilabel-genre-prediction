import time
import json
import spacy
import string
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from tensorflow.keras import models
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

st.set_page_config(
    page_title="Genre Prediction",
    page_icon="icons/favicon.svg",
    menu_items={
        "Get Help": None,
        'Report a bug': 'mailto:avcton@gmail.com',
        'About': "The underlying model is a trained and hypertuned BiLSTM RNN."
    },
    layout="centered")

# Initialising Model Requirements


@st.cache_resource(show_spinner='Catching the Train...')
def init_spaCy():
    # Load the spaCy Module
    time.sleep(0.3)
    return spacy.load("en_core_web_sm")


@st.cache_resource(show_spinner='Hopping inside it ...')
def init_tokenizer():
    # Load the Tokeniser Embeddings
    time.sleep(0.3)
    with open("utils/tokenizer.json", "r") as json_file:
        tokenizer_json = json.load(json_file)
        return tokenizer_from_json(tokenizer_json)


@st.cache_resource(show_spinner='Fuelling the Nueral Model...')
def init_model():
    # Load the trained RNN Model
    return models.load_model('model/Genre-Prediction.h5')


@st.cache_data
def init_genreClasses():
    # Initiate Genre Classes
    return np.array([
        'Action', 'Action & Adventure', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'History',
        'Horror', 'Kids', 'Music', 'Mystery', 'News', 'Reality', 'Romance', 'Sci-Fi & Fantasy', 'Science Fiction', 'Soap', 'TV Movie', 'Talk', 'Thriller', 'War', 'War & Politics', 'Western'])

# Utility Functions for Pipeline


@st.cache_data
def filterName(description: str):
    '''
    This function given a description, normalises it by removing names and punctuations
    '''
    # Remove Punctuations
    description = description.translate(
        str.maketrans('', '', string.punctuation))

    # Perform NER using spaCy
    nlp = init_spaCy()
    doc = nlp(description)

    for entity in doc.ents:
        if entity.label_ == 'PERSON':
            description = description.replace(str(entity), '')

    # Convert to Lowercase
    description = description.lower()

    # description = remove_stopwords(description)

    return " ".join(description.split())


@st.cache_data(show_spinner=False)
def predictGenre(description: str):

    p_bar = st.progress(0, ':phone: Waking up the model .')
    time.sleep(0.3)

    tokenizer = init_tokenizer()

    p_bar.progress(30, ':necktie: Dressing up the model ..')
    time.sleep(0.3)

    description = filterName(description)
    description = tokenizer.texts_to_sequences(description)
    description = pad_sequences(description, padding='post', maxlen=500)

    p_bar.progress(50, ":shallow_pan_of_food: Feeding food to the model ...")

    model = init_model()
    pred = model.predict([description])

    p_bar.progress(98, ":dodo: Patting its head ...")
    time.sleep(0.3)
    p_bar.empty()

    return pred[0]


# inits

genre = init_genreClasses()
init_spaCy()
init_model()
init_tokenizer()

# Input Variables

if 'input_desc' not in st.session_state:
    st.session_state['input_desc'] = ''

if 'input_desc_word_count' not in st.session_state:
    st.session_state['input_desc_word_count'] = 0

if 'dataframe' not in st.session_state:
    dataframe = pd.DataFrame({
        'Genre': genre,
        'Likelihood': np.zeros_like(genre, dtype=int)
    })
    st.session_state['dataframe'] = dataframe

if 'probs' not in st.session_state:
    st.session_state['probs'] = np.array([])

if 'threshold' not in st.session_state:
    st.session_state['threshold'] = 0.5

# Callbacks


def textInputUpdate():
    input = st.session_state.input_desc
    length = len(input.split())
    st.session_state.input_desc_word_count = length

    if length == 0:
        st.session_state.dataframe = pd.DataFrame({
            'Genre': genre,
            'Likelihood': np.zeros_like(genre, dtype=int)
        })

        st.session_state.probs = np.array([])


def predictionUpdate():
    if st.session_state.input_desc_word_count > 0:
        input = st.session_state.input_desc
        st.session_state.dataframe['Likelihood'] = predictGenre(input)

# User Interface


st.title("Genre Prediction")

input_label_col, input_info_col = st.columns([2, 1])

input_label_col.markdown("Enter a Description / Plot")

input_info_max_count_col, input_info_word_count_col = input_info_col.columns(2)

input_info_max_count_col.caption('Max Count = :violet[500]')

if st.session_state.input_desc_word_count > 500:
    input_info_word_count_col.caption(
        f'Word Count = :red[{st.session_state.input_desc_word_count}]')

else:
    input_info_word_count_col.caption(
        f'Word Count = :green[{st.session_state.input_desc_word_count}]')

st.text_area('Enter a Description / Plot', key='input_desc', height=150,
             placeholder="It was all going good for billy but one day...", label_visibility='collapsed', on_change=textInputUpdate)

header_col, button_col = st.columns([3, 1])

header_col.header("Predicted Genre", divider='gray')

if button_col.button(
        'Predict', use_container_width=True, type="primary"):

    if st.session_state.input_desc_word_count > 0:
        input = st.session_state.input_desc
        st.session_state.dataframe.sort_values(by='Genre', inplace=True)
        st.session_state.probs = predictGenre(input)
        st.session_state.dataframe['Likelihood'] = st.session_state.probs
        st.session_state.dataframe.sort_values(
            by='Likelihood', ascending=False, inplace=True)

qualified_genre = list(
    np.where(st.session_state.probs >= st.session_state.threshold)[0])

with st.container(border=True):
    predictedGenre = '-'
    if len(qualified_genre) > 0:
        predictedGenre = ', '.join(genre[qualified_genre])

    st.markdown(f'<center>{predictedGenre}</center>',
                unsafe_allow_html=True)

    st.slider('Select Threshold Level', key='threshold',
              min_value=0.1, max_value=1.0, format='%f')

df = st.session_state.dataframe.copy()

df['Genre'] = df.apply(
    lambda row: row['Genre'] if row['Likelihood'] >= 0.1 else 'Others', axis=1)

grouped_data = df.groupby(
    'Genre', as_index=False).agg({'Likelihood': 'sum'})

pie = go.Figure(
    data=[go.Pie(labels=df['Genre'], values=df['Likelihood'], hole=.3)]
)

pie.update_layout(title='Genre Likelihoods')

graph_col, table_col = st.columns([2, 1])

table_col.dataframe(st.session_state.dataframe,
                    use_container_width=True, hide_index=True)

graph_col.plotly_chart(pie, use_container_width=True)

st.divider()
st.caption("<center>made with <3 by avcton</center>",
           unsafe_allow_html=True)
