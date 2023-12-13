import json
import string
import spacy
import numpy as np
import pandas as pd
from taipy.gui import Gui, notify
from tensorflow.keras import models
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Load the spaCy Module
nlp = spacy.load("en_core_web_sm")

# Load the trained RNN Model
model = models.load_model('Genre-Prediction.h5')


# Load the Tokeniser Embeddings
with open("tokenizer.json", "r") as json_file:
    tokenizer_json = json.load(json_file)
    tokenizer = tokenizer_from_json(tokenizer_json)

# Initiate Genre Classes
genre = np.array(['Action', 'Action & Adventure', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'History',
                  'Horror', 'Kids', 'Music', 'Mystery', 'News', 'Reality', 'Romance', 'Sci-Fi & Fantasy', 'Science Fiction', 'Soap', 'TV Movie', 'Talk', 'Thriller', 'War', 'War & Politics', 'Western'])


def filterName(description: str):
    '''
    This function given a description, normalises it by removing names and punctuations
    '''
    # Remove Punctuations
    description = description.translate(
        str.maketrans('', '', string.punctuation))

    # Perform NER using spaCy
    doc = nlp(description)

    for entity in doc.ents:
        if entity.label_ == 'PERSON':
            description = description.replace(str(entity), '')

    # Convert to Lowercase
    description = description.lower()

    # description = remove_stopwords(description)

    return " ".join(description.split())


def predictGenre(description: str):
    description = tokenizer.texts_to_sequences(description)
    description = pad_sequences(description, padding='post', maxlen=500)
    pred = model.predict([description])

    return pred[0]


StatusBar = {
    'status': 'info',
    'message': 'Write any Plot to Predict'
}

# Taipy Dynamic Variables

desc = ""
probs = []
boundary_level = 50
word_count = 0
predictedGenre = '-'

# Taipy UI

index = """
<|text-center|
# Welcome to **KDrama Insights**{: .color-primary}

<|toggle|theme=true|>

<|layout|columns=1 1|
<|
<|{StatusBar}|status|>
<|Predict|button|on_action=button_pressed|>


<|
**<|Word Count = {word_count} -> MAX = 500 Without Names|hover_text=This is without Removing Names|>**{: .color-primary}

<|{desc}|input|multiline=true|lines_shown=3|class_name=fullwidth|hover_text=Input Plot Sire ;)|>
|>
|>

<|
# Predicted Genre
<|text-center|
<br></br>

**<|{predictedGenre}|hover_text=Yes, This is your Prediction|>**{: .color-primary}
|>

<br></br>
<br></br>

<|{boundary_level}|slider|min=10|max=100|on_change=selectGenre|change_delay=1|hover_text=Confidence Level of the Model|continuous=false|>
|>
|>

<|text-center|
# Analysis
|>

<|layout|columns=1 1|
<|Table|expandable|
<|{dataframe}|table|height=30.1vh|filter=true|number_format=%.2f|>
|>
<|text-center|
<|{dataframe_dic}|chart|type=pie|values=Likelihood|labels=Genre|options={pieChartOptions}|layout={pieChartLayout}|>
|>
|>
|>
"""

# Taipy Functions


def on_change(state, var_name, var_val):
    if var_name == "desc":
        state.word_count = len(var_val.split())

        if state.word_count >= 500:
            notify(state, "warning", f"Be Careful, Words Execude 500 Count")


def selectGenre(state):
    qualified_genre = list(
        np.where(state.probs >= state.boundary_level/100)[0])
    state.predictedGenre = ', '.join(state.genre[qualified_genre])
    if state.predictedGenre == '':
        state.predictedGenre = '-'


def button_pressed(state):

    if len(state.desc) == 0:
        notify(state, "error", f"Please Enter a Description")
        return None

    notify(state, "info", f"Hold Tight, The Model is Predicting...")
    state.probs = predictGenre([state.desc])
    selectGenre(state)

    state.dataframe = state.dataframe.sort_values(
        by='Genre')
    state.dataframe['Likelihood'] = state.probs
    state.dataframe = state.dataframe.sort_values(
        by='Likelihood', ascending=False)
    state.dataframe_dic = state.dataframe.to_dict()
    notify(state, "success", f"Your Prediction is Ready ;)")


# Analysis  Variables

dataframe = pd.DataFrame({
    'Genre': genre,
    'Likelihood': np.zeros_like(genre, dtype=int)
})

dataframe_dic = dataframe.to_dict()

pieChartOptions = {
    "hole": 0.4,
}

pieChartLayout = {
    "title": "Genre Likelihoods",
}

app = Gui(page=index)

if __name__ == "__main__":
    app.run(use_reloader=True, port=5001, title="Genre Predictor",
            watermark="developed with love by avcton <3", favicon="icons/favicon.svg")
