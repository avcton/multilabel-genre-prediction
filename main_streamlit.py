import json
import string
import spacy
import numpy as np
import pandas as pd
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


# Input Variables

input_desc = ""
input_desc_word_count = 0

# Prediction Variables

probs = []
boundary_level = 50

# Output Variables

predictedGenre = '-'

# Analysis Visualisations

dataframe = pd.DataFrame({
    'Genre': genre,
    'Likelihood': np.zeros_like(genre, dtype=int)
})
