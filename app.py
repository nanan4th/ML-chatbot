import random
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import re

from flask import Flask, request, jsonify
from sklearn import preprocessing
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = tf.keras.models.load_model('model/model.h5')

app = Flask(__name__)

swords = set(stopwords.words("english"))


def parse_text(text):
    # Remove punctuation
    text = re.sub("[^a-zA-Z]", ' ', text)

    # Convert to lowercase
    text = text.lower().split()

    # Remove stopwords
    swords = set(stopwords.words("english"))
    text = [w for w in text if w not in swords]
    text = " ".join(text)

    return text


def to_df(data):
    tags = []
    inputs = []
    responses = {}
    for intent in data['intents']:
        responses[intent['tag']] = intent['responses']
        for lines in intent['input']:
            lines = parse_text(lines)
            inputs.append(lines)
            tags.append(intent['tag'])

    data = pd.DataFrame({"inputs": inputs,
                         "tags": tags})

    return data, responses


def fit_tokenizer(data, oov_token):
    tokenizer = Tokenizer(num_words=1000, oov_token=oov_token)
    tokenizer.fit_on_texts(data)

    return tokenizer


def tok_pad_seq(text_pred, tokenizer):
    pred_input = tokenizer.texts_to_sequences(text_pred)
    pred_input = np.array(pred_input).reshape(-1)
    pred_input = pad_sequences(
        [pred_input], maxlen=11, padding='post', truncating='post')

    return pred_input


@app.route('/', methods=['POST'])
def index():
    text_pred = []

    # Request input text
    json_data = request.json
    pred_input = json_data['text']

    # Load intent json
    with open('data/data.json') as content:
        data = json.load(content)

    # Convert to dataframe
    content_data = to_df(data)
    data = content_data[0]
    responses = content_data[1]

    # Encode intent labels
    label_encoder = preprocessing.LabelEncoder()
    labels = np.array(data['tags'])
    labels = label_encoder.fit_transform(labels)

    # Tokenize intent data
    tokenizer = fit_tokenizer(data['inputs'], "<OOV>")

    # Input text cleaning
    pred_input = parse_text(pred_input)
    text_pred.append(pred_input)

    # Tokenize input text
    pred_input = tok_pad_seq(text_pred, tokenizer)

    # Predict output
    output = model.predict(pred_input)
    output = output.argmax()

    tag = label_encoder.inverse_transform([output])[0]

    return jsonify(
        tag=tag,
        message=random.choice(responses[tag])
    )


if __name__ == '__main__':
    app.run(port=5000, debug=True)
