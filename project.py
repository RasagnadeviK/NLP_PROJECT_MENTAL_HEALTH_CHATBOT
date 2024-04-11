import numpy as np
import pandas as pd
import json
import os
import re
import random
from flask import Flask, render_template, request

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, LayerNormalization, Dense, Dropout

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']
    response = generate_answer(user_input)
    return {'response': response}

def load_data():
    with open('intents.json', 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data['intents'])
    dic = {"tag": [], "patterns": [], "responses": []}
    for i in range(len(df)):
        ptrns = df[df.index == i]['patterns'].values[0]
        rspns = df[df.index == i]['responses'].values[0]
        tag = df[df.index == i]['tag'].values[0]
        for j in range(len(ptrns)):
            dic['tag'].append(tag)
            dic['patterns'].append(ptrns[j])
            dic['responses'].append(rspns)
    df = pd.DataFrame.from_dict(dic)
    return df

def preprocess_data(df):
    tokenizer = Tokenizer(lower=True, split=' ')
    tokenizer.fit_on_texts(df['patterns'])
    vocab_size = len(tokenizer.word_index)
    pattern_sequences = tokenizer.texts_to_sequences(df['patterns'])
    X = pad_sequences(pattern_sequences, padding='post')
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['tag'])
    return X, y, tokenizer, label_encoder

def create_model(vocab_size, max_sequence_len, num_classes):
    model = Sequential([
        Input(shape=(max_sequence_len,)),
        Embedding(input_dim=vocab_size + 1, output_dim=100, mask_zero=True),
        LSTM(32, return_sequences=True),
        LayerNormalization(),
        LSTM(32, return_sequences=True),
        LayerNormalization(),
        LSTM(32),
        LayerNormalization(),
        Dense(128, activation="relu"),
        LayerNormalization(),
        Dropout(0.2),
        Dense(128, activation="relu"),
        LayerNormalization(),
        Dropout(0.2),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    return model

def train_model(model, X, y):
    model.fit(x=X, y=y, batch_size=10, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)], epochs=50)
    return model

def generate_answer(pattern):
    if not pattern:
        return "RDZ: Please provide a valid input."
    text = [re.sub('[^a-zA-Z\']', ' ', pattern).lower()]
    x_test = tokenizer.texts_to_sequences(text)
    if not x_test:
        return "RDZ: I'm sorry, I don't understand that."
    x_test = pad_sequences(x_test, padding='post', maxlen=X.shape[1])
    y_pred = model.predict(x_test)
    y_pred = y_pred.argmax()
    tag = label_encoder.inverse_transform([y_pred])[0]
    responses = df[df['tag'] == tag]['responses'].values[0]
    return "RDZ: " + random.choice(responses)

df = load_data()
X, y, tokenizer, label_encoder = preprocess_data(df)
model = create_model(vocab_size=len(tokenizer.word_index), max_sequence_len=X.shape[1], num_classes=len(np.unique(y)))
model = train_model(model, X, y)

if __name__ == '__main__':
    app.run(debug=True)
