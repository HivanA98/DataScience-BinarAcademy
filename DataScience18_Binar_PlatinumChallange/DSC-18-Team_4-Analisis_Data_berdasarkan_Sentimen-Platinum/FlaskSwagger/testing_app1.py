from flask import Flask, jsonify, request
from flasgger import Swagger, LazyString, LazyJSONEncoder, swag_from
import tensorflow as tf
import pandas as pd
from keras.initializers import Orthogonal
from keras.layers import LSTM
import pickle, re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import string

app = Flask(__name__)
app.json_encoder = LazyJSONEncoder

swagger_template = {
    'info': {
        'title': 'API Documentation for Data Processing and Modeling',
        'version': '1.0.0',
        'description': 'Dokumentasi API untuk Data Processing dan Modeling',
    },
    'host': '127.0.0.1:5000'
}

swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'docs',
            "route": '/docs.json',
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}
swaggerr = Swagger(app, template=swagger_template, config=swagger_config)

max_features = 100000
tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)

sentimen = ['negative', 'neutral', 'positive']

df_kbbi = pd.read_csv('KamusAlayIvan.csv', header=None, encoding='ISO-8859-1', names=['TIDAKBAKU', 'BAKU'])

def removechars(text):
    text = re.sub(r'[^\w]', ' ', text)
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))','',text)
    text = re.sub('\d', '', text)
    text = re.sub(r'\b\w\b', '', text)
    stopwords = ['di', 'yang', 'dan', 'nya', 'saya', 'ini', 'itu', 'aku', 'kamu', 'th', 'sekali']
    for stopword in stopwords:
        text = re.sub(r'\b{}\b'.format(stopword), '', text)
    return text

def changealay(text):
    alay = dict(zip(df_kbbi['TIDAKBAKU'], df_kbbi['BAKU']))
    text = ' '.join([alay[word] if word in alay else word for word in text.split(' ')])
    return text

def cleansing(text):
    text = removechars(text)
    text = changealay(text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Load models and tokenizer NN|TFIDF
with open(r"resoure_of_tfidf\tfidf_vect.pkl", 'rb') as file:
    feature_file_from_nn = pickle.load(file)

with open('model_of_tfidf\model.p', 'rb') as file:
    model_file_from_nn = pickle.load(file)

# Load models and tokenizer LSTM
with open("tokenizer.pickle", 'rb') as file:
    tokenizer = pickle.load(file)

model_lstm = load_model('model.h5')

# Endpoint for NN prediction
@swag_from("docs/nn.yml", methods=['POST'])
@app.route('/nn', methods=['POST'])
def nn():
    original_text = request.form.get('text')
    text = [cleansing(original_text)]
    feature = feature_file_from_nn.transform(text)
    prediction = model_file_from_nn.predict(feature)[0]

    json_response = {
        'status_code': 200,
        'description': "Result of Sentiment Analysis using NN",
        'data': {
            'text': original_text,
            'sentiment': prediction.tolist()  # Ensure the prediction is JSON serializable
        },
    }
    return jsonify(json_response)

@swag_from(r"docs\nn_file.yml", methods=['POST'])
@app.route('/nn_file', methods=['POST'])
def text_processing_file():
    # Uploaded file
    file = request.files.getlist('file')[0]

    # Import file csv ke Pandas
    df = pd.read_csv(file)

    # Ambil teks yang akan diproses dalam format list
    texts = df.text.to_list()

    # Lakukan cleansing pada teks
    cleaned_text = []
    label = []
    for input_text in texts:
        cleaned_text.append(cleansing(input_text))
        text = [cleansing(input_text)]
        feature = feature_file_from_nn.transform(text)

        prediction = model_file_from_nn.predict(feature)[0]
        polarity = np.argmax(prediction)
        label.append(sentimen[polarity])

    json_response = {
        'status_code': 200,
        'description': "Teks yang sudah diproses",
        'data_text': texts,
        'data_sentiment': label
    }

    response_data = jsonify(json_response)
    return response_data


# Endpoint for LSTM prediction
@swag_from("docs/lstm.yml", methods=['POST'])
@app.route('/lstm', methods=['POST'])
def lstm():
    original_text = request.form.get('text')
    text = [cleansing(original_text)]
    predicted = tokenizer.texts_to_sequences(text)
    maxlen = 100  # Use a predefined maxlen based on your model's training
    guess = pad_sequences(predicted, maxlen=maxlen)
    
    prediction = model_lstm.predict(guess)
    polarity = np.argmax(prediction[0])
    
    json_response = {
        'status_code': 200,
        'description': "Result of Sentiment Analysis using LSTM",
        'data': {
            'text': original_text,
            'sentiment': sentimen[polarity]
        },
    }
    return jsonify(json_response)

# Endpoint for text processing from file
@swag_from("docs/lstm_file.yml", methods=['POST'])
@app.route('/lstm_file', methods=['POST'])
def lstm_file():
    # Uploaded file
    file = request.files.getlist('file')[0]

    # Import file csv ke Pandas
    df = pd.read_csv(file)

    # Ambil teks yang akan diproses dalam format list
    texts = df.text.to_list()

    # Lakukan cleansing pada teks
    cleaned_text = []
    label = []
    for input_text in texts:
        cleaned_text.append(cleansing(input_text))  # Cleansed text added here
        text = [cleansing(input_text)]
        predicted = tokenizer.texts_to_sequences(text)
        maxlen = 100  # Ensure consistent maxlen
        guess = pad_sequences(predicted, maxlen=maxlen)

        prediction = model_lstm.predict(guess)
        polarity = np.argmax(prediction[0])
        label.append(sentimen[polarity])

    json_response = {
        'status_code': 200,
        'description': "Teks yang sudah diproses",
        'data_text': texts,
        'data_sentiment': label
    }

    return jsonify(json_response)

##running api
if __name__ == '__main__':
   app.run()
