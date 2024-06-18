import pandas as pd
import re
from flask import Flask, jsonify
from flask import request
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from
import demoji
import sqlite3

app = Flask(__name__)

df_kbbi = pd.read_csv('new_kamusalay.csv', header=None, encoding='ISO-8859-1', names=['TIDAKBAKU', 'BAKU'])
df_kbbi

ABS = pd.read_csv('abusive.csv')
ABS


def remove_emoji(text):
    dem = demoji.findall(text)
    for demoj in dem.keys():
        text = text.replace(demoj, ' ')
    return text

def lowercase(text):
    return text.lower()

def removechars(text):
    text = re.sub(r'[^\w]', ' ', text)
    text = re.sub('rt',' ',text) 
    text = re.sub('user',' ',text) 
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',' ',text)
    text = re.sub('xf0',' ',text) 
    text = re.sub('x9f',' ',text) 
    text = re.sub('x98',' ',text) 
    text = re.sub('x82',' ',text) 
    text = re.sub('x84',' ',text) 
    text = re.sub('x86',' ',text) 
    text = re.sub('x8f',' ',text) 
    text = re.sub('xa4',' ',text)
    text = re.sub('xa2',' ',text)
    text = re.sub('x8b',' ',text)
    return text

def sensor(text):
    abusiveword = ABS['ABUSIVE'].tolist()
    for word in abusiveword:
        pattern = re.compile(r'\b{}\b'.format(word))
        length = len(word)
        replacement = '*' * length
        text = pattern.sub(replacement, text.lower())
    return text

def changealay(text):
    alay = dict(zip(df_kbbi['TIDAKBAKU'], df_kbbi['BAKU']))
    text = ' '.join([alay[word] if word in alay else word for word in text.split(' ')])
    return text


def cleaning(text):
    text = remove_emoji(text)
    text = removechars(text)
    text = lowercase(text)
    text = changealay(text)
    text = sensor(text)
    return text

app.json_encoder = LazyJSONEncoder

swagger_template = {
	"swagger":"2.0",
	"info":{
		"title": "API Documentation for Data Processing and Modelling",
		"description": "Dokumentasi API untuk Data Processing dan Modelling",
		"version": "1.0.0"
	}
}

swagger_config = {
	"headers" : [],
	"specs" : [
		{
			"endpoint" : 'docs',
			"route" : '/docs.json'
		}
	],
	"static_url_path" : '/flasgger_statis',
	"swagger_ui" : True,
	"specs_route" : "/docs/"
}

swagger = Swagger(app, config=swagger_config, template=swagger_template)

# endpoint default
@swag_from("hello_world.yml", methods=['GET'])
@app.route('/', methods=['GET'])
def hello_world():
    json_response = {
        'status_code': 200,
        'description': "Menyapa Hello World",
        'data': "Hello World",
    }

    response_data = jsonify(json_response)
    return response_data

# endpoint 1
@swag_from("text_processing.yml", methods=['POST'])
@app.route('/text_processing', methods=['POST'])
def text_clean():

    text = request.form.get('text')

    json_response = {
        'status_code': 200,
        'description': "Hasil teks yang sudah dibersihkan",
        'data': cleaning(text)
    }

    response_data = jsonify(json_response)
    return response_data

# endpoint 2
@swag_from("text_processing_file.yml", methods=['POST'])
@app.route('/text_processing_file', methods=['POST'])
def processing_file():

    # Uploaded file
    file = request.files.getlist('file')[0]

    # Import file csv ke Pandas
    df = pd.read_csv(file, encoding='latin-1')


    # Lakukan cleansing pada teks
    cleaned_text = []
    for text in df['Tweet']:
        cleaned_text.append(cleaning(text))

    # save ke sql
    conn = sqlite3.connect('datacsv_tosql1.db')
    df.to_sql('Datanew', conn, if_exists='replace', index=False)

    conn.close()

    json_response = {
        'status_code': 200,
        'description': "Teks yang sudah diproses",
        'data': cleaned_text
    }

    response_data = jsonify(json_response)
    return response_data

if __name__ == '__main__':
   app.run()
