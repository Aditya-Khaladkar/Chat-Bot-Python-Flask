import tensorflow as tf
import numpy as np
import pandas as pd
import json
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Flatten, GlobalMaxPooling1D
from tensorflow.keras.models import Model

with open('data.json') as database:
    data1 = json.load(database)
tags = []
inputs = []
responses = {}
for intent in data1['intents']:
    responses[intent['tag']] = intent['responses']
    for lines in intent['patterns']:
        inputs.append(lines)
        tags.append(intent['tag'])
data = pd.DataFrame({"input patterns": inputs, 'tags': tags})
data = data.sample(frac=1)
import string

data['input patterns'] = data['input patterns'].apply(
    lambda wrd: [ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data['input patterns'] = data['input patterns'].apply(lambda wrd: ''.join(wrd))

from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data['input patterns'])
train = tokenizer.texts_to_sequences(data['input patterns'])
from tensorflow.keras.preprocessing.sequence import pad_sequences

x_train = pad_sequences(train)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train = le.fit_transform(data['tags'])
input_shape = x_train.shape[1]
print(input_shape)
vocabulary = len(tokenizer.word_index)
print("Number of unique words : ", vocabulary)
output_length = le.classes_.shape[0]
print("Output length : ", output_length)
i = Input(shape=(input_shape,))
x = Embedding(vocabulary + 1, 10)(i)
x = LSTM(10, return_sequences=True)(x)
x = Flatten()(x)
x = Dense(output_length, activation="softmax")(x)
model = Model(i, x)
model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
train = model.fit(x_train, y_train, epochs=200, batch_size=5, verbose=1)
model.save('model.h5', train)
print("model created")
from tensorflow import keras
from keras.models import load_model

model = load_model('model.h5')
import random
import requests
import string
from tensorflow.keras.preprocessing.text import Tokenizer

c = " "
hotelId = 0

url = "https://booking-com.p.rapidapi.com/v1/hotels/search"
headers = {
    "X-RapidAPI-Host": "booking-com.p.rapidapi.com",
    "X-RapidAPI-Key": "85926a460emsh91319b0ae6a051ap188c86jsn74155e259fc6"
}


def getResponse(msg):
    hotellist = " "
    hotelFacilities = " "
    destid = ""
    global c
    global hotelId
    global url, headers
    while True:
        texts_p = []
        prediction_input = msg
        prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
        prediction_input = ''.join(prediction_input)
        texts_p.append(prediction_input)
        prediction_input = tokenizer.texts_to_sequences(texts_p)
        prediction_input = np.array(prediction_input).reshape(-1)
        prediction_input = pad_sequences([prediction_input], input_shape)
        output = model.predict(prediction_input)
        output = output.argmax()
        response_tag = le.inverse_transform([output])[0]
        res = random.choice(responses[response_tag])
        querystring = {"checkout_date": "2022-10-01", "units": "metric", "dest_id": "", "dest_type": "city",
                       "locale": "en-gb", "adults_number": "2", "order_by": "popularity", "filter_by_currency": "AED",
                       "checkin_date": "2022-09-30", "room_number": "1", "children_number": "2", "page_number": "0",
                       "children_ages": "5,0", "categories_filter_ids": "class::2,class::4,free_cancellation::1",
                       "include_adjacency": "true"
                       }
        global data
        if response_tag == 'city':
            c = msg
            if (c.lower()) == 'pune' or (c.lower()) == 'pnq':
                querystring.update({"dest_id": "-2108361"})

            response = requests.request("GET", url, headers=headers, params=querystring)


            data = response.json()
            print(type(data))

            k = f"---------- Top 5 hotels in {c} ----------\n"

            for i in range(5):
                hotel = data["result"][i]["hotel_name"]
                hotellist += hotel
                print(i + 1, "", hotel)
                res = hotellist
        if response_tag == "no":
            url1 = "https://booking-com.p.rapidapi.com/v1/hotels/facilities"

            if msg == "1":
                hotelId = data["result"][0]["hotel_id"]
                querystring1 = {"locale": "en-gb", "hotel_id": hotelId}
                print(hotelId)
                response = requests.request("GET", url1, headers=headers, params=querystring1)
                print(response.text)

        return res


import requests


def chatbot_response(msg):
    result = getResponse(msg)
    return result


from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)


if __name__ == "__main__":
    app.run()
