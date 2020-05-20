import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer

import numpy as np
# import tflearn
import tensorflow as tf
import random
import pickle

import json


def chatbot(inp):

    stemmer = LancasterStemmer()

    with open('intents.json') as file:
        data = json.load(file)

    try:
        with open("data.pickle","rb") as f:
            words, labels, training, output = pickle.load(f)

    except:

        words = []
        labels = []
        docs_x = []
        docs_y = []

        for intent in data['intents']:
            for pattern in intent['patterns']:
                wrds = nltk.word_tokenize(pattern)
                words.extend(wrds)
                docs_x.append(wrds)
                docs_y.append(intent['tag'])

            if intent['tag'] not in labels:
                labels.append(intent['tag'])

        words = [stemmer.stem(w.lower()) for w in words if w not in ["?"]]
        words = sorted(list(set(words)))

        labels = sorted(labels)

        training = []
        output = []

        out_empty = [0 for _ in range(len(labels))]

        for x, doc in enumerate(docs_x):
            bag = []

            wrds = [stemmer.stem(w) for w in doc]

            for w in words:
                if w in wrds:
                    bag.append(1)
                else:
                    bag.append(0)

            output_row = out_empty[:]
            output_row[labels.index(docs_y[x])] = 1

            training.append(bag)
            output.append(output_row)

        training = np.array(training)
        output = np.array(output)

        with open("data.pickle","wb") as f:
            pickle.dump((words, labels, training, output), f)

    # tf.reset_default_graph()
    # print(training.shape)
    # print(output.shape)

    i = tf.keras.layers.Input(shape = len(training[0]))
    x  = tf.keras.layers.Dense(8, activation = 'relu')(i)
    x  = tf.keras.layers.Dense(8, activation = 'relu')(x)
    x = tf.keras.layers.Dense(len(output[0]), activation = 'softmax')(x)

    model = tf.keras.models.Model(i,x)

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    # print(model.summary())
    try:
        model = tf.keras.models.load_model("model.h5")
    except:
        model.fit(training, output, epochs=1000)
        model.save('model.h5')

    def bag_of_words(s, words):
        bag = [0 for _ in range(len(words))]

        s_words = nltk.word_tokenize(s)
        s_words = [stemmer.stem(word.lower()) for word in s_words]

        for se in s_words:
            for i, w in enumerate(words):
                if w == se:
                    bag[i] = (1)

        return np.array(bag).reshape(-1,len(training[0]))

    # def chat():
    #     print("Start talkign with the bot (type quit to stop)!")
    #     while True:
    #         inp = input("You: ")
    #         if inp.lower() == "quit":
    #             break

    results = model.predict(bag_of_words(inp, words))[0]
    results_index = np.argmax(results)
    tag = labels[results_index]

    if results[results_index] > 0.5:
        for tg in data['intents']:
            if tg['tag'] == tag:
                responses = tg['responses']

    else:
        for tg in data['intents']:
            if tg['tag'] == 'not_understood':
                responses = tg['responses']

    out = random.choice(responses)

    return out


    # chat()
