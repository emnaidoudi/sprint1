import pickle
import json
import numpy as np
import tflearn
import tensorflow as tf
import random
from nlp import bag_of_words

# import our chatbot intents file
with open('intents.json') as json_data:
    intents = json.load(json_data)

idata = pickle.load( open("training_data_intents", "rb" ) )
iwords = idata['words']
iclasses = idata['classes']
itrain_x = idata['train_x']
itrain_y = idata['train_y']

# Build neural network
net = tflearn.input_data(shape=[None, len(itrain_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(itrain_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model_intent = tflearn.DNN(net, tensorboard_dir='tflearn_logs_intents')    

# load our saved model
model_intent.load('./model_intents.tflearn')



def classify(sentence,words,classes):
    ERROR_THRESHOLD = 0.25
    # generate probabilities from the model
    results = model_intent.predict([bag_of_words(sentence, words)])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    # return tuple of intent and probability
    return  classes[results[0][0]]


def response_intents(sentence):
    intent=classify(sentence,iwords,iclasses)  
    for i in intents["intents"]:
        if i['tag']==intent:
            return random.choice(i["responses"])





















# import our chatbot entities file
"""with open('entities.json') as json_data:
    data = json.load(json_data)
edata = pickle.load( open( "training_data_entities", "rb" ) )
ewords = data['words']
eclasses = data['classes']
etrain_x = data['train_x']
etrain_y = data['train_y']"""