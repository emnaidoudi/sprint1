import pickle
import json
import numpy as np
import tflearn
import tensorflow as tf
import random
from nlp import bag_of_words, stemming
from services import *
from entities import spacy_entity
from fromDatabase import get_manager

# import our chatbot intents file
with open('intents.json') as json_data:
    intents = json.load(json_data)
idata = pickle.load( open("training_data_intents", "rb" ) )
iwords = idata['words']
iclasses = idata['classes']
itrain_x = idata['train_x']
itrain_y = idata['train_y']
# import our chatbot entities file
with open('entities.json') as json_data:
    entities = json.load(json_data)
edata = pickle.load( open("training_data_entities", "rb" ) )
ewords = edata['words']
eclasses = edata['classes']
etrain_x = edata['train_x']
etrain_y = edata['train_y']

def build_nn(x,y,intent_or_entities):
    # Build neural network
    net = tflearn.input_data(shape=[None, len(x[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(y[0]), activation='softmax')
    net = tflearn.regression(net)
    # Define model and setup tensorboard
    model = tflearn.DNN(net, tensorboard_dir='tflearn_logs_%s'%(intent_or_entities))    
    # load our saved model
    #model.load('./model_%s.tflearn'%(intent_or_entities))
    return model
    
intent_model=build_nn(itrain_x,itrain_y,"intents")
intent_model.load('./model_intents.tflearn')
tf.reset_default_graph()
entities_model=build_nn(etrain_x,etrain_y,"entities")
entities_model.load('./model_entities.tflearn')


def classify(sentence,words,classes,model):
    ERROR_THRESHOLD = 0.25
    # generate probabilities from the model
    results = model.predict([bag_of_words(sentence, words)])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]  
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    # return tuple of intent and probability
    return_list = []
    for r in results:
        return_list.append((classes[r[0]]))
    # return tuple of intent and probability
    return  return_list

def entity_exact(sentence):
    all_entities=classify(sentence,ewords,eclasses,entities_model)
    print("all ",all_entities) 
    list_entities=list()
    for e in all_entities:
        for i in entities["entities"]:
            if i["tag"]==e:
                examples=i["patterns"]
                print("11" , examples)
                entities_stemmed=[stemming(word) for word in examples ]
                for i in entities_stemmed:
                    if(i in sentence):
                        x=entities_stemmed.index(i)
                        list_entities.append(examples[x])
                break 
    return list_entities           


# create a data structure to hold user context
context=""
def response(sentence):
    spacy_detections=spacy_entity(sentence)
    global context
    intent=classify(sentence,iwords,iclasses,intent_model)[0]
    if intent=="weather":
        loc="tunis"
        if(spacy_detections):
            loc=[i[0] for i in spacy_detections if 'GPE' in i][0]
        return get_wheather(loc)
    if intent=="askManager":
        try:
           group=entity_exact(sentence)[0]
           print("group ",group)
           year=[i[0] for i in spacy_detections if 'DATE' in i][0]
        except:  
            return get_manager("devops",'2019')
        return get_manager(group,year)  

    for i in intents["intents"]:
        if len(context)>0 and 'context_filter' in i:
            if i["context_filter"]==context: 
                context=""
                if i["context_filter"]=="ibm_translation":
                    return str(ibm_watson_translation(sentence))
                return random.choice(i["responses"])
         
        if i['tag']==intent:
            # set context for this intent if necessary
            if 'context_set' in i and context=="":
                context = i['context_set']
                print("coooooo",context)
                return random.choice(i["responses"])
            if context=="":
                return random.choice(i["responses"])
    return "I didn't understand what you're saying, sorry"            




            





















# import our chatbot entities file
"""with open('entities.json') as json_data:
    data = json.load(json_data)
edata = pickle.load( open( "training_data_entities", "rb" ) )
ewords = data['words']
eclasses = data['classes']
etrain_x = data['train_x']
etrain_y = data['train_y']"""