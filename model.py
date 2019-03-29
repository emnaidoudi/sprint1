from nlp import bag_of_words, vocab_ready, tokenization
import json
import tflearn
import tensorflow as tf
import random
import pickle
import numpy as np

words = list()
classes = list()
documents = list()

def get_data(file):
    with open(file) as json_data:
        data = json.load(json_data)
    return data  

def extract_vocab(file): 
    data=get_data(file)
    global words , classes,  documents 
    for info in data[file.split(".")[0]]:
        for pattern in info['patterns']:
            w = tokenization(pattern)
            words.extend(w)
            documents.append((pattern, info['tag']))
            if info['tag'] not in classes:
                classes.append(info['tag'])  
    words=vocab_ready(words)
    classes = sorted(list(set(classes)))
 
def train_x_train_y(file):
    extract_vocab(file)
    training = []
    output_empty = [0] * len(classes)
    for doc in documents:
        bow=list()
        sentence = doc[0] 
        bow=bag_of_words(sentence,words)
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bow, output_row]) 
    random.shuffle(training)
    training = np.array(training)
    train_x = list(training[:,0])
    train_y = list(training[:,1])
    return train_x,train_y

def train_model(file):
    f=file.split(".")[0]
    train_x_train_y(file)
    train_x,train_y=train_x_train_y(file)
    tf.reset_default_graph()
    net = tflearn.input_data(shape=[None, len(train_x[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
    net = tflearn.regression(net)
    model = tflearn.DNN(net, tensorboard_dir='tflearn_logs_%s'%(f))
    model.fit(train_x,train_y, n_epoch=1000, batch_size=8, show_metric=True)
    model.save('model_%s.tflearn'%(f))  
    pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "training_data_%s"%(f), "wb" ) )  
