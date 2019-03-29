import nltk
from nltk.stem.lancaster import LancasterStemmer # for english
from nltk.stem.snowball import FrenchStemmer # for french
from nltk.corpus import stopwords 
from textblob import TextBlob #to detect the language used => 'fr' or 'en'
from spellchecker import SpellChecker
import random
import numpy as np

"""
takes a sentence(a string) and a Vocabulary(list of word) and return the Bow of this sentence 
"""
def tokenization(sentence):
    return nltk.word_tokenize(sentence)
def spell_correction(word):
    spell = SpellChecker()
    misspelled=spell.unknown([word])
    if word in misspelled:
           word=spell.correction(word)
    return word
def stopwords_list():
    return list(set(stopwords.words('french')))+ list(set(stopwords.words('english')))+["?","!",".",";",","]
stopwords=stopwords_list()
def detect_language(sentence):
    if(len(sentence)>3):
        b = TextBlob(sentence)
        return b.detect_language()  
    else:
        return "en"
def stemming(word):
    word=word.lower()
    """lang=detect_language(word)
    stemmer = LancasterStemmer() if lang=="en" else FrenchStemmer()"""
    stemmer=LancasterStemmer()
    return stemmer.stem(word)
def get_ready_for_bow(sentence):
    #stopwords=stopwords_list()
    sentence_tokenized=tokenization(sentence)
    tokenized_sentence_ready = [stemming(spell_correction(w)) for w in sentence_tokenized if w not in stopwords]
    return tokenized_sentence_ready    
# to import ===>  from nlp import bag_of_words, vocab_ready
def vocab_ready(words):
    #stopwords=stopwords_list()
    vocab=list()
    vocab=[stemming(spell_correction(w)) for w in words if w not in stopwords]
    vocab = sorted(list(set(vocab)))
    return vocab
def bag_of_words(sentence,vocab_ready):
    bag=list()
    tokenized_sentence_ready=get_ready_for_bow(sentence)
    for w in vocab_ready :
        bag.append(tokenized_sentence_ready.count(w)) 
    return bag

