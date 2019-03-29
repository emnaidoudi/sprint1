from nlp import bag_of_words, vocab_ready
from spellchecker import SpellChecker
#from model import train_model
from framework import response_intents
from flask import Flask, jsonify

print(response_intents("hello"))

print(response_intents("thaks"))
print(response_intents("bye"))

