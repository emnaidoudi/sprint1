from nlp import bag_of_words, vocab_ready
from spellchecker import SpellChecker
#from model import train_model
from framework import response_intents
from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/api/chatbot/basic/<string:sentence>")
def basic(sentence):
    return jsonify({"response":response_intents(sentence)})

print("master")

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')