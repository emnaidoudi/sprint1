from nlp import bag_of_words, vocab_ready
from spellchecker import SpellChecker
from model import train_model
from framework import response
from flask import Flask, jsonify,Response
from chatterbot import ChatBot
from entities import spacy_entity

#print(spacy_entity("paris is bea"))

app = Flask(__name__)

bot_answer=""
bot = ChatBot(
    'Math & Time Bot',
    logic_adapters=[
       'chatterbot.logic.MathematicalEvaluation',
        #'chatterbot.logic.TimeLogicAdapter'
   ]) 

@app.route("/api/chatbot/train_intents")
def train():
    train_model("intents.json")
    return Response(status=200)

@app.route("/api/chatbot/train_entities")
def train_en():
    train_model("entities.json")
    return Response(status=200)

@app.route("/api/chatbot/basic/<string:sentence>")
def basic(sentence):
    """try:
        bot_answer=str(bot.get_response(sentence))
        return jsonify({"response":"yaaaaaaa"})        
    except:"""
    return jsonify({"response":response(sentence)})


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')

#train_model("entities.json") """