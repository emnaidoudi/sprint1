from watson_developer_cloud import LanguageTranslatorV3
from nlp import spell_correction, tokenization
import requests

def ibm_watson_translation(sentence):
    to=tokenization(sentence)
    s=list()
    for i in to:
        s.append(spell_correction(i))
    sentence=' '.join(s)    
    language_translator = LanguageTranslatorV3(
    version='2018-05-01',
    iam_apikey='6N2fgkLzRaDtj7vM90uDNYiBZY8rxYRdfzXKvQ0vqio9',
    url='https://gateway-lon.watsonplatform.net/language-translator/api' )    
    translation = language_translator.translate(
    text=sentence,
    model_id='en-fr').get_result()
    return translation["translations"][0]["translation"]


def get_wheather(loc="tunis"):
    # print("inin %s" %(loc))
     r = requests.get('http://api.openweathermap.org/data/2.5/weather?q=%s&units=metric&APPID=acf0a678438a992a21999196194f42c0'%(loc))
     j=r.json()
     x="Temperature : %s°C, %s "  % (int(j['main']["temp_max"]),j["weather"][0]["description"])
     return  x

   
