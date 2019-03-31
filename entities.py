import spacy

def spacy_entity(sentence):
    sentence=sentence.title()    
    all_extractions = list()
    # Load English tokenizer, tagger, parser, NER and word vectors
    nlp = spacy.load('en_core_web_sm')
    def entity_extraction():
        #sentence=sentence.upper()
        print(sentence)
        doc = nlp(sentence)
        for entity in doc.ents:
            all_extractions.append((entity.text.lower(), entity.label_))
    entity_extraction()        
    return all_extractions

    


  