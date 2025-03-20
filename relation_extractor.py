import requests
import spacy
import sys
sys.path.append("./SpanBERT")  # Ensure SpanBERT directory is in the path
from spanbert import SpanBERT
from spacy_help_functions import extract_relations

nlp = spacy.load('en_core_web_lg')
SPANBERT_PATH = "./SpanBERT/pretrained_spanbert"
spanbert = SpanBERT(SPANBERT_PATH)

def extract_entities(text):
    """
    Extract named entities from the text.
    Returns a list of (sentence, [(entity, entity_type)]).
    """
    accepted=["ORG", "PERSON","GPE", "LOC", "DATE"]
    entities = set()
    doc = nlp(text)
    for sent in doc.sents:
        for ent in sent.ents:
            if ent.label_ in accepted: 
                entities.add(ent.label_)
     
    return entities

def extract_relations_with_spanbert(text, entities_of_interest, threshold=0.5):
    entities=list(extract_entities(text))
    doc = nlp(text)
    print(doc)
    print(entities)
    relations = extract_relations(doc, spanbert, entities)
    
    # Filter based on confidence threshold
    return [(subj, obj, rel, conf) for (subj, rel, obj), conf in relations.items() if conf >= threshold]










'''
def extract_relations_with_gemini(entities, relation_type, gemini_key):
    # Use Google Gemini API for relation extraction
    relations = []
    for sentence, entity_pairs in entities:
        for subj, obj in entity_pairs:
            # Construct the prompt and call Gemini API
            relation = classify_with_gemini(sentence, subj, obj, gemini_key)
            relations.append((subj, obj, relation, 1.0))  # Confidence is hardcoded to 1.0 for Gemini
    return relations

def classify_with_gemini(sentence, subj, obj, gemini_key):
    # Replace with actual Google Gemini API call
    # For now, returning a dummy example
    return 'per:employee_of'  # Example output'
'''