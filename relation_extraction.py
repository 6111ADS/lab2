# relation_extraction.py
import spacy
import google.generativeai as genai
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load spaCy model
nlp = spacy.load("en_core_web_lg")

# Load SpanBERT model
spanbert_model = AutoModelForSequenceClassification.from_pretrained("SpanBERT/model")
spanbert_tokenizer = AutoTokenizer.from_pretrained("SpanBERT/tokenizer")

def extract_entities(text):
    """
    Extract named entities from text using spaCy.
    """
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def extract_relations_spanbert(text, relation):
    """
    Extract relations using SpanBERT.
    """
    doc = nlp(text)
    results = []
    
    for ent1 in doc.ents:
        for ent2 in doc.ents:
            if ent1 == ent2:
                continue

            input_text = f"[CLS] {ent1.text} [SEP] {ent2.text} [SEP] {text}"
            tokens = spanbert_tokenizer(input_text, return_tensors="pt")
            with torch.no_grad():
                logits = spanbert_model(**tokens).logits

            confidence = torch.nn.functional.softmax(logits, dim=1)[0][1].item()
            results.append((ent1.text, ent2.text, confidence))

    return sorted(results, key=lambda x: -x[2])

def extract_relations_gemini(text, relation, gemini_api_key):
    """
    Extract relations using Gemini API.
    """
    genai.configure(api_key=gemini_api_key)

    prompt = f"""Extract a relation of type {relation} from the text below. 
    If there are multiple instances, list them all.
    Text: {text}"""
    
    try:
        response = genai.generate_text(prompt)
        return [tuple(r.strip().split(",")) for r in response.text.split("\n") if r]
    except Exception as e:
        print(f"Gemini API error: {e}")
        return []
