import spacy
import sys
sys.path.append("./SpanBERT")  # Ensure SpanBERT directory is in the path
from spanbert import SpanBERT
from spacy_help_functions import get_entities, create_entity_pairs, extract_relations
import os
import google.generativeai as genai
import time
from google import genai
import re


# Load spaCy model
nlp = spacy.load("en_core_web_lg")

# Load pre-trained SpanBERT model
SPANBERT_PATH = "./pretrained_spanbert"
spanbert = SpanBERT(SPANBERT_PATH)
assert os.path.exists(SPANBERT_PATH), "SpanBERT model path does not exist!"
# Mapping relation types
entities_of_interest = ["ORGANIZATION", "PERSON", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]
RELATION_TYPES = {
    1: "per:schools_attended",
    2: "per:employee_of",
    3: "per:cities_of_residence",
    4: "org:top_members/employees"
}


def extract_relations_with_spanbert(text, t,final_ans, relation_type):
    if relation_type not in RELATION_TYPES:
        raise ValueError("Invalid relation type. Choose from 1-4.")
    
    doc = nlp(text)  
    total = 0
    se_count = 0
    for s in doc.sents:
        total = total + 1
    print("\tExtracted ", total, " sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ...")
    before=len(final_ans)
    for sentence in doc.sents: 

        se_count = se_count + 1
        if int(se_count) % 5 == 0:
            print('\tProcessed ' + str(se_count) + '/' + str(total) + ' sentences')
        relation_preds=[]
        ents = get_entities(sentence, entities_of_interest)
        candidate_pairs = []
        sentence_entity_pairs = create_entity_pairs(sentence, entities_of_interest)
        for ep in sentence_entity_pairs:
				
            if (ep[1][1] == 'PERSON' and ep[2][1] == 'ORGANIZATION') or (ep[2][1] == 'PERSON' and ep[1][1] == 'ORGANIZATION'):
                candidate_pairs.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})  
                candidate_pairs.append({"tokens": ep[0], "subj": ep[2], "obj": ep[1]}) 
            if (ep[1][1] == 'PERSON' and ep[2][1] in ["LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]) or ((ep[2][1] == 'PERSON' and ep[1][1] in ["LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"])) :
                candidate_pairs.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})  
                candidate_pairs.append({"tokens": ep[0], "subj": ep[2], "obj": ep[1]})
        candidate_pairs = [p for p in candidate_pairs if p["subj"][1] in ["PERSON", "ORGANIZATION"]]
        candidate_pairs = [p for p in candidate_pairs if p["obj"][1] in ["ORGANIZATION", "PERSON", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]]  
      
        if len(candidate_pairs) == 0:
            continue
        relation_preds = spanbert.predict(candidate_pairs)  
       
        #dic order dict: {key: (subj, obj, rel), value = (confi, token: [])}
        for ex, pred in list(zip(candidate_pairs, relation_preds)):
            if pred[0] == RELATION_TYPES[relation_type] and pred[1]>=t :
                key= (ex["subj"][0], ex["obj"][0], pred[0])
                value = (pred[1],ex["tokens"])
                print(f"\t=== Extracted Relation ===")
                print(f"\tInput Token: {ex["tokens"]}")
                print(f"\tSubject: {ex["subj"][0]} | Object: {ex["obj"][0]} | Confidence: {pred[1]}")
                if key not in final_ans.keys(): 
                    final_ans[key]=value
                    print(f"\tAdding to set of extracted relations")
                    print()
                else: 
                    if value[0] > final_ans[key][0]:
                        final_ans[key]=value
                    else:
                        print(f"\tDuplicate with lower confidence than existing record. Ignoring this.")
                        print()
    after=len(final_ans)  
    print(f"\t Extracted annotations total {total} sentences")
    print(f"\t Relations extracted from this website: {after-before} (Overall: {after})")

    return final_ans


def extract_relations_with_gemini(sentences, relation_id, gemini_key):
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    # Define relations for specific relations if needed
    relation_templates = {
        1: "per:schools_attended",
        2: "per:employee_of",
        3: "per:cities_of_residence",
        4: "org:top_members/employees",
        
    }

    relation = relation_templates.get(relation_id, "unknown")
    extracted = []

    for sent in sentences:
        # Make prompt specific to the kind of relations we want
        prompt = (
            f"Extract pairs of entities (subject, object) where the subject is a person and the object is an organization "
            f"from the following sentence. The relation should be 'works for'. Return only meaningful pairs in the format "
            f"[(subject, object)].\n"
            f"Sentence: {sent}\n"
            f"Relation: works for\n"
            f"Expected Format: [(subject, object)]"
        )

        try:
            response = model.generate_content(prompt)
            result = response.text.strip()

            # Match tuples like ('X', 'Y') or ("X", "Y")
            pattern = re.findall(r'\(["\'](.*?)["\'],\s*["\'](.*?)["\']\)', result)
            for subj, obj in pattern:
                # Clean up any additional whitespace and assign a default confidence score (1.0)
                extracted.append((subj.strip(), obj.strip(), 1.0))
        except Exception as e:
            print(f"        Error calling Gemini API: {e}")
    return extracted
