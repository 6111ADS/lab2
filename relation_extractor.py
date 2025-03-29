import spacy
import sys
sys.path.append("./SpanBERT")  # Ensure SpanBERT directory is in the path
from spanbert import SpanBERT
from spacy_help_functions import get_entities, create_entity_pairs, extract_relations
import os
import google.generativeai as genai
import time
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
                print(f"\tInput Token: {ex['tokens']}")
                print(f"\tSubject: {ex['subj'][0]} | Object: {ex['obj'][0]} | Confidence: {pred[1]}")
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
    import google.generativeai as genai

    if relation_id not in RELATION_TYPES:
        raise ValueError("Invalid relation type. Choose from 1-4.")

    target_relation = RELATION_TYPES[relation_id]
    print(f"\tTarget relation type: {target_relation}")

    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    results = []
    num_sentences = len(sentences)

    print(f"\tExtracted {num_sentences} sentences. Prompting Gemini on each sentence ...")

    if relation_id == 1:
        # per:schools_attended
        prompt_header = (
            f"Extract only subject-object pairs for the relation type: {target_relation}.\n"
            f"The subject must be a person, and the object must be a school or educational institution.\n"
            f"Return only a list of (subject, object) tuples in this exact format.\n"
            f"If there are no valid tuples, respond with 'None'.\n"
        )
    elif relation_id == 2:
        # per:employee_of
        prompt_header = (
            f"Extract only subject-object pairs for the relation type: {target_relation}.\n"
            f"The subject must be a person, and the object must be a company or organization they work or worked for.\n"
            f"Return only a list of (subject, object) tuples in this exact format.\n"
            f"If there are no valid tuples, respond with 'None'.\n"
        )
    elif relation_id == 3:
        # per:cities_of_residence
        prompt_header = (
            f"Extract subject-object pairs for the relation type: {target_relation}.\n"
            f"The subject must be a person, and the object must be a city or place where the person lives or lived.\n"
        
            f"Return only (subject, object) tuples in that exact format.\n"
            f"If there are no valid tuples, respond with 'None'.\n"
        )
    elif relation_id == 4:
        # org:top_members/employees
        prompt_header = (
            f"Extract subject-object pairs for the relation type: {target_relation}.\n"
            f"The subject must be an organization, and the object must be a person who is or was a top member or employee.\n"
            f"Return only (subject, object) tuples in that exact format.\n"
            f"If there are no valid tuples, respond with 'None'.\n"
        )
    else:
        raise ValueError("Invalid relation_id.")



    for i, sentence in enumerate(sentences):
        if (i + 1) % 5 == 0 or i == 0:
            print(f"\tProcessed {i+1}/{num_sentences} sentences")

        prompt = prompt_header + f"Sentence: {sentence}"

        try:
            response = model.generate_content(prompt)
            reply = response.text.strip()

            if not reply or reply.lower() == "none":
                continue

            matches = re.findall(r"\('(.*?)',\s*'(.*?)'\)", reply)
            for subj, obj in matches:
                if subj and obj and (subj, obj) not in results:
                    print(f"\t=== Extracted Relation ===")
                    print(f"\tInput Sentence: {sentence}")
                    print(f"\tSubject: {subj} | Object: {obj}")
                    print(f"\tAdding to set of extracted relations\n")
                    results.append((subj, obj))

        except Exception:
            continue

    print(f"\tTotal relations extracted with Gemini: {len(results)}")
    return results

