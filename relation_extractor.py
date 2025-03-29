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

Relation = {
    1: "Schools_Attended",
    2: "Work_For",
    3: "Live_In",
    4: "Top_Member_Employees"
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

def extract_relations_with_gemini(text, relation_id, gemini_key):

    doc = nlp(text)  
    se_count = 0
    total=0
    for s in doc.sents:
        total = total + 1
    print("\tExtracted ", total, " sentences. Prompting Gemini on each sentence ...")
    
    
    target_relation = RELATION_TYPES[relation_id]
    print(f"\tTarget relation type: {target_relation}")

    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    results = []

    if relation_id == 1:
        example_sentence = "Jeff Bezos graduated from Princeton University."
        example_output = "[('Jeff Bezos', 'Princeton University')]"
    elif relation_id == 2:
        example_sentence = "Alec Radford works at OpenAI as a researcher."
        example_output = "[('Alec Radford', 'OpenAI')]"
    elif relation_id == 3:
        example_sentence = "Mariah Carey lives in New York City."
        example_output = "[('Mariah Carey', 'New York City')]"
    elif relation_id == 4:
        example_sentence = "Nvidia's CEO, Jensen Huang, announced the new GPU architecture."
        example_output = "[('Nvidia', 'Jensen Huang')]"
    else:
        raise ValueError("Invalid relation_id.")

    prompt_header = (
        f"You are a relation extraction system. In this thread, I will give you a bunch of sentences, which contain desired entities, your task is to extract the subject-object pairs for the relation type: {Relation[relation_id]} (internal name: {RELATION_TYPES[relation_id]} in spaCy).\n"
        f"For example,"
        f"Sentence: {example_sentence}\n"
        f"Output: {example_output}\n\n"
        f"Now extract subject-object pairs from the sentence below.\n"
        f"Return only a list of (subject, object) tuples in this exact format: ('SUBJECT', 'OBJECT').\n"
        f"If there are no valid tuples, respond with 'None'.\n"
    )

    for sentence in doc.sents: 
        se_count = se_count + 1
        if int(se_count) % 5 == 0:
            print('\tProcessed ' + str(se_count) + '/' + str(total) + ' sentences')
        takegemini=False
        sentence_entity_pairs = create_entity_pairs(sentence, entities_of_interest)
        for ep in sentence_entity_pairs:
            if relation_id in [1,2,4] and ((ep[1][1] == 'PERSON' and ep[2][1] == 'ORGANIZATION') or (ep[2][1] == 'PERSON' and ep[1][1] == 'ORGANIZATION')):
                takegemini=True
                break
            elif relation_id ==3 and (ep[1][1] == 'PERSON' and ep[2][1] in ["LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]) or ((ep[2][1] == 'PERSON' and ep[1][1] in ["LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"])) :
                takegemini=True
                break

        if takegemini:
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
