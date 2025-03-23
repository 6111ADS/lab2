import spacy
import sys
sys.path.append("./SpanBERT")  # Ensure SpanBERT directory is in the path
from spanbert import SpanBERT
from spacy_help_functions import get_entities, create_entity_pairs, extract_relations
import os
import google.generativeai as genai
import time
from google import genai


# Load spaCy model
nlp = spacy.load("en_core_web_lg")

# Load pre-trained SpanBERT model
SPANBERT_PATH = "./pretrained_spanbert"
spanbert = SpanBERT(SPANBERT_PATH)
assert os.path.exists(SPANBERT_PATH), "SpanBERT model path does not exist!"
# Mapping relation types
RELATION_TYPES = {
    1: "per:schools_attended",
    2: "per:employee_of",
    3: "per:cities_of_residence",
    4: "org:top_members/employees"
}

def extract_relations_with_spanbert(text, relation_type, threshold=0.5):
    if relation_type not in RELATION_TYPES:
        raise ValueError("Invalid relation type. Choose from 1-4.")
    
    entities_of_interest = ["ORGANIZATION", "PERSON", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]
    doc = nlp(text)  
    res=[]
   
    for sentence in doc.sents:  
        ents = get_entities(sentence, entities_of_interest)
        candidate_pairs = []
        sentence_entity_pairs = create_entity_pairs(sentence, entities_of_interest)
        for ep in sentence_entity_pairs:
            if ep[1][1] == 'PERSON' and ep[2][1] == 'ORGANIZATION':
                candidate_pairs.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})  
            if ep[2][1] == 'PERSON' and ep[1][1] == 'ORGANIZATION':
                candidate_pairs.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})
            if ep[1][1] == 'PERSON' and ep[2][1] in ["LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]:
                candidate_pairs.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]}) 

        candidate_pairs = [p for p in candidate_pairs if p["subj"][1] in ["PERSON", "ORGANIZATION"]]
        candidate_pairs = [p for p in candidate_pairs if p["obj"][1] in ["ORGANIZATION", "PERSON", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]]
        
        if len(candidate_pairs) == 0:
            continue
        
        relation_preds = spanbert.predict(candidate_pairs)  

        # Print Extracted Relations
        print("\nExtracted relations:")
        
        for ex, pred in list(zip(candidate_pairs, relation_preds)):
            print("\tSubject: {}\tObject: {}\tRelation: {}\tConfidence: {:.2f}".format(ex["subj"][0], ex["obj"][0], pred[0], pred[1]))
            res.append((ex["subj"][0], ex["obj"][0], pred[0], pred[1]))
    
    return res
  
def extract_relations_with_gemini(trimmed_text, r, gemini_key, final_ans):
    client = genai.Client(api_key=gemini_key)
    sentences = trimmed_text.split('.')  # Splitting into sentences
    batch_size = 5  # Send 5 sentences at a time
    read=0
    # Define the prompt outside the loop for efficiency
    prompt = ("Identify relationships in the following text that match the specified relation type. "
              "For example, the sentence: Bill Gates stepped down as chairman of Microsoft in February 2014 "
              "and assumed a new post as technology adviser to support the newly appointed CEO Satya Nadella. "
              "You should have the relations: ('Bill Gates', 'per:employee_of', 'Microsoft'), "
              "('Microsoft', 'org:top_members/employees', 'Bill Gates'), "
              "('Satya Nadella', 'per:employee_of', 'Microsoft'). "
              f"You only need to return the relations that match {RELATION_TYPES[r]}\n\n")
    
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        batch_prompt = prompt + "\n".join(batch)
        read+=5
        print("I have read ", read, "sentences")
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash", contents=batch_prompt
            )
            output_text = response.text
            
            for line in output_text.split('\n'):
                line = line.strip()
                #print(line)
                if line.startswith("*"):

                    start = line.find("(") + 1
                    end = line.find(")")
                    relation_content = line[start:end]
                    parts = relation_content.split(",")
                    
                    if len(parts) == 3:
                        subj = parts[0].strip().strip("'")
                        obj = parts[2].strip().strip("'")
                        if (subj, obj) not in final_ans:
                            
                            final_ans.append((subj, obj))
                            for sent in batch:
                                if subj in sent and obj in sent:
                                    break
        except Exception as e:
            if 'RESOURCE_EXHAUSTED' in str(e):
                print("Quota exceeded, waiting before retrying...")
                time.sleep(10)  # Wait for 60 seconds before retrying
                continue  # Retry the function
            else:
                print(f"Error processing batch: {e}")
     
    return final_ans
