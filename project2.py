import sys
import json
import requests
from google_search import perform_google_search
from webpage_retriever import retrieve_webpage
from text_processor import extract_entities
from relation_extractor import extract_relations_with_spanbert
from result_selector import select_top_k_tuples, remove_duplicates

def main():
    # Ensure the correct number of arguments
    if len(sys.argv) != 9:
        print("Usage: python project2.py [-spanbert|-gemini] <google api key> <google engine id> <google gemini key> <r> <t> <q> <k>")
        sys.exit(1)
    
    model = sys.argv[1]
    api_key = sys.argv[2]
    engine_id = sys.argv[3]
    gemini_key = sys.argv[4]
    r = int(sys.argv[5])  
    t = float(sys.argv[6])  
    q = sys.argv[7]  
    k = int(sys.argv[8])  
    
    print(f"Parameters:\nClient key = {api_key}\nEngine key = {engine_id}\nGemini key = {gemini_key}")
    print(f"Method = {model}\nRelation = {r}\nThreshold = {t}\nQuery = {q}")
    print(f"# of Tuples = {k}")
    print("Loading necessary libraries; This should take a minute or so ...\n")
    
    # Step 1: Perform Google Custom Search
    print(f"=========== Iteration: 0 - Query: {q} ===========")
    urls = perform_google_search(api_key, engine_id, q)
    
    
    extracted_tuples = set()
    
    for idx, url in enumerate(urls[:k]):
        print(f"URL ({idx+1} / {len(urls)}): {url}")
        webpage_text = retrieve_webpage(url)
        if webpage_text:
            #print(f"\tFetching text from url ...")
            trimmed_text = webpage_text[:10000]  # Trimming to 10000 characters
            print(f"\tTrimming webpage content from {len(webpage_text)} to {len(trimmed_text)} characters")
            print(f"\tWebpage length (num characters): {len(trimmed_text)}")
            
            #print("\tAnnotating the webpage using spacy...")
            
            #print(f"\tExtracted {len(trimmed_text)} sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ...")
            
            # Process sentences and extract relations
            if model == "-spanbert":
                relations = extract_relations_with_spanbert(trimmed_text, r, t)
            #elif model == "-gemini":
            #    relations = extract_relations_with_gemini(trimmed_text, r, gemini_key)
            
            # Add relations to set
            for subj, obj, rel, conf in relations:
                if conf >= 0.5:  # Explicit threshold check
                    print(f"\t=== Extracted Relation ===")
                    print(f"\tRelation Type: {rel}")
                    print(f"\tSubject: {subj} | Object: {obj} | Confidence: {conf:.2f}")
                    print("\tAdding to set of extracted relations")
                    extracted_tuples.add((subj, obj, rel, conf))

    # Step 2: Remove duplicates
    #print(f"\nProcessed {len(extracted_tuples)} relations")
    extracted_tuples = remove_duplicates(extracted_tuples)

    # Step 3: Select top k tuples
    top_k_tuples = select_top_k_tuples(extracted_tuples, k, model)
    
    # Output top k relations
    #print("\nTop Relations:")
    #for tuple in top_k_tuples:
        #print(f"({tuple[0]}, {tuple[1]}, {tuple[2]}, {tuple[3]})")

if __name__ == '__main__':
    main()