import sys
import json
import requests
from google_search import perform_google_search
from webpage_retriever import retrieve_webpage
from relation_extractor import extract_relations_with_spanbert, extract_relations_with_gemini
from result_selector import select_top_k_tuples, remove_duplicates
RELATION_TYPES = {
    1: "per:schools_attended",
    2: "per:employee_of",
    3: "per:cities_of_residence",
    4: "org:top_members/employees"
}
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
    final_ans=[]
    # Step 1: Perform Google Custom Search
    run=0
    seen_url=[]
    while len(final_ans)<k:
        print('=========== Iteration: %s - Query: %s ===========' %(run, q))
        urls = perform_google_search(api_key, engine_id, q)
        
        
        extracted_tuples = set()
        
        for idx, url in enumerate(urls[:k]):
            print(f"URL ({idx+1} / {len(urls)}): {url}")
            webpage_text = retrieve_webpage(url)
            seen_url.append(url)
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
                if model == "-gemini":
                    relations = extract_relations_with_gemini(trimmed_text, r, gemini_key,final_ans)
            
                if len(relations)==0:
                    continue


                if model == "-spanbert":
                    for subj, obj, rel, conf in relations:
                        print(RELATION_TYPES[r], rel)
                        if conf >= t and rel == RELATION_TYPES[r]:  # Explicit threshold check
                            print(f"\t=== Extracted Relation ===")
                            print(f"\tRelation Type: {rel}")
                            print(f"\tSubject: {subj} | Object: {obj} | Confidence: {conf:.2f}")
                            print("\tAdding to set of extracted relations")
                            extracted_tuples.add((subj, obj, rel, conf))
                if model == "-gemini":
                    for subj, obj in relations:
                        
                        final_ans.append((subj, obj))
                    print(final_ans)
                    q=final_ans[run][0]+ " " + final_ans[run][1]
                    continue

        # Step 2: Remove duplicates
        #print(f"\nProcessed {len(extracted_tuples)} relations")
        extracted_tuples = remove_duplicates(extracted_tuples)

        # Step 3: Select top k tuples
        #top_k_tuples = select_top_k_tuples(extracted_tuples, k, model)
        
        # Output top k relations
        print("\nTop Relations:")
        for tuple in extracted_tuples:
            print(f"({tuple[0]}, {tuple[1]}, {tuple[2]}, {tuple[3]})")
        for each in final_ans:
            final_ans.append(each)


        q = extracted_tuples[0][0] + " " + extracted_tuples[0][1]
        run+=1

if __name__ == '__main__':
    main()
