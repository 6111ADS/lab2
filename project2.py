
import sys
import json
import requests
from google_search import google_search
from webpage_retriever import retrieve_webpage
from relation_extractor import extract_relations_with_spanbert, extract_relations_with_gemini
from bs4 import BeautifulSoup
import spacy

RELATION_TYPES = {
    1: "per:schools_attended",
    2: "per:employee_of",
    3: "per:cities_of_residence",
    4: "org:top_members/employees"
}
nlp = spacy.load("en_core_web_lg")
def extract_sentences_and_entities(text):
    sentences = []
    doc = nlp(text)
    for sent in doc.sents:
        sentences.append(sent.text)
    return sentences

def clean_text(html):
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text()

def fetch_url(url):
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return clean_text(response.text)
    except:
        return ""
    return ""

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
    run=0
    if model == "-gemini":
        seen_urls = set()
        seen_tuples = set()
        final_ans = []
        used_queries = set()

        while len(final_ans) < k:
            print(f"\n=========== Iteration: {len(used_queries)} - Query: {q} ===========")
            urls,seen_urls = google_search(q, api_key, engine_id, seen_urls)
            used_queries.add(q)
            if not urls:
                print("No new search results found. Exiting.")
                sys.exit(1)

            for idx, url in enumerate(urls):
                print(f"URL ({idx+1} / {len(urls)}): {url}")
                url = url.get("URL")
                webpage_text = retrieve_webpage(url)
        
                if webpage_text:

                    trimmed_text = webpage_text[:10000]
                    print(f"        Trimming webpage content from {len(webpage_text)} to {len(trimmed_text)} characters")
                    print(f"        Webpage length (num characters): {len(trimmed_text)}")
                    sentences = extract_sentences_and_entities(trimmed_text)

                try:
                    relations = extract_relations_with_gemini(sentences, r, gemini_key)
                    for subj, obj in relations:
                        if (subj, obj) not in seen_tuples:
                            seen_tuples.add((subj, obj))
                            final_ans.append((subj, obj))
                            print(f"        Extracted Tuple: ({subj}, {obj})")
                except Exception as e:
                    print(f"        Error extracting relations: {e}")

            used_queries.add(q)

            if len(final_ans) >= k:
                break

            if len(final_ans) > len(used_queries):
                q = list(seen_tuples - used_queries)[0][0] + " " + list(seen_tuples - used_queries)[0][1]
            else:
                break

        print(f"\t================== ALL RELATIONS for {RELATION_TYPES[r]} ( {len(final_ans)} ) =====================")
        for subj, obj in final_ans:
            print(f"Subject: {subj},      | Object: {obj}")
        print(f"\tTotal # of iterations = {len(used_queries)}") 

    if model == "-spanbert":
        final_ans=dict()
        seen_urls=set() 
        seen_query=set()
        while len(final_ans)<k:
            print('=========== Iteration: %s - Query: %s ===========' %(run, q))
       
            urls,seen_urls = google_search(q, api_key, engine_id, seen_urls)
            seen_query.add(q)
            if not urls:
                q = ""
                for key, value in final_ans.items(): 
                    q = key[0] + " " + key[1]
                    if q not in seen_query:
                        break   
                    q = ""  
                continue
                

            for idx, url in enumerate(urls):
                
                print(f"URL ({idx+1} / {len(urls)}): {url}")
                url = url.get("URL")
                webpage_text = retrieve_webpage(url)
        
                if webpage_text:
                    #print(f"\tFetching text from url ...")
                    trimmed_text = webpage_text[:10000]  # Trimming to 10000 characters
                    print(f"\tTrimming webpage content from {len(webpage_text)} to {len(trimmed_text)} characters")
                    print(f"\tWebpage length (num characters): {len(trimmed_text)}")
                    
                  
                    final_ans = extract_relations_with_spanbert(trimmed_text, t, final_ans, r)
                    if len(final_ans)==0:
                        continue

                    final_ans = dict(sorted(final_ans.items(), key=lambda item: item[1][0], reverse=True))

                    q = ""
                    for index, (key, value) in enumerate(final_ans.items()):
                        if len(q)==0:
                            q = key[0] + " " + key[1]
                            if q in seen_query:
                                q = ""
            run +=1 
        print(f"\t================== ALL RELATIONS for {RELATION_TYPES[r]} ( {len(final_ans)} ) =====================")
        for index, (key, value) in enumerate(final_ans.items()):
            print(f"Confidence: {value[0]},     | Subject: {key[0]},      | Object: {key[1]}")
        print(f"\tTotal # of iterations = {run}")                  
if __name__ == '__main__':
    main()
