# project2.py
import sys
import requests
import time
import config
from search import google_search
from scraper import extract_text_from_url
from relation_extraction import extract_entities, extract_relations_spanbert, extract_relations_gemini

def parse_args():
    """
    Parses command-line arguments.
    """
    if len(sys.argv) != 9:
        print("Usage: python3 project2.py [-spanbert|-gemini] <google_api_key> <google_engine_id> <google_gemini_api_key> <r> <t> <q> <k>")
        sys.exit(1)

    extraction_method = sys.argv[1]
    google_api_key = sys.argv[2]
    google_cx = sys.argv[3]
    gemini_api_key = sys.argv[4]
    relation_type = int(sys.argv[5])
    confidence_threshold = float(sys.argv[6])
    seed_query = sys.argv[7].strip('"')
    k = int(sys.argv[8])

    if extraction_method not in ["-spanbert", "-gemini"]:
        print("Error: Extraction method must be -spanbert or -gemini")
        sys.exit(1)

    if relation_type not in [1, 2, 3, 4]:
        print("Error: Relation type must be 1 (Schools_Attended), 2 (Work_For), 3 (Live_In), or 4 (Top_Member_Employees)")
        sys.exit(1)

    return extraction_method, google_api_key, google_cx, gemini_api_key, relation_type, confidence_threshold, seed_query, k

def main():
    extraction_method, google_api_key, google_cx, gemini_api_key, relation_type, confidence_threshold, seed_query, k = parse_args()

    # Map relation ID to relation name
    relation_map = {
        1: "Schools_Attended",
        2: "Work_For",
        3: "Live_In",
        4: "Top_Member_Employees"
    }
    relation_name = relation_map[relation_type]

    print(f"\nExtracting relation: {relation_name}")
    print(f"Using method: {extraction_method[1:].capitalize()}\n")

    extracted_tuples = set()
    iteration = 0

    while len(extracted_tuples) < k:
        iteration += 1
        print(f"\n### Iteration {iteration} ###")
        
        # Step 1: Perform Google Search
        urls = google_search(seed_query, google_api_key, google_cx)
        print(f"Found {len(urls)} URLs.")

        for url in urls:
            if len(extracted_tuples) >= k:
                break

            print(f"\nProcessing: {url}")

            # Step 2: Extract text from webpage
            text = extract_text_from_url(url)
            if not text:
                continue

            # Step 3: Extract relations
            if extraction_method == "-spanbert":
                relations = extract_relations_spanbert(text, relation_name)
                # Filter by confidence threshold
                relations = [rel for rel in relations if rel[2] >= confidence_threshold]
            else:
                relations = extract_relations_gemini(text, relation_name, gemini_api_key)

            for rel in relations:
                if len(extracted_tuples) >= k:
                    break
                extracted_tuples.add(tuple(rel))
                print(f"Extracted: {rel}")

        time.sleep(1)  # Avoid API rate limits

    print("\nFinal extracted tuples:")
    for t in extracted_tuples:
        print(t)

if __name__ == "__main__":
    main()
