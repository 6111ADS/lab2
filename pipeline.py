# pipeline.py
from search import google_search
from scraper import extract_text_from_url
from relation_extraction import extract_entities, extract_relations
import config

def main():
    query = "Recent AI breakthroughs"
    
    for iteration in range(config.MAX_ITERATIONS):
        print(f"\n### Iteration {iteration + 1} ###")

        # 1. Search for new documents
        urls = google_search(query)
        print(f"Found {len(urls)} URLs.")

        for url in urls:
            print(f"\nProcessing: {url}")

            # 2. Extract text
            text = extract_text_from_url(url)
            if not text:
                continue

            # 3. Extract entities & relations
            entities = extract_entities(text)
            relations = extract_relations(text)

            print("Entities Found:", entities)
            print("Relations Extracted:", relations)

if __name__ == "__main__":
    main()
