import requests
import sys

def google_search(query, API_KEY, Engine_ID, seen_urls):
    num_results=10
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": API_KEY,
        "cx": Engine_ID,
        "num": num_results
    }
   
    response = requests.get(url, params=params)
    results = response.json()

    urls = []
    
    for item in results.get("items", []):
        if item.get("link") not in seen_urls:
            urls.append({
                "URL": item.get("link")
            })
            seen_urls.add(item.get("link"))
        else:
            continue

    return urls, seen_urls
