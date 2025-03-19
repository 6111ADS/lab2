# search.py
import requests

def google_search(query, api_key, cx):
    """
    Perform a Google search and return a list of result URLs.
    """
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": api_key, "cx": cx, "q": query, "num": 10}
    
    response = requests.get(search_url, params=params)
    results = response.json()

    urls = [item["link"] for item in results.get("items", [])]
    return urls
