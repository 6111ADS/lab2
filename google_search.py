import requests

def perform_google_search(api_key, engine_id, query):
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={engine_id}&num={10}"
    response = requests.get(url)
    data = response.json()
    urls = [item['link'] for item in data.get('items', [])]
    return urls
