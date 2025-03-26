import requests
from bs4 import BeautifulSoup

def retrieve_webpage(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.content, 'html.parser')

        for script in soup(['script', 'style', 'img', 'footer', 'header', 'nav']):
            script.decompose()
        text = soup.get_text(separator=' ', strip=True)
        return text[:10000]  
    except requests.exceptions.RequestException as e:
        print(f"Error retrieving {url}: {e}")
        return None
