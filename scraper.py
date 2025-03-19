# scraper.py
import requests
from bs4 import BeautifulSoup

def extract_text_from_url(url):
    """
    Fetch the webpage content and extract the main text.
    """
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract paragraphs
        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text() for p in paragraphs])
        return text.strip()
    
    except requests.RequestException as e:
        print(f"Failed to fetch {url}: {e}")
        return None

if __name__ == "__main__":
    sample_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    text = extract_text_from_url(sample_url)
    print(text[:500])  # Print first 500 characters
