import requests
from bs4 import BeautifulSoup
import re
import unicodedata

def remove_backslash(line):
    return re.sub(r"\\'", ' ', line)

def clean_text(text):
    text = unicodedata.normalize("NFKD", text)
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def replace_sentence_ends(text):
    """Replaces sentence ending punctuation with commas."""
    text = re.sub(r'(?<=[.!?])\s+', ', ', text)
    return text

def retrieve_webpage(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        for script in soup(['script', 'style', 'img', 'footer', 'header', 'nav', 'aside', 'form', 'svg', 'button', 'input', 'select', 'textarea']):
            script.decompose()

        paragraphs = [p.get_text() for p in soup.find_all('p')]
        combined_text = ""
        for paragraph in paragraphs:
            cleaned_paragraph = clean_text(remove_backslash(paragraph))
            if cleaned_paragraph:
                combined_text += cleaned_paragraph + " "

        combined_text = replace_sentence_ends(combined_text.strip())
        return combined_text

    except requests.exceptions.RequestException:
        return ''
    except Exception:
        return ''
'''
import requests
from bs4 import BeautifulSoup
import re
import unicodedata

def remove_backslash(line):
    return re.sub(r"\\'", ' ', line)

def clean_text(text):
    text = unicodedata.normalize("NFKD", text)
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def retrieve_webpage(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  
        soup = BeautifulSoup(response.content, 'html.parser')

        for script in soup(['script', 'style', 'img', 'footer', 'header', 'nav', 'aside', 'form', 'svg', 'button', 'input', 'select', 'textarea']):
            script.decompose()

        text_elements = [p.get_text() for p in soup.find_all(text=True)]
        text = ' '.join(text_elements)
        text = remove_backslash(text)
        text = clean_text(text)

        return text

    except requests.exceptions.RequestException:
        return '' 
    except Exception:
        return ''  


'''

