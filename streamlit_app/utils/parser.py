import requests
from bs4 import BeautifulSoup

def fetch_html(url):
    """Fetch raw HTML content from a URL."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        return res.text
    except Exception as e:
        print(f"⚠️ Error fetching {url}: {e}")
        return ""

def extract_text(html):
    """Remove scripts, styles, and return cleaned visible text."""
    soup = BeautifulSoup(html, 'html.parser')
    for s in soup(['script', 'style']):
        s.extract()
    text = soup.get_text(separator=' ')
    return ' '.join(text.split())

def parse_url(url):
    """Convenience wrapper: fetch + clean text from URL."""
    html = fetch_html(url)
    if not html:
        return ""
    return extract_text(html)
