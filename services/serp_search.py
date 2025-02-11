# services/serp_search.py
import requests
from config import settings

def get_serp_results(query: str, num_results: int = 5) -> str:
    """
    Query SERP API to retrieve organic results for the given query.
    Returns a formatted string listing titles, links, and snippets.
    """
    params = {
        "engine": "google",
        "q": query,
        "api_key": settings.serp_api_key,
        "num": num_results
    }
    response = requests.get("https://serpapi.com/search", params=params)
    if response.status_code == 200:
        data = response.json()
        organic_results = data.get("organic_results", [])
        formatted = []
        for res in organic_results:
            title = res.get("title", "No title")
            link = res.get("link", "No link")
            snippet = res.get("snippet", "No snippet")
            formatted.append(f"Title: {title}\nLink: {link}\nSnippet: {snippet}")
        return "\n\n".join(formatted)
    else:
        return "No search results found."
