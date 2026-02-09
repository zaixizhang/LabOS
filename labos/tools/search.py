"""
LabOS Search Tools — web + academic search.
"""

import os
import re
import threading

import requests
from smolagents import tool

try:
    from googlesearch import search as _gsearch
    GOOGLE_SEARCH_AVAILABLE = True
except ImportError:
    GOOGLE_SEARCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Google search
# ---------------------------------------------------------------------------

@tool
def enhanced_google_search(query: str, num_results: int = 5, include_snippets: bool = True) -> str:
    """Enhanced Google search with reliable implementation.

    Args:
        query: Search query
        num_results: Number of results to return (default: 5)
        include_snippets: Whether to include result snippets (default: True)

    Returns:
        Formatted search results with titles, URLs, and descriptions
    """
    try:
        if GOOGLE_SEARCH_AVAILABLE:
            results = []
            for i, result in enumerate(
                _gsearch(query, num_results=num_results, advanced=True), 1
            ):
                if i > num_results:
                    break
                title = getattr(result, "title", f"Result {i}")
                url = getattr(result, "url", "No URL")
                desc = getattr(result, "description", "")
                if include_snippets:
                    results.append(f"**{i}. {title}**\n  {url}\n  {desc}\n")
                else:
                    results.append(f"**{i}. {title}**\n  {url}\n")
            if results:
                return (
                    f"Google Search Results for '{query}':\n\n" + "\n".join(results)
                )
            return f"No search results for: '{query}'"
        return "Google search unavailable. Install googlesearch-python."
    except Exception as exc:
        return f"Google search failed: {exc}"


@tool
def search_with_serpapi(query: str, num_results: int = 5) -> str:
    """Google search via SerpAPI (requires SERPAPI_API_KEY).

    Args:
        query: Search query
        num_results: Number of results to return (default: 5)

    Returns:
        Formatted search results
    """
    key = os.getenv("SERPAPI_API_KEY")
    if not key:
        return "SerpAPI key not set. Set SERPAPI_API_KEY in .env."
    try:
        resp = requests.get(
            "https://serpapi.com/search",
            params={"engine": "google", "q": query, "api_key": key, "num": min(num_results, 10)},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        organic = data.get("organic_results", [])
        if not organic:
            return f"No SerpAPI results for: '{query}'"
        lines = []
        for r in organic[:num_results]:
            lines.append(f"**{r.get('title', '')}**\n{r.get('link', '')}\n{r.get('snippet', '')}\n")
        return f"SerpAPI Results for '{query}':\n\n" + "\n".join(lines)
    except Exception as exc:
        return f"SerpAPI search failed: {exc}"


@tool
def multi_source_search(query: str, sources: str = "google,serpapi") -> str:
    """Unified search across multiple sources.

    Args:
        query: Search query
        sources: Comma-separated sources (google, serpapi, knowledge)

    Returns:
        Consolidated search results
    """
    source_list = [s.strip().lower() for s in sources.split(",")]
    results = []

    if "google" in source_list:
        gr = enhanced_google_search(query, num_results=3)
        if gr and "failed" not in gr.lower():
            results.append(f"## Google Search\n{gr}")

    if "serpapi" in source_list:
        sr = search_with_serpapi(query)
        if sr and "not set" not in sr.lower() and "failed" not in sr.lower():
            results.append(f"## SerpAPI\n{sr}")

    if "knowledge" in source_list:
        kr = enhanced_knowledge_search(query)
        if kr and "failed" not in kr.lower() and "not found" not in kr.lower():
            results.append(f"## Knowledge\n{kr}")

    if not results:
        return f"No results from any source for: '{query}'"
    return f"# Multi-Source Results for '{query}'\n\n" + "\n\n---\n\n".join(results)


@tool
def enhanced_knowledge_search(query: str, model_name: str = "gemini-3") -> str:
    """Use LLM knowledge to provide detailed information.

    Args:
        query: Knowledge query
        model_name: LLM model for knowledge expansion

    Returns:
        Detailed information from LLM training knowledge
    """
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        return "OpenRouter API key not found."
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json={
                "model": f"google/{model_name}",
                "messages": [
                    {"role": "system", "content": "You are a knowledgeable research assistant."},
                    {"role": "user", "content": f"Provide detailed information on: {query}"},
                ],
                "temperature": 0.1,
                "max_tokens": 3000,
            },
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if "choices" in data and data["choices"]:
            return f"Knowledge Response for '{query}':\n\n{data['choices'][0]['message']['content']}"
        return "No knowledge response generated."
    except Exception as exc:
        return f"Knowledge search failed: {exc}"


# ---------------------------------------------------------------------------
# Academic search
# ---------------------------------------------------------------------------

@tool
def query_arxiv(query: str, max_papers: int = 10) -> str:
    """Query arXiv for papers.

    Args:
        query: Search query
        max_papers: Max papers to retrieve (default: 10)

    Returns:
        Formatted search results
    """
    import arxiv

    try:
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=max_papers, sort_by=arxiv.SortCriterion.Relevance)
        papers = [f"Title: {p.title}\nSummary: {p.summary}" for p in client.results(search)]
        return "\n\n".join(papers) if papers else "No arXiv papers found."
    except Exception as exc:
        return f"arXiv query error: {exc}"


@tool
def query_pubmed(query: str, max_papers: int = 10) -> str:
    """Query PubMed for papers.

    Args:
        query: Search query
        max_papers: Max papers to retrieve (default: 10)

    Returns:
        Formatted search results
    """
    from pymed import PubMed

    try:
        pubmed = PubMed(tool="LabOS", email="labos@example.com")
        papers = list(pubmed.query(query, max_results=max_papers))
        if papers:
            return "\n\n".join(
                f"Title: {p.title}\nAbstract: {p.abstract}\nJournal: {p.journal}"
                for p in papers
            )
        return "No PubMed papers found."
    except Exception as exc:
        return f"PubMed query error: {exc}"


@tool
def query_scholar(query: str, timeout_seconds: int = 30) -> str:
    """Query Google Scholar for papers (with timeout).

    Args:
        query: Search query
        timeout_seconds: Max wait time (default: 30)

    Returns:
        Scholar results or fallback
    """
    try:
        from scholarly import scholarly
    except ImportError:
        return enhanced_google_search(query, num_results=3)

    container: dict = {"text": None}

    def _worker():
        try:
            result = next(scholarly.search_pubs(query), None)
            if result:
                bib = result.get("bib", {})
                container["text"] = (
                    f"Title: {bib.get('title', 'N/A')}\n"
                    f"Year: {bib.get('pub_year', 'N/A')}\n"
                    f"Abstract: {bib.get('abstract', 'N/A')}"
                )
            else:
                container["text"] = "No Google Scholar results."
        except Exception as exc:
            container["text"] = f"Scholar error: {exc}"

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout_seconds)
    if t.is_alive():
        return f"Google Scholar timed out after {timeout_seconds}s."
    return container["text"] or "No results."


# ---------------------------------------------------------------------------
# GitHub search
# ---------------------------------------------------------------------------

@tool
def search_github_repositories(
    query: str, language: str = "", sort: str = "stars", per_page: int = 10
) -> str:
    """Search GitHub repositories.

    Args:
        query: Search query
        language: Programming language filter
        sort: Sort by (stars, forks, updated)
        per_page: Number of results

    Returns:
        Formatted repository list
    """
    try:
        q = query + (f" language:{language}" if language else "")
        resp = requests.get(
            "https://api.github.com/search/repositories",
            params={"q": q, "sort": sort, "order": "desc", "per_page": min(per_page, 100)},
            timeout=10,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
        if not items:
            return f"No repos found for: {query}"
        lines = []
        for i, r in enumerate(items, 1):
            lines.append(
                f"{i}. **{r['name']}** ({r['full_name']})\n"
                f"   {r.get('description', 'No description')}\n"
                f"   {r.get('language', 'N/A')} | Stars: {r.get('stargazers_count', 0)}\n"
                f"   {r.get('html_url', '')}\n"
            )
        return f"GitHub Results for '{query}':\n\n" + "\n".join(lines)
    except Exception as exc:
        return f"GitHub search failed: {exc}"


@tool
def search_github_code(
    query: str, language: str = "", extension: str = "", per_page: int = 10
) -> str:
    """Search for code on GitHub.

    Args:
        query: Code search query
        language: Language filter
        extension: File extension filter
        per_page: Number of results

    Returns:
        Code search results
    """
    try:
        q = query
        if language:
            q += f" language:{language}"
        if extension:
            q += f" extension:{extension}"
        resp = requests.get(
            "https://api.github.com/search/code",
            params={"q": q, "per_page": min(per_page, 100)},
            timeout=10,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
        if not items:
            return f"No code found for: {query}"
        lines = []
        for i, item in enumerate(items, 1):
            lines.append(
                f"{i}. **{item['name']}**\n"
                f"   Repo: {item.get('repository', {}).get('full_name', 'N/A')}\n"
                f"   Path: {item.get('path', 'N/A')}\n"
                f"   URL: {item.get('html_url', '')}\n"
            )
        return f"GitHub Code Results for '{query}':\n\n" + "\n".join(lines)
    except Exception as exc:
        return f"GitHub code search failed: {exc}"


@tool
def get_github_repository_info(repo_owner: str, repo_name: str) -> str:
    """Get detailed info about a GitHub repository.

    Args:
        repo_owner: Repository owner
        repo_name: Repository name

    Returns:
        Detailed repository information
    """
    try:
        resp = requests.get(f"https://api.github.com/repos/{repo_owner}/{repo_name}", timeout=10)
        resp.raise_for_status()
        r = resp.json()
        info = (
            f"Repository: {repo_owner}/{repo_name}\n"
            f"Description: {r.get('description', 'N/A')}\n"
            f"Language: {r.get('language', 'N/A')}\n"
            f"Stars: {r.get('stargazers_count', 0)} | Forks: {r.get('forks_count', 0)}\n"
            f"URL: {r.get('html_url', '')}\n"
            f"Clone: {r.get('clone_url', '')}\n"
        )
        # Try to get README
        try:
            rm = requests.get(
                f"https://api.github.com/repos/{repo_owner}/{repo_name}/readme", timeout=10
            ).json()
            readme = requests.get(rm["download_url"], timeout=10).text[:1000]
            info += f"\nREADME preview:\n{readme}"
        except Exception:
            pass
        return info
    except Exception as exc:
        return f"Failed to get repo info: {exc}"
