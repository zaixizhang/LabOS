"""
LabOS Web Tools — URL / PDF content extraction.
"""

import os
import re

import requests
from bs4 import BeautifulSoup
from io import BytesIO
from markdownify import markdownify
from requests.exceptions import RequestException
from smolagents import tool


@tool
def visit_webpage(url: str) -> str:
    """Visit a webpage and return its content as Markdown.

    Args:
        url: The URL of the webpage to visit.

    Returns:
        Webpage content converted to Markdown.
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        md = markdownify(response.text).strip()
        md = re.sub(r"\n{3,}", "\n\n", md)
        return md
    except RequestException as exc:
        return f"Error fetching webpage: {exc}"
    except Exception as exc:
        return f"Unexpected error: {exc}"


@tool
def extract_url_content(url: str) -> str:
    """Extract text content from a webpage using BeautifulSoup.

    Args:
        url: Webpage URL to extract content from

    Returns:
        Text content of the webpage
    """
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        ct = response.headers.get("Content-Type", "")
        if "text/plain" in ct or "application/json" in ct:
            return response.text.strip()

        soup = BeautifulSoup(response.text, "html.parser")
        content = soup.find("main") or soup.find("article") or soup.body
        if content is None:
            return "No extractable content found."
        for tag in content(["script", "style", "nav", "header", "footer", "aside", "iframe"]):
            tag.decompose()

        paragraphs = content.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
        return "\n\n".join(p.get_text().strip() for p in paragraphs if p.get_text().strip())
    except Exception as exc:
        return f"Error extracting URL content: {exc}"


@tool
def extract_pdf_content(url: str) -> str:
    """Extract text from a PDF given its URL.

    Args:
        url: URL of the PDF file

    Returns:
        Extracted text content
    """
    import PyPDF2

    try:
        if not url.lower().endswith(".pdf"):
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                links = re.findall(r'href=[\'"]([^\'"]+\.pdf)[\'"]', resp.text)
                if links:
                    base = "/".join(url.split("/")[:3])
                    url = links[0] if links[0].startswith("http") else base + links[0]
                else:
                    return f"No PDF found at {url}"

        resp = requests.get(url, timeout=30)
        ct = resp.headers.get("Content-Type", "").lower()
        if "application/pdf" not in ct and not resp.content.startswith(b"%PDF"):
            return f"URL did not return a PDF. Content-Type: {ct}"

        reader = PyPDF2.PdfReader(BytesIO(resp.content))
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        text = re.sub(r"\s+", " ", text).strip()
        return text if text else "PDF contains no extractable text (may require OCR)."
    except Exception as exc:
        return f"PDF extraction error: {exc}"


@tool
def fetch_supplementary_info_from_doi(doi: str, output_dir: str = "supplementary_info") -> str:
    """Fetch supplementary materials for a paper given its DOI.

    Args:
        doi: The paper DOI
        output_dir: Directory to save supplementary files

    Returns:
        Research log of the download process
    """
    from urllib.parse import urljoin

    log: list = [f"Processing DOI: {doi}"]
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        resp = requests.get(f"https://doi.org/{doi}", headers=headers, timeout=30)
        if resp.status_code != 200:
            return f"Failed to resolve DOI: {doi}"
        publisher_url = resp.url
        log.append(f"Resolved to: {publisher_url}")

        resp = requests.get(publisher_url, headers=headers, timeout=30)
        soup = BeautifulSoup(resp.content, "html.parser")
        supp_links = []
        for link in soup.find_all("a", href=True):
            text = link.get_text().lower()
            if any(kw in text for kw in ("supplementary", "supplemental", "appendix")):
                supp_links.append(urljoin(publisher_url, link["href"]))

        if not supp_links:
            log.append("No supplementary materials found.")
            return "\n".join(log)

        os.makedirs(output_dir, exist_ok=True)
        downloaded = 0
        for link in supp_links:
            fname = os.path.join(output_dir, link.split("/")[-1])
            try:
                r = requests.get(link, headers=headers, timeout=30)
                if r.status_code == 200:
                    with open(fname, "wb") as f:
                        f.write(r.content)
                    downloaded += 1
                    log.append(f"Downloaded: {fname}")
            except Exception:
                log.append(f"Failed to download: {link}")

        log.append(f"Total downloaded: {downloaded}")
    except Exception as exc:
        log.append(f"Error: {exc}")
    return "\n".join(log)
