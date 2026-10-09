"""Internet search and bounded public webpage reading for the local model.

Search reuses the existing DDGS helper without entering its Gemini fallback.
Workers give both search and page retrieval a hard 20-second process deadline.
Public URLs are resolved and pinned before connecting; every redirect is checked.
The reader accepts text/HTML only and removes scripts, forms and embedded media.
Downloaded text is untrusted evidence, never an instruction to run local tools.
No accounts, API keys, Docker changes or additional listening services are used.
"""
import ipaddress
import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

import certifi
import urllib3
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
MAX_BYTES = 256_000
MAX_CHARS = 8_000


def public_url(url):
    """Reject local addresses before making any request, including redirects."""
    parts = urlsplit(url)
    if (parts.scheme not in ("http", "https") or not parts.hostname
            or parts.username or parts.password or parts.port not in (None, 80, 443)):
        raise ValueError("Only public HTTP/HTTPS URLs on ports 80/443 are allowed")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    addresses = socket.getaddrinfo(parts.hostname, port, type=socket.SOCK_STREAM)
    ips = list(dict.fromkeys(item[4][0] for item in addresses))
    if not ips or any(not ipaddress.ip_address(ip).is_global
                      or ipaddress.ip_address(ip).is_multicast for ip in ips):
        raise ValueError("Local, private and reserved addresses are unavailable to webpage reading")
    clean = urlunsplit((parts.scheme, parts.netloc, parts.path or "/", parts.query, ""))
    return clean, parts, ips[0], port


def read_public_page(url):
    deadline = time.monotonic() + 15
    for _ in range(4):
        url, parts, address, port = public_url(url)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Webpage deadline exceeded")
        # Connect to the validated address, while preserving the TLS hostname.
        # This avoids a second DNS resolution switching to a private address.
        options = {"timeout": urllib3.Timeout(connect=min(5, remaining), read=min(5, remaining)),
                   "maxsize": 1}
        if parts.scheme == "https":
            pool = urllib3.HTTPSConnectionPool(address, port, server_hostname=parts.hostname,
                assert_hostname=parts.hostname, cert_reqs="CERT_REQUIRED", ca_certs=certifi.where(), **options)
        else:
            pool = urllib3.HTTPConnectionPool(address, port, **options)
        response = None
        try:
            target = urlunsplit(("", "", parts.path or "/", parts.query, ""))
            response = pool.request("GET", target, redirect=False, retries=False, preload_content=False,
                headers={"Host": parts.netloc, "User-Agent": "Mark-LV/LocalWebReader",
                         "Accept": "text/html, text/plain", "Accept-Encoding": "identity"})
            if response.status in (301, 302, 303, 307, 308):
                location = response.headers.get("Location")
                if not location:
                    raise ValueError("Redirect has no destination")
                url = urljoin(url, location)
                continue
            if response.status != 200:
                raise ValueError(f"Webpage returned HTTP {response.status}")
            mime = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if mime not in ("text/html", "text/plain", "application/xhtml+xml"):
                raise ValueError("Only HTML or plain-text webpages can be read")
            body = bytearray()
            while len(body) <= MAX_BYTES:
                if time.monotonic() > deadline:
                    raise TimeoutError("Webpage deadline exceeded")
                block = response.read(min(8192, MAX_BYTES + 1 - len(body)))
                if not block:
                    break
                body.extend(block)
            if len(body) > MAX_BYTES:
                raise ValueError("Webpage exceeds the 256 KB download limit")
            if mime == "text/plain":
                text, title = body.decode("utf-8", errors="replace"), parts.hostname
            else:
                soup = BeautifulSoup(bytes(body), "html.parser")
                title = soup.title.get_text(" ", strip=True) if soup.title else parts.hostname
                for element in soup(["script", "style", "noscript", "iframe", "form", "svg", "nav", "footer"]):
                    element.decompose()
                text = (soup.find("main") or soup.find("article") or soup).get_text(" ", strip=True)
            return {"url": url, "title": title[:200], "text": text[:MAX_CHARS],
                    "truncated": len(text) > MAX_CHARS, "untrusted_web_content": True}
        finally:
            if response is not None:
                response.close()
            pool.close()
    raise ValueError("Too many webpage redirects")


def search_public_web(query):
    from actions.web_search import _ddg_search
    results = _ddg_search(query, max_results=5, timeout=8, backend="duckduckgo")
    # DDG occasionally returns an empty transient response; retry the same
    # provider once within the outer worker deadline, without switching services.
    if not results:
        time.sleep(0.5)
        results = _ddg_search(query, max_results=5, timeout=8, backend="duckduckgo")
    cleaned = []
    for result in results[:5]:
        url = result.get("url", "")
        parts = urlsplit(url)
        if parts.scheme in ("http", "https") and parts.hostname and not parts.username and not parts.password:
            cleaned.append({"title": result.get("title", "")[:200],
                            "snippet": result.get("snippet", "")[:1000], "url": url})
    return {"query": query, "provider": "DuckDuckGo via DDGS", "results": cleaned,
            "untrusted_web_content": True,
            "error": None if cleaned else "Search returned no results; do not invent current facts."}


class LocalWeb:
    def run(self, operation, value):
        try:
            result = subprocess.run([sys.executable, "-m", "core.local_web", operation],
                input=json.dumps({"value": value}), capture_output=True, text=True,
                cwd=ROOT, timeout=20, check=True)
            return json.loads(result.stdout.splitlines()[-1])
        except subprocess.TimeoutExpired:
            return {"error": "Web request exceeded its 20-second deadline"}
        except (subprocess.CalledProcessError, ValueError, IndexError) as exc:
            return {"error": f"Web request failed: {type(exc).__name__}"}


if __name__ == "__main__":
    try:
        value = json.loads(sys.stdin.read())["value"]
        if sys.argv[1] == "search":
            result = search_public_web(value)
        elif sys.argv[1] == "read":
            result = read_public_page(value)
        else:
            raise ValueError("Unknown web operation")
    except Exception as exc:
        result = {"error": str(exc)}
    print(json.dumps(result))
