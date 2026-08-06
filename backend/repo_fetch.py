"""Repo Fetch: safe, read-only access to a GitHub repo's metadata, file tree, and
individual file contents.

Deliberately does NOT `git clone` or execute anything from the target repo --
this is a backend service accepting a URL extracted from an uploaded PDF (or
supplied by the user), so treat it like any other external input:
- only github.com URLs are accepted (no arbitrary host -> no SSRF via other
  git hosts or internal network addresses)
- all calls go through GitHub's REST API / raw content CDN over HTTPS, with a
  timeout, so a slow/hanging host can't stall a request indefinitely
- file contents are fetched individually, on demand, and truncated -- there's
  no bulk download, no disk write, no extraction of an archive
"""
import logging
import re

import requests

from backend import config

logger = logging.getLogger("paperpilot.repo_fetch")

GITHUB_URL_RE = re.compile(
    r"https?://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?(?:[/#?].*)?$",
    re.IGNORECASE,
)

# Used to *find* a code URL inside a paper's full text (footnotes, abstract,
# "Code is available at ..." sentences, etc). Looser than GITHUB_URL_RE since
# it just needs to spot a candidate substring, not fully parse it.
CODE_URL_SCAN_RE = re.compile(
    r"https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", re.IGNORECASE
)


class RepoFetchError(Exception):
    """Raised for problems reaching or parsing the target repo (bad URL, 404,
    rate-limited, network failure) -- distinct from a bug in our own code."""


def extract_code_url(full_text: str) -> str | None:
    """Scan a paper's full text for the first GitHub URL mentioned. Papers
    commonly link code in the abstract or a footnote ("Code available at
    https://github.com/..."). Returns None if no GitHub URL is found -- the
    caller/user can then supply one explicitly instead."""
    match = CODE_URL_SCAN_RE.search(full_text)
    return match.group(0) if match else None


def parse_github_url(url: str) -> tuple[str, str]:
    """Extract (owner, repo) from a github.com URL. Raises RepoFetchError for
    anything that isn't a github.com repo URL -- this is the SSRF boundary."""
    match = GITHUB_URL_RE.match(url.strip())
    if not match:
        raise RepoFetchError(
            f"'{url}' doesn't look like a github.com repository URL. "
            "Only GitHub repos are supported."
        )
    return match.group(1), match.group(2)


def _headers() -> dict:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "PaperPilot-ReproAgent"}
    if config.GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {config.GITHUB_TOKEN}"
    return headers


def _get(url: str, **kwargs) -> requests.Response:
    try:
        resp = requests.get(url, headers=_headers(), timeout=config.REPRO_HTTP_TIMEOUT_SECONDS, **kwargs)
    except requests.RequestException as e:
        raise RepoFetchError(f"Network error reaching GitHub: {e}") from e

    if resp.status_code == 404:
        raise RepoFetchError("Repository (or path) not found -- it may be private, moved, or deleted.")
    if resp.status_code == 403 and "rate limit" in resp.text.lower():
        raise RepoFetchError(
            "GitHub API rate limit exceeded. Set GITHUB_TOKEN in .env for a much higher limit."
        )
    if resp.status_code >= 400:
        raise RepoFetchError(f"GitHub API returned {resp.status_code} for {url}")
    return resp


def fetch_repo_metadata(owner: str, repo: str) -> dict:
    """Repo-level metadata: license, default branch, archived status, stars.
    One cheap API call -- no tree or file content yet."""
    data = _get(f"https://api.github.com/repos/{owner}/{repo}").json()
    license_info = data.get("license") or {}
    return {
        "full_name": data.get("full_name", f"{owner}/{repo}"),
        "description": data.get("description"),
        "default_branch": data.get("default_branch", "main"),
        "archived": bool(data.get("archived", False)),
        "stargazers_count": data.get("stargazers_count", 0),
        "license_spdx_id": license_info.get("spdx_id"),
        "license_name": license_info.get("name"),
        "html_url": data.get("html_url", f"https://github.com/{owner}/{repo}"),
        "pushed_at": data.get("pushed_at"),
    }


def fetch_file_tree(owner: str, repo: str, branch: str) -> list[str]:
    """Full list of file paths in the repo (recursive), capped so a huge
    monorepo doesn't blow up memory/prompt size. Directories are excluded --
    only blob (file) entries are returned."""
    data = _get(
        f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}",
        params={"recursive": "1"},
    ).json()
    paths = [item["path"] for item in data.get("tree", []) if item.get("type") == "blob"]
    if data.get("truncated"):
        logger.warning("repo_tree_truncated owner=%s repo=%s entries=%d", owner, repo, len(paths))
    return paths[: config.REPRO_MAX_TREE_ENTRIES]


def fetch_file_content(owner: str, repo: str, branch: str, path: str) -> str | None:
    """Raw text content of a single file, truncated to a byte budget. Returns
    None (rather than raising) if the file can't be fetched or decoded --
    static checks treat a missing file as "check not satisfied", not a crash."""
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
    try:
        resp = requests.get(url, headers=_headers(), timeout=config.REPRO_HTTP_TIMEOUT_SECONDS, stream=True)
        if resp.status_code != 200:
            return None
        content = resp.raw.read(config.REPRO_MAX_FILE_BYTES + 1, decode_content=True)
        text = content.decode("utf-8", errors="replace")
        return text[: config.REPRO_MAX_FILE_BYTES]
    except requests.RequestException as e:
        logger.warning("file_fetch_failed owner=%s repo=%s path=%s error=%s", owner, repo, path, e)
        return None
