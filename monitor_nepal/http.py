"""HTTP helpers: retrying sessions, a polite User-Agent and a byte cache for map tiles."""
from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import DISK_CACHE_DIR, HTTP_TIMEOUT, USER_AGENT

_local = threading.local()


class FetchError(RuntimeError):
    """A data source could not be reached or returned something unusable."""


def session() -> requests.Session:
    """One session per thread (requests sessions are not guaranteed thread-safe)."""
    s = getattr(_local, "session", None)
    if s is None:
        s = requests.Session()
        retry = Retry(total=3, backoff_factor=0.6, status_forcelist=(500, 502, 503, 504),
                      allowed_methods=("GET", "POST"), respect_retry_after_header=True)
        adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=32)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        s.headers["User-Agent"] = USER_AGENT
        _local.session = s
    return s


def get(url: str, *, params: dict | None = None, timeout: float | None = None, **kw: Any) -> requests.Response:
    try:
        with _host_semaphore(url):
            r = session().get(url, params=params, timeout=timeout or HTTP_TIMEOUT, **kw)
    except requests.RequestException as exc:  # network error after retries
        raise FetchError(f"Could not reach {url.split('?')[0]}: {exc.__class__.__name__}") from exc
    if r.status_code == 429:
        raise FetchError(f"{url.split('?')[0]} is rate-limiting requests right now; try again in a minute.")
    if r.status_code >= 400:
        raise FetchError(f"{url.split('?')[0]} answered HTTP {r.status_code}.")
    return r


def get_json(url: str, *, params: dict | None = None, timeout: float | None = None) -> Any:
    r = get(url, params=params, timeout=timeout)
    try:
        return r.json()
    except ValueError as exc:
        raise FetchError(f"{url.split('?')[0]} returned invalid JSON.") from exc


def post_json(url: str, payload: dict, *, timeout: float | None = None) -> Any:
    try:
        r = session().post(url, json=payload, timeout=timeout or HTTP_TIMEOUT)
    except requests.RequestException as exc:
        raise FetchError(f"Could not reach {url}: {exc.__class__.__name__}") from exc
    if r.status_code >= 400:
        raise FetchError(f"{url} answered HTTP {r.status_code}: {r.text[:200]}")
    return r.json()


class ByteCache:
    """Thread-safe LRU cache of response bodies (map tiles), optionally mirrored on disk."""

    def __init__(self, max_bytes: int = 192 * 1024 * 1024, disk_dir: str | None = DISK_CACHE_DIR):
        self.max_bytes = max_bytes
        self._data: OrderedDict[str, bytes] = OrderedDict()
        self._size = 0
        self._lock = threading.Lock()
        self.disk = Path(disk_dir) if disk_dir else None
        if self.disk:
            self.disk.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        h = hashlib.sha1(key.encode()).hexdigest()
        return self.disk / h[:2] / h  # type: ignore[operator]

    def get(self, key: str) -> bytes | None:
        with self._lock:
            value = self._data.get(key)
            if value is not None:
                self._data.move_to_end(key)
                return value
        if self.disk:
            p = self._path(key)
            if p.exists():
                value = p.read_bytes()
                self.put(key, value, write_disk=False)
                return value
        return None

    def put(self, key: str, value: bytes, write_disk: bool = True) -> None:
        with self._lock:
            if key in self._data:
                return
            self._data[key] = value
            self._size += len(value)
            while self._size > self.max_bytes and self._data:
                _, old = self._data.popitem(last=False)
                self._size -= len(old)
        if self.disk and write_disk:
            p = self._path(key)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_bytes(value)
            tmp.replace(p)


TILE_CACHE = ByteCache()

# Some providers answer "no tile here" with an empty 200/204/404; we remember those too.
EMPTY = b""

# Free services throttle bursts, and on a hosted app every visitor shares one IP address: cap how many requests
# run at once against each host, and back off when a host pushes back (403/429/5xx).
HOST_LIMITS = {"wayback.maptiles.arcgis.com": 3, "metadata.maptiles.arcgis.com": 2, "tiles.maps.eox.at": 6,
               "planetarycomputer.microsoft.com": 6, "server.arcgisonline.com": 8, "gibs.earthdata.nasa.gov": 6}
_host_sems: dict[str, threading.BoundedSemaphore] = {}
_sem_lock = threading.Lock()


def _host_semaphore(url: str) -> threading.BoundedSemaphore:
    host = url.split("/")[2] if "://" in url else url
    with _sem_lock:
        if host not in _host_sems:
            _host_sems[host] = threading.BoundedSemaphore(HOST_LIMITS.get(host, 8))
        return _host_sems[host]


def get_tile_bytes(url: str, *, timeout: float = 25, attempts: int = 4) -> bytes:
    cached = TILE_CACHE.get(url)
    if cached is not None:
        return cached
    import time

    status = None
    for attempt in range(attempts):
        try:
            with _host_semaphore(url):
                r = session().get(url, timeout=timeout)
        except requests.RequestException as exc:
            if attempt == attempts - 1:
                raise FetchError(f"Tile request failed: {exc.__class__.__name__}") from exc
            time.sleep(0.8 * 2 ** attempt)
            continue
        status = r.status_code
        if status in (204, 404) or (status == 200 and not r.content):
            TILE_CACHE.put(url, EMPTY)
            return EMPTY
        if status in (403, 429) or status >= 500:
            time.sleep(0.8 * 2 ** attempt)
            continue
        if status >= 400:
            raise FetchError(f"Tile server answered HTTP {status}.")
        TILE_CACHE.put(url, r.content)
        return r.content
    raise FetchError("The imagery server is throttling requests right now; please try again in a minute."
                     if status in (403, 429) else f"Tile server answered HTTP {status}.")
