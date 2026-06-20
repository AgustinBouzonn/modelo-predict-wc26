"""
Fotos desde la API de Wikipedia (pageimages). Dado un título de página,
devuelve la URL del thumbnail. Procesa en lotes de 50 y resuelve
normalizaciones y redirects para mapear cada título original a su foto.
"""
from __future__ import annotations

import time
import unicodedata

import requests

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "WC26-Predictor/0.1 (educational use)"}


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", str(s))


def fetch_thumbnails(titles: list[str], size: int = 200, pause: float = 0.4,
                     batch_size: int = 30, retries: int = 3) -> dict[str, str]:
    """Mapa título_original -> URL de foto (omite los que no tienen).

    Normaliza a NFC en ambos lados: la API devuelve los títulos en una forma
    unicode que puede diferir de la del href de Wikipedia (acentos), y sin esto
    el mapeo de vuelta falla para nombres con tildes. Reintenta ante rate-limit.
    """
    uniq = list(dict.fromkeys(t for t in titles if t))
    result: dict[str, str] = {}

    for i in range(0, len(uniq), batch_size):
        batch = uniq[i:i + batch_size]
        params = {
            "action": "query", "format": "json", "prop": "pageimages",
            "piprop": "thumbnail", "pithumbsize": size, "redirects": 1,
            "titles": "|".join(batch),
        }
        data = None
        for attempt in range(retries):
            try:
                r = requests.get(API, params=params, headers=HEADERS, timeout=30)
                if r.status_code == 200:
                    data = r.json().get("query", {})
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.6 * (attempt + 1))
        if data is None:
            continue

        norm = {_nfc(n["from"]): _nfc(n["to"]) for n in data.get("normalized", [])}
        redir = {_nfc(rd["from"]): _nfc(rd["to"]) for rd in data.get("redirects", [])}
        final_thumb = {_nfc(p["title"]): p["thumbnail"]["source"]
                       for p in data.get("pages", {}).values() if "thumbnail" in p}

        def resolve(orig):
            t = norm.get(orig, orig)
            return redir.get(t, t)

        for orig in batch:
            thumb = final_thumb.get(resolve(_nfc(orig)))
            if thumb:
                result[orig] = thumb
        time.sleep(pause)
    return result
