"""Modelo predictivo del Mundial 2026 (ensemble Elo + Poisson + ML)."""

__version__ = "0.1.0"

# La consola de Windows usa cp1252 por defecto y rompe al imprimir UTF-8
# (✓, •, etc.). Forzamos UTF-8 en stdout/stderr para los scripts de CLI.
import sys as _sys

for _stream in ("stdout", "stderr"):
    _s = getattr(_sys, _stream, None)
    if _s is not None and hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
