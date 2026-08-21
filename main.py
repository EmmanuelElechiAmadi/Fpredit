"""Vercel entrypoint — re-exports the FastAPI app.

Vercel's Python runtime detects an ASGI entrypoint named ``main.py`` / ``app.py``
etc. at the project root. The real application lives in ``app/main.py``, so this
shim simply re-exports it for Vercel's function runtime.
"""

from app.main import app  # noqa: F401
