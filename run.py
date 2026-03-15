"""
run.py

Project root entry point for Windows development.

Problem: Python on Windows defaults to ProactorEventLoop, but psycopg's
async driver requires SelectorEventLoop. The policy MUST be set before
any SQLAlchemy/uvicorn imports — once create_async_engine() runs, the
engine is already bound to whatever loop was active at import time.

Usage:
    python run.py              # default: port 8000, reload on
    python run.py --no-reload  # reload off (faster startup)
    python run.py --port 8001  # custom port

Production (Linux/Docker): use uvicorn directly — no policy needed:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
"""

import asyncio
import sys

# ── MUST be before all other imports ─────────────────────────────────────────
# Sets SelectorEventLoop as the global policy so that psycopg async,
# SQLAlchemy async engine, and LangGraph ainvoke() all run on the
# correct loop. ProactorEventLoop (Windows default) is incompatible
# with psycopg's socket-readiness-based async implementation.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ── Now safe to import everything else ───────────────────────────────────────
import argparse
import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI dev server")
    parser.add_argument("--host",      default="0.0.0.0",  help="Bind host")
    parser.add_argument("--port",      default=8000, type=int, help="Bind port")
    parser.add_argument("--no-reload", action="store_true",   help="Disable auto-reload")
    args = parser.parse_args()

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
    )

    