"""
run.py — definitive Windows fix using manual asyncio.run() with SelectorEventLoop.
Bypasses uvicorn's loop management entirely.
"""

import sys
import asyncio

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import uvicorn

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host",   default="127.0.0.1")
    parser.add_argument("--port",   default=8000, type=int)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    if args.reload:
        # --reload mode: uvicorn manages the loop — use the CLI flag workaround
        import subprocess
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "app.main:app",
            "--host", args.host,
            "--port", str(args.port),
            "--reload",
            "--loop", "asyncio",   # only works in some uvicorn versions
        ])
    else:
        # Production mode: use asyncio.run() with explicit SelectorEventLoop
        # This gives us FULL control over the loop before uvicorn sees it
        async def _serve():
            config = uvicorn.Config(
                "app.main:app",
                host=args.host,
                port=args.port,
                loop="none",   # tells uvicorn NOT to manage the loop
            )
            server = uvicorn.Server(config)
            await server.serve()

        # Run with a SelectorEventLoop we created — uvicorn cannot override this
        loop = asyncio.SelectorEventLoop()
        asyncio.set_event_loop(loop)
        print(f"[run.py] Running on {type(loop).__name__}")
        try:
            loop.run_until_complete(_serve())
        finally:
            loop.close()