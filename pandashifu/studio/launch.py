from shiny.express import wrap_express_app
from pathlib import Path

import uvicorn
import asyncio
import warnings
import socket

import os
import sys
import time
import socket
import threading
import contextlib


def occupied_port(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def run(port=8010):

    if occupied_port(port):
        raise ValueError(f"The port {port} is invalid or occupied.")

    current_file = Path(__file__)
    directory = current_file.parent

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        app = wrap_express_app(Path(directory/"app.py"))

        print(f"Uvicorn running on: http://localhost:{port}")
        config = uvicorn.Config(app, port=port)
        server = uvicorn.Server(config)
        loop = asyncio.get_running_loop()
        loop.create_task(server.serve())


def _wait_until_ready(host, port, timeout):
    """Poll until the server accepts connections, or give up after `timeout`s."""
    start = time.time()
    while time.time() - start < timeout:
        with contextlib.closing(
            socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ) as sock:
            sock.settimeout(1)
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.5)
    return False


def _run_server(host, port):
    """Run the Shiny Express app with uvicorn (in a background thread)."""
    import uvicorn
    from shiny.express import wrap_express_app

    current_file = Path(__file__)
    directory = current_file.parent
    app_obj = wrap_express_app(Path(directory/"app.py"))
    uvicorn.run(app_obj, host=host, port=port, log_level="warning")


def colab(host="127.0.0.1", port=8000, timeout=60):
    """
    Launch `app.py` on Colab and print a link to open it in a new browser tab.

    Parameters
    ----------
    host, port : where to serve (Colab proxies this for us).
    timeout    : seconds to wait for startup.
    """
    threading.Thread(
        target=_run_server,
        args=(host, port),
        daemon=True,
    ).start()

    print(f"Starting Shiny app on {host}:{port} ...", flush=True)
    if not _wait_until_ready(host, port, timeout):
        raise RuntimeError(
            "Server did not start in time. Check for errors in app.py "
            "or try a different port."
        )

    try:
        from google.colab.output import eval_js
        url = eval_js(f'google.colab.kernel.proxyPort({port})')
    except ImportError:
        url = f"http://{host}:{port}"

    print("Server is up. Open your app here:", flush=True)
    print(f"  {url}", flush=True)