from shiny.express import wrap_express_app
from pathlib import Path

import uvicorn
import asyncio
#import webbrowser
import socket

def occupied_port(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def run(port=8000):

    if occupied_port(port):
        raise ValueError(f"The port {port} is invalid or occupied.")

    current_file = Path(__file__)
    directory = current_file.parent
    app = wrap_express_app(Path(directory/"app.py"))

    print(f"Uvicorn running on: http://localhost:{port}")
    config = uvicorn.Config(app, port=port)
    server = uvicorn.Server(config)
    loop = asyncio.get_running_loop()
    loop.create_task(server.serve())
