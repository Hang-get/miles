import socket
from contextlib import asynccontextmanager

from aiohttp import web
from tinker_cookbook.capture.proxy.app import ProxyDeps, make_app
from tinker_cookbook.capture.proxy.serve import validate_bind

from tinker_agent.recording import RecordingSampler


@asynccontextmanager
async def serve_proxy(
    renderer, recorder: RecordingSampler, *, model_label: str, default_max_tokens: int, host="127.0.0.1", auth_token=None
):
    auth_token = validate_bind(host, auth_token)
    app = make_app(
        ProxyDeps(
            renderer=renderer,
            sampling_client=recorder,
            model_label=model_label,
            default_max_tokens=default_max_tokens,
        ),
        auth_token=auth_token,
    )
    runner = web.AppRunner(app, handler_cancellation=False)
    await runner.setup()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        try:
            listener.bind((host, 0))
            listener.setblocking(False)
            port = listener.getsockname()[1]
            await web.SockSite(runner, listener).start()
            yield port
        finally:
            await runner.cleanup()
