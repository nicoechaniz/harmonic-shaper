"""Standalone FastAPI and WebSocket state surface for the Shaper."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional, Set

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    import uvicorn

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from . import config
from .state import VoiceParameterStore

log = logging.getLogger(__name__)


def create_app(store: VoiceParameterStore) -> "FastAPI":
    """Create the extracted Shaper-only HTTP/WebSocket application."""

    if not HAS_FASTAPI:
        raise ImportError("fastapi and uvicorn are required for the web API")

    class _WsManager:
        def __init__(self) -> None:
            self._connections: Set[WebSocket] = set()

        async def connect(self, ws: WebSocket) -> None:
            await ws.accept()
            self._connections.add(ws)

        def disconnect(self, ws: WebSocket) -> None:
            self._connections.discard(ws)

        async def broadcast(self, data: dict) -> None:
            dead: Set[WebSocket] = set()
            for ws in self._connections:
                try:
                    await ws.send_json(data)
                except Exception:
                    dead.add(ws)
            self._connections -= dead

    ws_manager = _WsManager()
    event_loop: Optional[asyncio.AbstractEventLoop] = None

    @asynccontextmanager
    async def _lifespan(app_instance):
        del app_instance
        nonlocal event_loop
        event_loop = asyncio.get_running_loop()
        yield
        event_loop = None

    app = FastAPI(title="Harmonic Shaper", version="0.1.0", lifespan=_lifespan)

    def _on_change() -> None:
        if event_loop is None or not event_loop.is_running():
            return
        try:
            asyncio.run_coroutine_threadsafe(
                ws_manager.broadcast(store.to_dict()),
                event_loop,
            )
        except Exception:
            log.exception("WebSocket state broadcast failed")

    # This is the state_sync mechanism documented by the v1 manifest.
    store._on_change = _on_change

    @app.get("/")
    async def root() -> dict:
        return {
            "instrument": "harmonic-shaper",
            "wire_namespace": config.WIRE_OSC_NAMESPACE,
            "planned_namespace": config.PLANNED_OSC_NAMESPACE,
            "state": "/api/state",
            "websocket": "/ws",
            "docs": "/docs",
        }

    @app.get("/api/state")
    async def get_state() -> dict:
        return store.to_dict()

    @app.post("/api/panic")
    async def panic() -> dict:
        store.panic()
        return {"ok": True, "action": "panic"}

    # Must be registered before /api/shaper/{n}/{param}.
    @app.post("/api/shaper/global/{param}")
    async def set_shaper_global(param: str, body: dict) -> dict:
        if param == "lfo_waveform":
            value = str(body.get(param, "sine"))
            store.set_lfo_waveform(value)
            return {"ok": True, "param": param, "value": value}

        # Dedicated bodies: clock uses {"bpm": ...}, settle uses {"beats": ...}.
        if param == "clock":
            try:
                value = float(body.get("bpm", body.get("clock_bpm", 0.0)))
            except (TypeError, ValueError) as exc:
                raise HTTPException(400, "invalid value for clock bpm") from exc
            store.set_clock_bpm(value)
            return {
                "ok": True,
                "param": param,
                "bpm": store.get_clock_bpm(),
            }
        if param == "settle":
            try:
                value = float(body.get("beats", body.get("settle_beats", 0.0)))
            except (TypeError, ValueError) as exc:
                raise HTTPException(400, "invalid value for settle beats") from exc
            store.set_settle_beats(value)
            return {
                "ok": True,
                "param": param,
                "beats": store.get_settle_beats(),
            }

        try:
            value = float(body.get(param, 0.0))
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, f"invalid value for {param}") from exc

        if param == "attack":
            store.set_global_attack(value)
        elif param == "release":
            store.set_global_release(value)
        elif param == "master":
            store.set_master_gain(value)
        elif param == "sidechain":
            store.set_sidechain_amount(value)
        elif param == "lfo_rate_divisor":
            store.set_lfo_rate_divisor(int(value))
        elif param == "lfo_amount":
            store.set_lfo_amount(value)
        elif param == "ceiling":
            # Body level 0..1 → integer partial_ceiling 1..32.
            store.set_partial_ceiling_from_level(value)
            return {
                "ok": True,
                "param": param,
                "value": value,
                "partial_ceiling": store.get_partial_ceiling(),
            }
        elif param == "clock_bpm":
            store.set_clock_bpm(value)
            return {"ok": True, "param": param, "value": store.get_clock_bpm()}
        elif param == "settle_beats":
            store.set_settle_beats(value)
            return {"ok": True, "param": param, "value": store.get_settle_beats()}
        elif param == "generator_enable":
            store.set_generator_enable(int(value))
            return {
                "ok": True,
                "param": param,
                "value": int(store.get_generator_enable()),
            }
        else:
            raise HTTPException(400, f"unknown global param: {param}")
        return {"ok": True, "param": param, "value": value}

    @app.post("/api/shaper/{n}/{param}")
    async def set_shaper_param(n: int, param: str, body: dict) -> dict:
        if not 1 <= n <= config.N_BANDS:
            raise HTTPException(400, f"n must be 1..{config.N_BANDS}")
        try:
            value = float(body.get(param, 0.0))
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, f"invalid value for {param}") from exc

        setters = {
            "gain": store.set_gain,
            "pan": store.set_pan,
            "phase_deg": store.set_phase,
            "attack_s": store.set_attack,
            "release_s": store.set_release,
            "shape": store.set_shape,
            "lfo_gain": store.set_lfo_gain,
            "lfo_pan": store.set_lfo_pan,
            "lfo_phase": store.set_lfo_phase,
        }
        setter = setters.get(param)
        if setter is None:
            raise HTTPException(400, f"unknown param: {param}")
        setter(n, value)
        return {"ok": True, "n": n, "param": param, "value": value}

    @app.websocket("/ws")
    async def websocket_state(ws: WebSocket) -> None:
        await ws_manager.connect(ws)
        await ws.send_json(store.to_dict())
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            ws_manager.disconnect(ws)

    return app


def run_server(
    store: VoiceParameterStore,
    host: str = config.API_HOST,
    port: int = config.API_PORT,
) -> None:
    """Run the state API in the current thread."""

    if not HAS_FASTAPI:
        raise ImportError("fastapi and uvicorn are required for the web API")
    uvicorn.run(create_app(store), host=host, port=port, log_level="warning")
