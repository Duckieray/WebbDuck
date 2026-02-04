"""WebSocket event broadcasting."""

import json

active_sockets = set()


async def broadcast(event: dict):
    """Broadcast event to all connected WebSocket clients."""
    dead = []
    for ws in list(active_sockets):
        try:
            await ws.send_text(json.dumps(event))
        except Exception:
            dead.append(ws)
    for d in dead:
        active_sockets.discard(d)


async def broadcast_state(state):
    """Broadcast state update."""
    await broadcast({
        "type": "state",
        **state
    })