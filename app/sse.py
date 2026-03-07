"""Server-Sent Events manager — per-session live log streams.

Each session can have multiple SSE listeners. When a log is emitted,
all connected listeners for that session receive the event.
"""

import asyncio
import json
from collections import defaultdict
from typing import AsyncGenerator, Dict, Optional


class SSEManager:
    """Manages per-session SSE event queues."""

    def __init__(self):
        # session_id -> list of asyncio.Queue
        self._listeners: Dict[str, list] = defaultdict(list)

    def subscribe(self, session_id: str) -> asyncio.Queue:
        """Create a new listener queue for a session."""
        queue = asyncio.Queue()
        self._listeners[session_id].append(queue)
        return queue

    def unsubscribe(self, session_id: str, queue: asyncio.Queue) -> None:
        """Remove a listener queue."""
        queues = self._listeners.get(session_id, [])
        if queue in queues:
            queues.remove(queue)
        if not queues and session_id in self._listeners:
            del self._listeners[session_id]

    def emit(self, session_id: str, data: dict) -> None:
        """Send an event to all listeners for a session."""
        queues = self._listeners.get(session_id, [])
        for queue in queues:
            try:
                queue.put_nowait(data)
            except asyncio.QueueFull:
                pass  # Drop if queue is full (slow client)

    async def stream(self, session_id: str) -> AsyncGenerator[str, None]:
        """Yield SSE-formatted events for a session. Used by EventSourceResponse."""
        queue = self.subscribe(session_id)
        try:
            while True:
                data = await queue.get()
                yield json.dumps(data)
        except asyncio.CancelledError:
            pass
        finally:
            self.unsubscribe(session_id, queue)


# Singleton
sse_manager = SSEManager()
