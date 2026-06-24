"""Server-Sent Events manager -- per-session live log streams.

Each session can have multiple SSE listeners. When a log is emitted,
all connected listeners for that session receive the event.

Thread-safe: emit() can be called from any thread (e.g. asyncio.to_thread
workers). Uses loop.call_soon_threadsafe() to schedule queue writes on the
event loop thread, ensuring await queue.get() wakes up correctly.
"""

import asyncio
import json
import logging
from collections import defaultdict
from typing import AsyncGenerator, Dict, Optional

logger = logging.getLogger("sse")


class SSEManager:
    """Manages per-session SSE event queues (thread-safe)."""

    def __init__(self):
        # session_id -> list of asyncio.Queue
        self._listeners: Dict[str, list] = defaultdict(list)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Capture the main event loop reference (call from startup)."""
        self._loop = loop
        logger.info("[SSE] Event loop registered for thread-safe emit")

    def subscribe(self, session_id: str) -> asyncio.Queue:
        """Create a new listener queue for a session."""
        queue = asyncio.Queue()
        self._listeners[session_id].append(queue)
        count = len(self._listeners[session_id])
        logger.info("[SSE] Client subscribed to %s (total listeners: %d)", session_id[:8], count)
        return queue

    def unsubscribe(self, session_id: str, queue: asyncio.Queue) -> None:
        """Remove a listener queue."""
        queues = self._listeners.get(session_id, [])
        if queue in queues:
            queues.remove(queue)
        remaining = len(queues)
        logger.info("[SSE] Client unsubscribed from %s (remaining: %d)", session_id[:8], remaining)
        if not queues and session_id in self._listeners:
            del self._listeners[session_id]

    def emit(self, session_id: str, data: dict) -> None:
        """Send an event to all listeners for a session.

        Thread-safe: detects if called from a background thread and uses
        call_soon_threadsafe() to schedule the queue write on the event loop.
        """
        queues = self._listeners.get(session_id, [])
        if not queues:
            return

        # Detect if we are on the event loop thread or a worker thread
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            # Called from event loop thread -- direct put is safe
            for queue in queues:
                try:
                    queue.put_nowait(data)
                except asyncio.QueueFull:
                    pass
        else:
            # Called from a worker thread (asyncio.to_thread) --
            # schedule on the event loop so queue.get() wakes up
            if self._loop is None or self._loop.is_closed():
                logger.warning("[SSE] emit from thread but no event loop -- dropping event for %s", session_id[:8])
                return

            def _put():
                for q in list(queues):
                    try:
                        q.put_nowait(data)
                    except asyncio.QueueFull:
                        pass

            self._loop.call_soon_threadsafe(_put)

    async def stream(self, session_id: str) -> AsyncGenerator[str, None]:
        """Yield SSE-formatted events for a session.

        Keepalive pings are handled by sse_starlette's built-in ping parameter
        (set ping=15 on EventSourceResponse).
        """
        queue = self.subscribe(session_id)
        logger.info("[SSE] Stream opened for %s", session_id[:8])
        try:
            while True:
                data = await queue.get()
                yield json.dumps(data)
        except asyncio.CancelledError:
            logger.info("[SSE] Stream cancelled for %s (client disconnected or proxy killed)", session_id[:8])
        except Exception as e:
            logger.error("[SSE] Stream error for %s: %s", session_id[:8], e)
        finally:
            self.unsubscribe(session_id, queue)
            logger.info("[SSE] Stream closed for %s", session_id[:8])


# Singleton
sse_manager = SSEManager()
