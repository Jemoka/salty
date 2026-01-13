"""Minimal Reachy Mini conversation app using OpenAI Realtime API."""

import asyncio
import logging
import threading
from datetime import datetime
from pathlib import Path

from reachy_mini import ReachyMini, ReachyMiniApp

from salty.realtime import RealtimeConfig, RealtimeHandler
from salty.audio_bridge import ReachyAudioBridge
from salty.movement_manager import MovementManager


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("salty").setLevel(logging.DEBUG)
logger = logging.getLogger("salty")


# ============================================================================
# Define your tools here
# ============================================================================

def save_note(content: str) -> dict:
    """Save a note to the meeting notes file."""
    notes_file = Path("notes.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with notes_file.open("a") as f:
        f.write(f"[{timestamp}] {content}\n")

    return {"success": True, "message": f"Saved note: {content}"}


def get_notes() -> dict:
    """Get all saved notes from the meeting."""
    notes_file = Path("notes.txt")

    if not notes_file.exists():
        return {"notes": ""}

    with notes_file.open("r") as f:
        notes = f.read()

    return {"notes": notes}


# Tool specifications in OpenAI format
TOOLS = [
    {
        "type": "function",
        "name": "save_note",
        "description": "Save a note to the meeting notes file",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The note content to save"
                }
            },
            "required": ["content"]
        }
    },
    {
        "type": "function",
        "name": "get_notes",
        "description": "Get all previously saved notes from the meeting",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]

# Map tool names to functions
TOOL_FUNCTIONS = {
    "save_note": save_note,
    "get_notes": get_notes,
}


# ============================================================================
# App
# ============================================================================

class ReachyConversationApp(ReachyMiniApp):  # type: ignore[misc]
    """Reachy Mini conversation app with OpenAI Realtime API."""

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Run the conversation app."""
        # Create new event loop to avoid nesting issues
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self._async_run(reachy_mini, stop_event))
        finally:
            loop.close()

    async def _async_run(self, robot: ReachyMini, stop_event: threading.Event) -> None:
        """Main async loop."""
        # Create movement manager
        movement_mgr = MovementManager(robot)
        movement_mgr.start()
        movement_mgr.start_doa_monitoring()
        logger.info("Movement manager started with DoA monitoring")

        # Configure OpenAI with tools
        config = RealtimeConfig(
            instructions="""You are Salty, a friendly Reachy Mini robot assistant for meetings.

Your job is to help take notes during meetings. When someone says something important
that should be written down, use the save_note tool to record it. You can also retrieve
previously saved notes using the get_notes tool when asked.

Be proactive - if you hear action items, decisions, or important points, offer to save them.""",
            voice="cedar",
            tools=TOOLS,
            tool_functions=TOOL_FUNCTIONS,
        )

        # Create handler
        handler = RealtimeHandler(config)

        # Create audio bridge
        bridge = ReachyAudioBridge(robot, handler, movement_mgr)

        # Start bridge
        await bridge.start()

        try:
            # Wait for stop signal
            while not stop_event.is_set():
                await asyncio.sleep(0.1)
        finally:
            await bridge.stop()
            movement_mgr.stop()
            logger.info("Movement manager stopped")


if __name__ == "__main__":
    app = ReachyConversationApp()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
