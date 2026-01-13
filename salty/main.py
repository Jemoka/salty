"""Reachy Mini conversation app using OpenAI Realtime API."""

import asyncio
import logging
import threading
from typing import Any

from reachy_mini import ReachyMini, ReachyMiniApp

from salty.realtime import RealtimeConfig, RealtimeHandler
from salty.audio_bridge import ReachyAudioBridge
from salty.movement_manager import MovementManager, EmotionMove
from salty.tool import Tool


logger = logging.getLogger("salty")


# Global references (set in _async_run)
_movement_manager = None
_audio_bridge = None


def express_emotion(emotion: str) -> dict:
    """Express an emotion through body language.

    Built-in tool for embodied interaction.
    Available emotions: happy, excited, curious, thinking, surprised, sad
    """
    if _movement_manager is None:
        return {"error": "Movement manager not available"}

    valid_emotions = ["happy", "excited", "curious", "thinking", "surprised", "sad"]
    emotion_lower = emotion.lower()

    if emotion_lower not in valid_emotions:
        return {"error": f"Unknown emotion '{emotion}'. Valid: {', '.join(valid_emotions)}"}

    try:
        move = EmotionMove(emotion_lower, duration=2.0)
        _movement_manager.queue_move(move)
        logger.info(f"Queued emotion: {emotion_lower}")
        return {"success": True, "emotion": emotion_lower}
    except Exception as e:
        logger.error(f"Failed to express emotion {emotion_lower}: {e}")
        return {"error": str(e)}


def mute() -> dict:
    """Mute audio output.

    Built-in tool to silence audio playback while LM continues running.
    """
    if _audio_bridge is None:
        return {"error": "Audio bridge not available"}

    try:
        _audio_bridge.set_muted(True)
        return {"success": True, "message": "Audio muted"}
    except Exception as e:
        logger.error(f"Failed to mute: {e}")
        return {"error": str(e)}


def unmute() -> dict:
    """Unmute audio output.

    Built-in tool to resume audio playback.
    """
    if _audio_bridge is None:
        return {"error": "Audio bridge not available"}

    try:
        _audio_bridge.set_muted(False)
        return {"success": True, "message": "Audio unmuted"}
    except Exception as e:
        logger.error(f"Failed to unmute: {e}")
        return {"error": str(e)}


# Built-in tools
EMOTION_TOOL = Tool(
    name="express_emotion",
    description="Express an emotion through body language to show engagement. Use during conversation to be more expressive.",
    function=express_emotion,
    parameters={
        "emotion": {
            "type": "string",
            "enum": ["happy", "excited", "curious", "thinking", "surprised", "sad"],
            "description": "The emotion to express",
        }
    },
    required=["emotion"],
)

MUTE_TOOL = Tool(
    name="mute",
    description="Mute audio output. The LM continues running and responding, but audio is not played.",
    function=mute,
)

UNMUTE_TOOL = Tool(
    name="unmute",
    description="Unmute audio output to resume playback.",
    function=unmute,
)

BUILTIN_TOOLS = [EMOTION_TOOL, MUTE_TOOL, UNMUTE_TOOL]


class ReachyConversationApp(ReachyMiniApp):  # type: ignore[misc]
    """Reachy Mini conversation app with OpenAI Realtime API."""

    def __init__(
        self,
        instructions: str,
        voice: str = "cedar",
        api_key: str | None = None,
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ):
        """Initialize the conversation app.

        Args:
            instructions: System instructions for the assistant
            voice: Voice for speech synthesis
            api_key: OpenAI API key (if None, reads from .env)
            tools: List of Tool instances to register
            **kwargs: Additional RealtimeConfig parameters
        """
        super().__init__()
        self.instructions = instructions
        self.voice = voice
        self.api_key = api_key
        self.config_kwargs = kwargs

        # Combine user tools with built-in tools
        all_tools = BUILTIN_TOOLS.copy()
        if tools:
            all_tools.extend(tools)

        # Build tool specs and function registry
        self.tool_specs = [tool.build() for tool in all_tools]
        self.tool_functions = {tool.name: tool.function for tool in all_tools}

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
        global _movement_manager, _audio_bridge

        # Create movement manager
        movement_mgr = MovementManager(robot)
        movement_mgr.start()
        _movement_manager = movement_mgr
        logger.info("Movement manager started")

        # Configure OpenAI with tools
        config = RealtimeConfig(
            api_key=self.api_key,
            instructions=self.instructions,
            voice=self.voice,
            tools=self.tool_specs,
            tool_functions=self.tool_functions,
            **self.config_kwargs,
        )

        # Create handler
        handler = RealtimeHandler(config)

        # Create audio bridge
        bridge = ReachyAudioBridge(robot, handler, movement_mgr)
        _audio_bridge = bridge

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


def build(
    instructions: str,
    voice: str = "cedar",
    api_key: str | None = None,
    tools: list[Tool] | None = None,
    **kwargs: Any,
) -> ReachyConversationApp:
    """Build a Reachy conversation app.

    Args:
        instructions: System instructions for the assistant
        voice: Voice for speech synthesis (alloy, echo, fable, onyx, nova, shimmer, cedar)
        api_key: OpenAI API key (if None, reads from .env)
        tools: List of Tool instances to register
        **kwargs: Additional RealtimeConfig parameters (e.g., temperature, turn_detection_threshold)

    Returns:
        ReachyConversationApp instance ready to run
    """
    return ReachyConversationApp(
        instructions=instructions,
        voice=voice,
        api_key=api_key,
        tools=tools,
        **kwargs,
    )
