"""OpenAI Realtime API handler for Reachy Mini.

Simple, clean interface for connecting Reachy Mini to OpenAI's Realtime API.
Handles bidirectional audio streaming using fastrtc.
"""

import os
import json
import base64
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Tuple, Literal, Final, Callable, Any, Dict

import numpy as np
from openai import AsyncOpenAI
from scipy.signal import resample
from numpy.typing import NDArray
from dotenv import load_dotenv
from fastrtc import AsyncStreamHandler, wait_for_item, audio_to_int16


logger = logging.getLogger("salty")

OPENAI_SAMPLE_RATE: Final[Literal[24000]] = 24000


@dataclass
class RealtimeConfig:
    """Configuration for OpenAI Realtime API connection.

    Attributes:
        api_key: OpenAI API key. If None, will read from OPENAI_API_KEY in .env file.
                Get one at https://platform.openai.com/api-keys
        model: OpenAI model to use. Default is "gpt-4o-realtime-preview-2024-12-17"
        voice: Voice for speech synthesis. Options: "alloy", "echo", "fable", "onyx", "nova", "shimmer"
        instructions: System instructions that define the assistant's personality and behavior.
                     This is like the system prompt - it tells the model how to behave.
        turn_detection_threshold: Sensitivity for detecting when user stops speaking (0.0-1.0).
                                 Lower = more sensitive. Default: 0.5
        turn_detection_silence_duration: How long to wait (ms) after speech before considering turn complete.
                                        Default: 500ms
        temperature: Sampling temperature for responses (0.0-2.0). Higher = more creative. Default: 0.8
        tools: List of tool specifications in OpenAI format. See example in main.py.
        tool_functions: Dict mapping tool names to their implementation functions.
    """

    api_key: str | None = None
    model: str = "gpt-realtime"
    voice: str = "nova"
    instructions: str = "You are a helpful assistant."
    turn_detection_threshold: float = 0.5
    turn_detection_silence_duration: int = 500
    temperature: float = 0.8
    tools: list[Dict[str, Any]] = field(default_factory=list)
    tool_functions: Dict[str, Callable] = field(default_factory=dict)


class RealtimeHandler(AsyncStreamHandler):
    """Handles OpenAI Realtime API connection using fastrtc.

    This handler extends AsyncStreamHandler and works with fastrtc.Stream to manage
    audio I/O with OpenAI's Realtime API.
    """

    def __init__(self, config: RealtimeConfig):
        """Initialize the handler.

        Args:
            config: Configuration for the Realtime API

        Raises:
            ValueError: If api_key is not provided and OPENAI_API_KEY not found in .env
        """
        super().__init__(
            expected_layout="mono",
            output_sample_rate=OPENAI_SAMPLE_RATE,
            input_sample_rate=OPENAI_SAMPLE_RATE,
        )

        self.config = config

        # Load API key from .env if not provided
        api_key = config.api_key
        if api_key is None:
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "No API key provided. Either pass api_key to RealtimeConfig or "
                    "set OPENAI_API_KEY in .env file"
                )

        self.client = AsyncOpenAI(api_key=api_key)
        self.connection = None

        # Audio output queue for fastrtc
        self.output_queue: asyncio.Queue[Tuple[int, NDArray[np.int16]]] = asyncio.Queue()

        # Transcription callbacks
        self.on_user_transcript: callable | None = None
        self.on_assistant_transcript: callable | None = None
        self.on_interrupt: callable | None = None
        self.on_user_speaking_start: callable | None = None
        self.on_user_speaking_stop: callable | None = None

        # Tool registry
        self.tools: Dict[str, Callable] = config.tool_functions
        self.tool_specs = config.tools

    def _empty(self) -> None:
        """Clear the audio output queue (for interruption)."""
        cleared = 0
        try:
            while True:
                self.output_queue.get_nowait()
                cleared += 1
        except asyncio.QueueEmpty:
            pass
        if cleared > 0:
            logger.debug(f"Audio queue cleared ({cleared} items)")

    def copy(self) -> "RealtimeHandler":
        """Create a copy of the handler."""
        return RealtimeHandler(self.config)

    async def start_up(self) -> None:
        """Connect to OpenAI Realtime API and configure session."""
        logger.info("Connecting to OpenAI Realtime API...")

        self.connection = await self.client.realtime.connect(model=self.config.model).__aenter__()

        # Configure session
        session_config: Dict[str, Any] = {
            "type": "realtime",
            "instructions": self.config.instructions,
            "audio": {
                "input": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": OPENAI_SAMPLE_RATE,
                    },
                    "transcription": {
                        "model": "gpt-4o-transcribe",
                        "language": "en"
                    },
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": self.config.turn_detection_threshold,
                        "silence_duration_ms": self.config.turn_detection_silence_duration,
                    },
                },
                "output": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": OPENAI_SAMPLE_RATE,
                    },
                    "voice": self.config.voice,
                },
            }
            # "temperature": self.config.temperature,
        }

        # Add tools if any
        if self.tool_specs:
            session_config["tools"] = self.tool_specs  # type: ignore[typeddict-item]
            session_config["tool_choice"] = "auto"

        await self.connection.session.update(session=session_config)

        logger.info(f"Connected with voice={self.config.voice}, model={self.config.model}")
        if self.tool_specs:
            tool_names = [tool["name"] for tool in self.tool_specs]
            logger.info(f"Registered tools: {', '.join(tool_names)}")

        # Start event handler
        asyncio.create_task(self._event_loop())

    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive audio from microphone and send to OpenAI.

        Called by fastrtc Stream with audio from the microphone.

        Args:
            frame: Tuple of (sample_rate, audio_data)
        """
        if not self.connection:
            return

        input_sample_rate, audio_frame = frame

        # Reshape to mono if needed
        if audio_frame.ndim == 2:
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            if audio_frame.shape[1] > 1:
                audio_frame = audio_frame[:, 0]

        # Resample if needed
        if self.input_sample_rate != input_sample_rate:
            audio_frame = resample(
                audio_frame,
                int(len(audio_frame) * self.input_sample_rate / input_sample_rate)
            )

        # Convert to int16
        audio_frame = audio_to_int16(audio_frame)

        # Send to OpenAI
        try:
            audio_b64 = base64.b64encode(audio_frame.tobytes()).decode("utf-8")
            await self.connection.input_audio_buffer.append(audio=audio_b64)
        except Exception as e:
            logger.debug(f"Dropping audio frame: {e}")

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | None:
        """Emit audio to be played by the speaker.

        Called by fastrtc Stream to get audio for the speaker.

        Returns:
            Tuple of (sample_rate, audio_data) or None
        """
        return await wait_for_item(self.output_queue)  # type: ignore[return-value]

    async def shutdown(self) -> None:
        """Shutdown and disconnect from OpenAI."""
        if self.connection:
            try:
                await self.connection.close()
            except Exception as e:
                logger.debug(f"Connection close error: {e}")
            finally:
                self.connection = None

        # Clear queue
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        logger.info("Disconnected from OpenAI Realtime API")

    async def _event_loop(self) -> None:
        """Internal event loop that processes OpenAI events."""
        try:
            async for event in self.connection:
                logger.debug(f"Event: {event.type}")

                # User started speaking - interrupt playback
                if event.type == "input_audio_buffer.speech_started":
                    logger.debug("User started speaking - interrupting playback")
                    self._empty()
                    if self.on_interrupt:
                        self.on_interrupt()
                    if self.on_user_speaking_start:
                        self.on_user_speaking_start()

                # User stopped speaking
                if event.type == "input_audio_buffer.speech_stopped":
                    logger.debug("User stopped speaking")
                    if self.on_user_speaking_stop:
                        self.on_user_speaking_stop()

                # User transcript (completed)
                if event.type == "conversation.item.input_audio_transcription.completed":
                    logger.info(f"User: {event.transcript}")
                    if self.on_user_transcript:
                        self.on_user_transcript(event.transcript)

                # Assistant transcript
                if event.type in ("response.audio_transcript.done", "response.output_audio_transcript.done"):
                    logger.info(f"Assistant: {event.transcript}")
                    if self.on_assistant_transcript:
                        self.on_assistant_transcript(event.transcript)

                # Audio from assistant
                if event.type in ("response.audio.delta", "response.output_audio.delta"):
                    audio_bytes = base64.b64decode(event.delta)
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16).reshape(1, -1)
                    await self.output_queue.put((OPENAI_SAMPLE_RATE, audio_array))

                # Tool call
                if event.type == "response.function_call_arguments.done":
                    tool_name = getattr(event, "name", None)
                    args_json = getattr(event, "arguments", None)
                    call_id = getattr(event, "call_id", None)

                    if tool_name and args_json and call_id:
                        logger.info(f"Tool call: {tool_name}({args_json})")

                        # Execute tool
                        try:
                            tool_func = self.tools.get(tool_name)
                            if tool_func:
                                args = json.loads(args_json)
                                result = tool_func(**args)
                                logger.info(f"Tool {tool_name} result: {result}")
                            else:
                                result = {"error": f"Tool {tool_name} not found"}
                                logger.error(f"Unknown tool: {tool_name}")
                        except Exception as e:
                            result = {"error": str(e)}
                            logger.error(f"Tool {tool_name} failed: {e}")

                        # Send result back
                        await self.connection.conversation.item.create(
                            item={
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": json.dumps(result),
                            }
                        )

                        # Ask assistant to respond with the result
                        await self.connection.response.create()

                # Errors
                if event.type == "error":
                    err = getattr(event, "error", None)
                    msg = getattr(err, "message", str(err))
                    code = getattr(err, "code", "")

                    logger.error(f"OpenAI error [{code}]: {msg}")

        except Exception as e:
            logger.error(f"Event loop error: {e}")
