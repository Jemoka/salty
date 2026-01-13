"""Bridge between Reachy Mini's audio system and fastrtc handlers."""

import asyncio
import logging
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample
from reachy_mini import ReachyMini


logger = logging.getLogger("salty")


class ReachyAudioBridge:
    """Bridges Reachy Mini's audio hardware to a fastrtc AsyncStreamHandler.

    This adapter allows using Reachy's microphone and speaker with handlers
    that expect the fastrtc interface.
    """

    def __init__(self, robot: ReachyMini, handler):
        """Initialize the audio bridge.

        Args:
            robot: Initialized ReachyMini instance
            handler: AsyncStreamHandler instance (e.g., RealtimeHandler)
        """
        self.robot = robot
        self.handler = handler
        self._stop_event = asyncio.Event()
        self._playing = False  # Track if speaker is active

    async def start(self) -> None:
        """Start the audio bridge and handler."""
        # Start robot audio
        self.robot.media.start_recording()
        logger.info("Reachy audio devices started")

        # Start handler
        await self.handler.start_up()

        # Hook up interruption callback
        self.handler.on_interrupt = self.interrupt_playback

        # Start audio pump tasks
        asyncio.create_task(self._microphone_loop(), name="microphone-loop")
        asyncio.create_task(self._speaker_loop(), name="speaker-loop")

    def interrupt_playback(self) -> None:
        """Interrupt current playback (called when user starts speaking)."""
        if self._playing:
            logger.debug("Interrupting playback - stopping speaker")
            self.robot.media.stop_playing()
            self._playing = False

    async def stop(self) -> None:
        """Stop the audio bridge."""
        self._stop_event.set()

        # Shutdown handler
        await self.handler.shutdown()

        # Stop robot audio
        self.robot.media.stop_recording()
        if self._playing:
            self.robot.media.stop_playing()
        logger.info("Reachy audio devices stopped")

    async def run(self) -> None:
        """Run the audio bridge until stopped (blocking)."""
        await self.start()
        try:
            await self._stop_event.wait()
        finally:
            await self.stop()

    async def _microphone_loop(self) -> None:
        """Continuously read from robot's microphone and send to handler."""
        logger.debug("Microphone loop started")
        robot_input_rate = self.robot.media.get_input_audio_samplerate()
        logger.info(f"Robot input sample rate: {robot_input_rate}Hz")

        try:
            while not self._stop_event.is_set():
                # Get audio from robot
                samples = self.robot.media.get_audio_sample()

                if samples is not None:
                    # Send to handler
                    await self.handler.receive((robot_input_rate, samples))

                # Small yield to prevent blocking
                await asyncio.sleep(0.01)
        except Exception as e:
            logger.error(f"Microphone loop error: {e}")
        finally:
            logger.debug("Microphone loop stopped")

    async def _speaker_loop(self) -> None:
        """Continuously get audio from handler and play on robot's speaker."""
        logger.debug("Speaker loop started")
        robot_output_rate = self.robot.media.get_output_audio_samplerate()
        logger.info(f"Robot output sample rate: {robot_output_rate}Hz")

        first_frame = True

        try:
            while not self._stop_event.is_set():
                # Get audio from handler (this blocks until audio is available)
                audio_frame = await self.handler.emit()

                if audio_frame is not None:
                    sample_rate, audio_data = audio_frame

                    if first_frame:
                        logger.info(f"First audio frame: rate={sample_rate}Hz, shape={audio_data.shape}, dtype={audio_data.dtype}, len={len(audio_data.flatten())}")
                        first_frame = False

                    # Reshape to 1D if needed
                    if audio_data.ndim == 2:
                        audio_data = audio_data.flatten()

                    # Convert int16 to float32 in range [-1.0, 1.0]
                    # Reachy expects normalized float to avoid graininess
                    if audio_data.dtype == np.int16:
                        audio_data = audio_data.astype(np.float32) / 32768.0

                    # Resample if sample rates don't match
                    if sample_rate != robot_output_rate:
                        num_samples_needed = int(len(audio_data) * robot_output_rate / sample_rate)
                        audio_data = resample(audio_data, num_samples_needed).astype(np.float32)
                        logger.debug(f"Resampled {sample_rate}Hz → {robot_output_rate}Hz: {len(audio_data)} samples")

                    # Start playing if not already
                    if not self._playing:
                        self.robot.media.start_playing()
                        self._playing = True
                        logger.debug("Started speaker playback")

                    # Play on robot
                    self.robot.media.push_audio_sample(audio_data)

        except Exception as e:
            logger.error(f"Speaker loop error: {e}")
        finally:
            logger.debug("Speaker loop stopped")
