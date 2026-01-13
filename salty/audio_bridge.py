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

    def __init__(self, robot: ReachyMini, handler, movement_mgr=None):
        """Initialize the audio bridge.

        Args:
            robot: Initialized ReachyMini instance
            handler: AsyncStreamHandler instance (e.g., RealtimeHandler)
            movement_mgr: Optional MovementManager instance
        """
        self.robot = robot
        self.handler = handler
        self.movement_mgr = movement_mgr
        self._stop_event = asyncio.Event()
        self._playing = False  # Track if speaker is active
        self._assistant_speaking = False  # Track if assistant is speaking
        self._muted = False  # Track if audio output is muted

    async def start(self) -> None:
        """Start the audio bridge and handler."""
        # Start robot audio
        self.robot.media.start_recording()
        logger.info("Reachy audio devices started")

        # Start handler
        await self.handler.start_up()

        # Hook up interruption callback
        self.handler.on_interrupt = self.interrupt_playback

        # Hook up listening state callbacks for movement manager
        if self.movement_mgr:
            self.handler.on_user_speaking_start = lambda: self.movement_mgr.set_listening(True)
            self.handler.on_user_speaking_stop = lambda: self.movement_mgr.set_listening(False)

        # Start audio pump tasks
        asyncio.create_task(self._microphone_loop(), name="microphone-loop")
        asyncio.create_task(self._speaker_loop(), name="speaker-loop")

    def set_muted(self, muted: bool) -> None:
        """Set muted state for audio output.

        Args:
            muted: If True, audio output is silenced but LM continues running
        """
        self._muted = muted
        if muted and self._playing:
            logger.debug("Muting - stopping speaker")
            self.robot.media.stop_playing()
            self._playing = False
        logger.info(f"Audio output {'muted' if muted else 'unmuted'}")

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
        last_movement_update = 0.0

        try:
            while not self._stop_event.is_set():
                # Get audio from handler (this blocks until audio is available)
                audio_frame = await self.handler.emit()

                if audio_frame is not None:
                    sample_rate, audio_data = audio_frame

                    if first_frame:
                        logger.info(f"First audio frame: rate={sample_rate}Hz, shape={audio_data.shape}, dtype={audio_data.dtype}, len={len(audio_data.flatten())}")
                        first_frame = False

                    # If muted, consume audio but don't play it
                    if self._muted:
                        # Still track speaking state for consistency
                        if not self._assistant_speaking and self.movement_mgr:
                            self._assistant_speaking = True
                        continue

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

                        # Notify movement manager
                        if not self._assistant_speaking and self.movement_mgr:
                            self._assistant_speaking = True
                            self.movement_mgr.start_assistant_speaking()

                    # Play on robot
                    self.robot.media.push_audio_sample(audio_data)

                    # Update speech movements periodically (every 100ms)
                    if self.movement_mgr:
                        current_time = asyncio.get_event_loop().time()
                        if current_time - last_movement_update >= 0.1:
                            self.movement_mgr.update_speaking_movements()
                            last_movement_update = current_time

                else:
                    # No audio - assistant may have stopped speaking
                    if self._assistant_speaking and self.movement_mgr:
                        self._assistant_speaking = False
                        self.movement_mgr.stop_assistant_speaking()

        except Exception as e:
            logger.error(f"Speaker loop error: {e}")
        finally:
            # Clean up speaking state
            if self._assistant_speaking and self.movement_mgr:
                self.movement_mgr.stop_assistant_speaking()
            logger.debug("Speaker loop stopped")
