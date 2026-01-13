"""Speech-related movements for the robot.

Provides:
- Subtle sway movements while assistant is speaking
- Face tracking to turn toward user based on Direction of Arrival
"""

import math
import time
import logging
import threading
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
from reachy_mini.motion.move import Move


logger = logging.getLogger("salty")


class SpeechSwayMove(Move):  # type: ignore
    """Subtle sway movement while assistant is speaking."""

    def __init__(self, duration: float = 1.0):
        """Initialize speech sway movement.

        Args:
            duration: Duration of the sway cycle in seconds
        """
        self._duration = duration
        self.sway_amplitude = 0.003  # 3mm gentle sway
        self.sway_frequency = 0.5  # Hz

    @property
    def duration(self) -> float:
        """Duration of one sway cycle."""
        return self._duration

    def evaluate(self, t: float) -> Tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        """Evaluate sway at time t."""
        # Gentle side-to-side head movement
        y_offset = self.sway_amplitude * math.sin(2 * math.pi * self.sway_frequency * t)
        head_pose = create_head_pose(x=0, y=y_offset, z=0, roll=0, pitch=0, yaw=0, degrees=True, mm=False)

        return (head_pose, None, None)


class TurnToUserMove(Move):  # type: ignore
    """Turn to face the user based on Direction of Arrival."""

    def __init__(self, doa_radians: float, current_body_yaw: float = 0.0, turn_duration: float = 0.8):
        """Initialize turn movement.

        Args:
            doa_radians: Direction of arrival in radians (0 = left, π/2 = front, π = right)
            current_body_yaw: Current body yaw position
            turn_duration: Time to complete the turn
        """
        self._duration = turn_duration

        # Convert DoA to body yaw target
        # DoA: 0 = left (robot's -π/2), π/2 = front (0), π = right (π/2)
        # We want to turn toward the sound
        target_yaw = doa_radians - math.pi / 2  # Map DoA to body yaw

        # Normalize to [-π, π]
        while target_yaw > math.pi:
            target_yaw -= 2 * math.pi
        while target_yaw < -math.pi:
            target_yaw += 2 * math.pi

        self.start_yaw = current_body_yaw
        self.target_yaw = target_yaw
        self.yaw_delta = target_yaw - current_body_yaw

    @property
    def duration(self) -> float:
        """Duration of the turn."""
        return self._duration

    def evaluate(self, t: float) -> Tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        """Evaluate turn at time t."""
        # Smooth interpolation
        progress = min(1.0, t / self._duration)
        # Ease in-out
        progress = progress * progress * (3 - 2 * progress)

        current_yaw = self.start_yaw + self.yaw_delta * progress

        return (None, None, current_yaw)


class SpeechMovementManager:
    """Manages robot movements related to speech.

    Handles:
    - Subtle movements while assistant speaks
    - Turning to face user when they speak
    """

    def __init__(self, robot: ReachyMini, movement_manager):
        """Initialize speech movement manager.

        Args:
            robot: ReachyMini instance
            movement_manager: MovementManager instance for queuing moves
        """
        self.robot = robot
        self.movement_manager = movement_manager

        self._assistant_speaking = False
        self._last_sway_time = 0.0
        self._sway_interval = 2.0  # Queue new sway every 2 seconds while speaking

        self._doa_monitor_active = False
        self._doa_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._last_doa: float | None = None
        self._last_turn_time = 0.0
        self._turn_cooldown = 3.0  # Don't turn again for 3 seconds

    def start_assistant_speaking(self) -> None:
        """Called when assistant starts speaking."""
        if not self._assistant_speaking:
            self._assistant_speaking = True
            self._last_sway_time = 0.0  # Reset to trigger immediate sway
            logger.debug("Assistant started speaking")

    def stop_assistant_speaking(self) -> None:
        """Called when assistant stops speaking."""
        if self._assistant_speaking:
            self._assistant_speaking = False
            logger.debug("Assistant stopped speaking")

    def update_speaking_movements(self) -> None:
        """Update movements while assistant is speaking.

        Call this periodically (e.g., every 100ms) to queue sway movements.
        """
        if not self._assistant_speaking:
            return

        current_time = time.time()
        if current_time - self._last_sway_time >= self._sway_interval:
            # Queue a subtle sway movement
            sway = SpeechSwayMove(duration=self._sway_interval)
            self.movement_manager.queue_move(sway)
            self._last_sway_time = current_time
            logger.debug("Queued speech sway movement")

    def start_doa_monitoring(self) -> None:
        """Start monitoring Direction of Arrival to face the user."""
        if self._doa_monitor_active:
            return

        self._stop_event.clear()
        self._doa_monitor_active = True
        self._doa_thread = threading.Thread(target=self._doa_loop, daemon=True)
        self._doa_thread.start()
        logger.info("Started DoA monitoring")

    def stop_doa_monitoring(self) -> None:
        """Stop monitoring Direction of Arrival."""
        if not self._doa_monitor_active:
            return

        self._stop_event.set()
        if self._doa_thread is not None:
            self._doa_thread.join(timeout=1.0)
        self._doa_monitor_active = False
        logger.info("Stopped DoA monitoring")

    def _doa_loop(self) -> None:
        """Background loop to monitor DoA and turn toward user."""
        logger.debug("DoA monitoring loop started")

        while not self._stop_event.is_set():
            try:
                # Get Direction of Arrival
                # 0 radians is left, π/2 radians is front/back, π radians is right
                doa, is_speech_detected = self.robot.media.get_DoA()

                if is_speech_detected:
                    current_time = time.time()

                    # Only turn if enough time has passed since last turn
                    if current_time - self._last_turn_time >= self._turn_cooldown:
                        # Check if DoA has changed significantly
                        if self._last_doa is None or abs(doa - self._last_doa) > 0.2:  # ~11 degrees
                            logger.info(f"User speech detected at DoA: {doa:.2f} radians ({math.degrees(doa):.1f}°)")

                            # Get current body yaw from movement manager status
                            status = self.movement_manager.get_status()
                            current_yaw = status.get("last_commanded_pose", {}).get("body_yaw", 0.0)

                            # Queue turn movement
                            turn_move = TurnToUserMove(doa, current_yaw, turn_duration=0.8)
                            self.movement_manager.queue_move(turn_move)

                            self._last_doa = doa
                            self._last_turn_time = current_time
                            logger.debug(f"Queued turn to face user at {math.degrees(doa):.1f}°")

                # Check every 100ms
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in DoA monitoring: {e}")
                time.sleep(0.5)  # Longer sleep on error

        logger.debug("DoA monitoring loop stopped")
