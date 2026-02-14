"""
Attention Dynamics for Conscious Agent.

Manages agent attention activation with decay bias and delta boosts.
Implements training wheels approach for gradual autonomy progression.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class AttentionDynamics:
    """
    Manages agent attention activation with natural decay bias and delta boosts.

    Core Principles:
    - Decay bias: Attention naturally decreases over time (agent wants to relax)
    - Delta boost: New events increase attention (wakes agent up)
    - Training wheels: User-set bounds that relax as autonomy increases
    - Emergent check-in scheduling: Based on activation level, not pre-programmed

    Attention Activation Scale:
    - 0.0: Deeply relaxed (check in every hour or more)
    - 0.5: Medium alertness (check in every few minutes)
    - 1.0: Fully alert (check in every 30 seconds)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize attention dynamics.

        Args:
            config: Configuration dictionary with optional fields:
                - initial_activation: Starting activation level (default: 0.5)
                - decay_rate: Decay rate per cycle (default: 0.05 = 5% decay)
                - boost_factor: Boost multiplier from deltas (default: 0.3)
                - min_activation: Minimum activation (training wheels, default: 0.1)
                - max_activation: Maximum activation (training wheels, default: 1.0)
                - autonomy_level: Current autonomy level (default: 0.3 = 30%)
                - max_check_in_delay: Maximum delay when fully relaxed (default: 3600s = 1 hour)
                - min_check_in_delay: Minimum delay when fully alert (default: 30s)
        """
        config = config or {}

        # Core dynamics
        self.activation = config.get('initial_activation', 0.5)
        self.decay_rate = config.get('decay_rate', 0.05)
        self.boost_factor = config.get('boost_factor', 0.3)

        # Training wheels (user-set bounds)
        self.min_activation = config.get('min_activation', 0.1)
        self.max_activation = config.get('max_activation', 1.0)
        self.autonomy_level = config.get('autonomy_level', 0.3)

        # Check-in delay parameters
        self.max_check_in_delay = config.get('max_check_in_delay', 3600)  # 1 hour
        self.min_check_in_delay = config.get('min_check_in_delay', 30)    # 30 seconds

        logger.info(f"[AttentionDynamics] Initialized: activation={self.activation:.2f}, "
                   f"autonomy={self.autonomy_level:.2f}, decay_rate={self.decay_rate:.2f}")

    def update_on_cycle(self):
        """
        Apply natural decay bias (agent wants to relax).

        Call this at the end of each decision cycle to allow attention to decay.
        """
        old_activation = self.activation

        # Apply exponential decay
        self.activation *= (1 - self.decay_rate)

        # Enforce training wheels (minimum activation)
        self.activation = max(self.min_activation, self.activation)

        decay_amount = old_activation - self.activation

        if decay_amount > 0.01:  # Only log significant decays
            logger.debug(f"[AttentionDynamics] Decay: {old_activation:.3f} → {self.activation:.3f} "
                        f"(-{decay_amount:.3f})")

    def boost_from_delta(self, delta_magnitude: float):
        """
        Boost activation from new events (delta wakes agent up).

        Args:
            delta_magnitude: Magnitude of delta (0.0-1.0)
                - 0.0: No change
                - 0.5: Moderate change (e.g., 50% change in object count)
                - 1.0: Major change (e.g., entirely new scene)
        """
        old_activation = self.activation

        # Calculate boost
        boost = delta_magnitude * self.boost_factor

        # Apply boost and enforce training wheels (maximum activation)
        self.activation = min(self.max_activation, self.activation + boost)

        boost_amount = self.activation - old_activation

        if boost_amount > 0.01:  # Only log significant boosts
            logger.info(f"[AttentionDynamics] Delta boost: {old_activation:.3f} → {self.activation:.3f} "
                       f"(+{boost_amount:.3f}, delta_magnitude={delta_magnitude:.2f})")

    def get_check_in_seconds(self) -> int:
        """
        Calculate next check-in delay based on activation level.

        Higher activation → shorter delay (more frequent check-ins)
        Lower activation → longer delay (less frequent check-ins)

        Formula: delay = max_delay * (1 - activation) + min_delay

        Examples:
        - activation=0.0 → 3600s (1 hour when fully relaxed)
        - activation=0.5 → 1815s (~30 minutes when medium)
        - activation=1.0 → 30s (30 seconds when fully alert)

        Returns:
            Check-in delay in seconds
        """
        delay = self.max_check_in_delay * (1 - self.activation) + self.min_check_in_delay
        delay_int = int(delay)

        logger.debug(f"[AttentionDynamics] Check-in delay: {delay_int}s (activation={self.activation:.3f})")

        return delay_int

    def set_activation(self, activation: float):
        """
        Manually set activation level (e.g., from manual trigger).

        Args:
            activation: New activation level (0.0-1.0)
        """
        old_activation = self.activation
        self.activation = max(self.min_activation, min(self.max_activation, activation))

        logger.info(f"[AttentionDynamics] Manual set: {old_activation:.3f} → {self.activation:.3f}")

    def increase_autonomy(self, increment: float = 0.05):
        """
        Gradually increase autonomy (remove training wheels).

        Called when system demonstrates reliability.

        Args:
            increment: Amount to increase autonomy (default: 0.05 = 5%)
        """
        old_autonomy = self.autonomy_level
        self.autonomy_level = min(1.0, self.autonomy_level + increment)

        # Relax bounds as autonomy increases
        self.min_activation = 0.1 * (1 - self.autonomy_level)  # → 0.0 at full autonomy

        logger.info(f"[AttentionDynamics] Autonomy increased: {old_autonomy:.2f} → {self.autonomy_level:.2f} "
                   f"(min_activation now {self.min_activation:.2f})")

    def decrease_autonomy(self, decrement: float = 0.1):
        """
        Decrease autonomy (tighten training wheels).

        Called when system makes errors or user provides negative feedback.

        Args:
            decrement: Amount to decrease autonomy (default: 0.1 = 10%)
        """
        old_autonomy = self.autonomy_level
        self.autonomy_level = max(0.0, self.autonomy_level - decrement)

        # Tighten bounds as autonomy decreases
        self.min_activation = 0.1 * (1 - self.autonomy_level)

        logger.warning(f"[AttentionDynamics] Autonomy decreased: {old_autonomy:.2f} → {self.autonomy_level:.2f} "
                      f"(min_activation now {self.min_activation:.2f})")

    def get_state(self) -> Dict[str, float]:
        """
        Get current state for logging/debugging.

        Returns:
            Dictionary with current state values
        """
        return {
            "activation": self.activation,
            "autonomy_level": self.autonomy_level,
            "min_activation": self.min_activation,
            "max_activation": self.max_activation,
            "decay_rate": self.decay_rate,
            "boost_factor": self.boost_factor,
            "check_in_seconds": self.get_check_in_seconds()
        }

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"AttentionDynamics(activation={self.activation:.3f}, "
                f"autonomy={self.autonomy_level:.2f}, "
                f"next_check_in={self.get_check_in_seconds()}s)")
