"""
Capability router for orchestrating provider invocations.

The router takes capability requests and routes them to registered providers.
It handles parallel execution, A/B evaluation, and error handling.
"""

from typing import List, Tuple, Optional, Dict, Any
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from .registry import CapabilityRegistry
from .m4_token import M4Token


class CapabilityRouter:
    """
    Routes capability requests to registered providers.

    The router:
        1. Looks up providers in the registry
        2. Invokes providers (serially or in parallel)
        3. Handles A/B routing when multiple providers exist
        4. Tracks latency and errors
    """

    def __init__(
        self,
        registry: CapabilityRegistry,
        max_workers: int = 4,
        default_timeout_sec: float = 10.0
    ):
        """
        Initialize router.

        Args:
            registry: CapabilityRegistry to use for lookups
            max_workers: Max parallel workers for multi-capability processing
            default_timeout_sec: Default timeout for provider invocations
        """
        self.registry = registry
        self.max_workers = max_workers
        self.default_timeout_sec = default_timeout_sec

        # Tracking for monitoring
        self.invocation_count = 0
        self.error_count = 0
        self.total_latency_ms = 0.0

    def process_capability(
        self,
        modality: str,
        capability: str,
        input_data: Any,
        line: Optional[str] = None,
        timeout_sec: Optional[float] = None
    ) -> M4Token:
        """
        Process a single capability request.

        Args:
            modality: Input modality (e.g., "audio")
            capability: Capability name (e.g., "speaker_id")
            input_data: Raw data or M4 token
            line: Optional A/B line ("a" or "b")
            timeout_sec: Optional timeout override

        Returns:
            M4Token with annotations from provider

        Raises:
            ValueError: If capability not registered
            TimeoutError: If provider exceeds timeout
            RuntimeError: If provider errors
        """
        # Get provider
        provider = self.registry.get_provider(modality, capability, line=line)
        if provider is None:
            raise ValueError(
                f"No provider registered for ({modality}, {capability})"
                + (f" line={line}" if line else "")
            )

        # Invoke provider with timing
        timeout = timeout_sec if timeout_sec is not None else self.default_timeout_sec
        start_time = time.time()

        try:
            # TODO: Add timeout enforcement
            token = provider.process(input_data)

            # Track metrics
            latency_ms = (time.time() - start_time) * 1000
            self.invocation_count += 1
            self.total_latency_ms += latency_ms

            # Add latency to token metadata if not present
            for ann_key, ann_val in token.annotations.items():
                if isinstance(ann_val, dict) and 'latency_ms' not in ann_val:
                    ann_val['latency_ms'] = latency_ms

            return token

        except Exception as e:
            self.error_count += 1
            raise RuntimeError(
                f"Provider {provider.provider_id} failed for ({modality}, {capability}): {e}"
            ) from e

    def process_multi_capability(
        self,
        capabilities: List[Tuple[str, str]],
        input_data: Any,
        parallel: bool = True,
        timeout_sec: Optional[float] = None
    ) -> List[M4Token]:
        """
        Process multiple capabilities, optionally in parallel.

        Args:
            capabilities: List of (modality, capability) tuples
            input_data: Raw data or M4 token (same input for all)
            parallel: If True, process in parallel; if False, serial
            timeout_sec: Optional timeout override

        Returns:
            List of M4Tokens, one per capability (in same order as input)

        Note:
            Failed capabilities will have None in their position.
            Check token is not None before using.
        """
        if not parallel:
            # Serial processing
            tokens = []
            for modality, capability in capabilities:
                try:
                    token = self.process_capability(
                        modality,
                        capability,
                        input_data,
                        timeout_sec=timeout_sec
                    )
                    tokens.append(token)
                except Exception as e:
                    print(f"Error processing ({modality}, {capability}): {e}")
                    tokens.append(None)
            return tokens

        # Parallel processing
        tokens = [None] * len(capabilities)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_idx = {}
            for idx, (modality, capability) in enumerate(capabilities):
                future = executor.submit(
                    self.process_capability,
                    modality,
                    capability,
                    input_data,
                    timeout_sec=timeout_sec
                )
                future_to_idx[future] = idx

            # Collect results
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                modality, capability = capabilities[idx]
                try:
                    token = future.result()
                    tokens[idx] = token
                except Exception as e:
                    print(f"Error processing ({modality}, {capability}): {e}")
                    tokens[idx] = None

        return tokens

    def process_ab(
        self,
        modality: str,
        capability: str,
        input_data: Any,
        timeout_sec: Optional[float] = None
    ) -> Dict[str, M4Token]:
        """
        Process capability with both A and B line providers.

        This is used for parallel A/B evaluation.

        Args:
            modality: Input modality
            capability: Capability name
            input_data: Raw data or M4 token
            timeout_sec: Optional timeout override

        Returns:
            Dict mapping line ("a", "b") to M4Token

        Raises:
            ValueError: If A/B providers not registered
        """
        ab_providers = self.registry.get_ab_providers(modality, capability)
        if len(ab_providers) < 2:
            raise ValueError(
                f"A/B providers not registered for ({modality}, {capability}). "
                f"Found lines: {list(ab_providers.keys())}"
            )

        results = {}

        with ThreadPoolExecutor(max_workers=len(ab_providers)) as executor:
            # Submit all A/B providers
            future_to_line = {}
            for line, provider in ab_providers.items():
                future = executor.submit(
                    self.process_capability,
                    modality,
                    capability,
                    input_data,
                    line=line,
                    timeout_sec=timeout_sec
                )
                future_to_line[future] = line

            # Collect results
            for future in as_completed(future_to_line):
                line = future_to_line[future]
                try:
                    token = future.result()
                    results[line] = token
                except Exception as e:
                    print(f"Error processing {line}-line: {e}")
                    results[line] = None

        return results

    def merge_tokens(
        self,
        tokens: List[M4Token],
        merge_strategy: str = "combine"
    ) -> M4Token:
        """
        Merge multiple M4 tokens into one.

        Useful when multiple providers contribute annotations for the same input.

        Args:
            tokens: List of M4Tokens to merge (should have same timestamp)
            merge_strategy: How to merge
                - "combine": Combine all annotations (default)
                - "latest": Use annotations from latest token
                - "highest_confidence": Use annotations with highest confidence

        Returns:
            Merged M4Token
        """
        if not tokens:
            raise ValueError("No tokens to merge")

        # Filter out None tokens
        tokens = [t for t in tokens if t is not None]
        if not tokens:
            raise ValueError("All tokens are None")

        # Start with first token
        merged = M4Token(
            modality=tokens[0].modality,
            source="merged",
            timestamp=tokens[0].timestamp,
            duration_ms=tokens[0].duration_ms,
            embedding=tokens[0].embedding,
            features=tokens[0].features.copy(),
            annotations={},
            raw_data=tokens[0].raw_data.copy() if tokens[0].raw_data else {},
            session_id=tokens[0].session_id,
        )

        if merge_strategy == "combine":
            # Combine all annotations
            for token in tokens:
                merged.annotations.update(token.annotations)

        elif merge_strategy == "latest":
            # Use annotations from last token
            merged.annotations = tokens[-1].annotations.copy()

        elif merge_strategy == "highest_confidence":
            # For each capability, use annotation with highest confidence
            capability_best = {}
            for token in tokens:
                for cap, ann in token.annotations.items():
                    conf = ann.get("confidence", 0.0)
                    if cap not in capability_best or conf > capability_best[cap]["confidence"]:
                        capability_best[cap] = ann
            merged.annotations = capability_best

        else:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")

        return merged

    def get_routing_stats(self) -> dict:
        """
        Get router statistics for monitoring.

        Returns:
            Dict with invocation count, error rate, average latency, etc.
        """
        avg_latency = (
            self.total_latency_ms / self.invocation_count
            if self.invocation_count > 0
            else 0.0
        )
        error_rate = (
            self.error_count / self.invocation_count
            if self.invocation_count > 0
            else 0.0
        )

        return {
            "invocation_count": self.invocation_count,
            "error_count": self.error_count,
            "error_rate": error_rate,
            "average_latency_ms": avg_latency,
            "max_workers": self.max_workers,
        }

    def reset_stats(self):
        """Reset routing statistics."""
        self.invocation_count = 0
        self.error_count = 0
        self.total_latency_ms = 0.0

    def __repr__(self) -> str:
        """Human-readable representation."""
        stats = self.get_routing_stats()
        return (
            f"CapabilityRouter("
            f"invocations={stats['invocation_count']}, "
            f"avg_latency={stats['average_latency_ms']:.2f}ms, "
            f"error_rate={stats['error_rate']:.2%})"
        )
