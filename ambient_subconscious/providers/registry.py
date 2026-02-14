"""
Capability registry for managing providers.

The registry maps (modality, capability) pairs to providers.
Multiple providers can register for the same capability (A/B testing).
"""

from typing import Dict, List, Tuple, Optional
from .base import Provider, ProviderAdapter


class CapabilityRegistry:
    """
    Central registry for capability providers.

    Manages mapping from (modality, capability) to providers.
    Supports multiple providers per capability for A/B testing.
    """

    def __init__(self):
        """Initialize empty registry."""
        # Primary mapping: (modality, capability) -> Provider
        self.providers: Dict[Tuple[str, str], Provider] = {}

        # A/B mapping: (modality, capability) -> {"a": Provider, "b": Provider}
        self.ab_providers: Dict[Tuple[str, str], Dict[str, Provider]] = {}

        # Provider ID mapping for lookups
        self.providers_by_id: Dict[str, Provider] = {}

    def register(
        self,
        modality: str,
        capability: str,
        provider: Provider,
        line: Optional[str] = None
    ):
        """
        Register a provider for a capability.

        Args:
            modality: Input modality (e.g., "audio", "vision_rgb")
            capability: Capability name (e.g., "speaker_id", "object_detection")
            provider: Provider instance
            line: Optional A/B line designation ("a" or "b")
                  If None, registers as primary provider

        Example:
            # Register primary provider
            registry.register("audio", "speaker_id", diart_provider)

            # Register A/B providers
            registry.register("audio", "speaker_id", diart_provider, line="a")
            registry.register("audio", "speaker_id", hybrid_provider, line="b")
        """
        key = (modality, capability)

        if line is None:
            # Register as primary
            self.providers[key] = provider
        else:
            # Register as A/B line
            if key not in self.ab_providers:
                self.ab_providers[key] = {}
            self.ab_providers[key][line] = provider

        # Also register by provider ID
        self.providers_by_id[provider.provider_id] = provider

    def get_provider(
        self,
        modality: str,
        capability: str,
        line: Optional[str] = None
    ) -> Optional[Provider]:
        """
        Get provider for a capability.

        Args:
            modality: Input modality
            capability: Capability name
            line: Optional A/B line ("a" or "b")
                  If None, returns primary provider

        Returns:
            Provider instance or None if not registered
        """
        key = (modality, capability)

        if line is None:
            # Get primary provider
            return self.providers.get(key)
        else:
            # Get A/B line provider
            ab_dict = self.ab_providers.get(key, {})
            return ab_dict.get(line)

    def get_provider_by_id(self, provider_id: str) -> Optional[Provider]:
        """Get provider by its unique ID."""
        return self.providers_by_id.get(provider_id)

    def list_capabilities(self) -> List[Tuple[str, str]]:
        """
        List all registered capabilities.

        Returns:
            List of (modality, capability) tuples
        """
        # Combine primary and A/B capabilities
        all_keys = set(self.providers.keys()) | set(self.ab_providers.keys())
        return sorted(all_keys)

    def list_providers(self) -> List[Provider]:
        """
        List all registered provider instances.

        Returns:
            List of unique Provider instances
        """
        return list(self.providers_by_id.values())

    def has_capability(self, modality: str, capability: str) -> bool:
        """Check if a capability is registered."""
        key = (modality, capability)
        return key in self.providers or key in self.ab_providers

    def has_ab_providers(self, modality: str, capability: str) -> bool:
        """Check if a capability has A/B providers registered."""
        key = (modality, capability)
        return key in self.ab_providers and len(self.ab_providers[key]) >= 2

    def get_ab_providers(
        self,
        modality: str,
        capability: str
    ) -> Dict[str, Provider]:
        """
        Get all A/B providers for a capability.

        Returns:
            Dict mapping line ("a", "b", etc.) to Provider
        """
        key = (modality, capability)
        return self.ab_providers.get(key, {})

    def unregister(self, modality: str, capability: str, line: Optional[str] = None):
        """
        Unregister a provider.

        Args:
            modality: Input modality
            capability: Capability name
            line: Optional A/B line to unregister
                  If None, unregisters primary provider
        """
        key = (modality, capability)

        if line is None:
            # Unregister primary
            if key in self.providers:
                provider = self.providers[key]
                del self.providers[key]
                # Also remove from ID mapping if no other registrations
                if provider.provider_id in self.providers_by_id:
                    if all(
                        p.provider_id != provider.provider_id
                        for p in self.providers.values()
                    ):
                        del self.providers_by_id[provider.provider_id]
        else:
            # Unregister A/B line
            if key in self.ab_providers and line in self.ab_providers[key]:
                provider = self.ab_providers[key][line]
                del self.ab_providers[key][line]
                # Clean up empty A/B dict
                if not self.ab_providers[key]:
                    del self.ab_providers[key]
                # Remove from ID mapping if needed
                if provider.provider_id in self.providers_by_id:
                    if all(
                        p.provider_id != provider.provider_id
                        for p in self.providers.values()
                    ) and all(
                        p.provider_id != provider.provider_id
                        for ab_dict in self.ab_providers.values()
                        for p in ab_dict.values()
                    ):
                        del self.providers_by_id[provider.provider_id]

    def get_registry_summary(self) -> dict:
        """
        Get summary of registry state for debugging/monitoring.

        Returns:
            Dict with registry statistics and provider info
        """
        return {
            "total_capabilities": len(self.list_capabilities()),
            "total_providers": len(self.providers_by_id),
            "capabilities": [
                {
                    "modality": mod,
                    "capability": cap,
                    "has_primary": (mod, cap) in self.providers,
                    "has_ab": (mod, cap) in self.ab_providers,
                    "ab_lines": list(self.ab_providers.get((mod, cap), {}).keys()),
                }
                for mod, cap in self.list_capabilities()
            ],
            "providers": [
                {
                    "provider_id": p.provider_id,
                    "version": p.version,
                    "capabilities": p.get_capabilities(),
                    "is_native_m4": p.is_native_m4(),
                }
                for p in self.list_providers()
            ],
        }

    def __repr__(self) -> str:
        """Human-readable representation."""
        return (
            f"CapabilityRegistry("
            f"capabilities={len(self.list_capabilities())}, "
            f"providers={len(self.providers_by_id)})"
        )
