"""
Personaplex LLM client for conversational interactions.

Model: nvidia/personaplex-7b-v1 via HuggingFace
Framework: Moshi (audio-to-audio capable)
Mode: Text input (audio mode future)
"""

import logging
from typing import Optional

import torch

from .base import LLMClient

logger = logging.getLogger(__name__)


class PersonaplexClient(LLMClient):
    """
    Personaplex conversational LLM client.

    Handles fast, natural dialogue and conversation flow.
    """

    def __init__(
        self,
        model_name: str = "nvidia/personaplex-7b-v1",
        device: Optional[str] = None,
        max_length: int = 512,
        temperature: float = 0.7
    ):
        """
        Initialize Personaplex client.

        Args:
            model_name: HuggingFace model identifier
            device: Device to use ("cuda", "cpu", or None for auto)
            max_length: Maximum generation length
            temperature: Sampling temperature
        """
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self.tokenizer = None
        self._initialized = False

        logger.info(
            f"PersonaplexClient initialized: model={model_name}, device={self.device}"
        )

    async def _load_model(self):
        """Lazy load model on first use."""
        if self._initialized:
            return

        try:
            logger.info(f"Loading Personaplex model: {self.model_name}")

            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )

            if self.device == "cpu":
                self.model = self.model.to(self.device)

            self._initialized = True
            logger.info("Personaplex model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Personaplex model: {e}")
            raise

    async def complete(self, prompt: str, **kwargs) -> str:
        """
        Generate conversational response.

        Args:
            prompt: Text prompt
            **kwargs: Optional overrides (max_length, temperature)

        Returns:
            Generated response text
        """
        # Ensure model is loaded
        await self._load_model()

        max_length = kwargs.get('max_length', self.max_length)
        temperature = kwargs.get('temperature', self.temperature)

        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode output
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove input prompt from response if present
            if response.startswith(prompt):
                response = response[len(prompt):].strip()

            return response

        except Exception as e:
            logger.error(f"Personaplex completion error: {e}")
            return f"[Error: {e}]"

    async def is_available(self) -> bool:
        """Check if Personaplex is available."""
        try:
            await self._load_model()
            return self._initialized
        except Exception:
            return False

    async def cleanup(self) -> None:
        """Cleanup model resources."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._initialized = False
        logger.info("Personaplex client cleaned up")
