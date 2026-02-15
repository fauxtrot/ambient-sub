"""Florence2 vision-language model wrapper.

Lazy-loading wrapper for Microsoft's Florence-2 model.
Used by the mirror skill to caption viewport screenshots.
"""

import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class Florence2Model:
    """Florence-2 wrapper for image captioning and description.

    Follows the same lazy-init pattern as TTSEngine — model loads
    on first use, not at startup.  All inference methods are blocking
    and should be called via ``run_in_executor()``.
    """

    def __init__(
        self,
        model_name: str = "microsoft/Florence-2-base",
        device: str = "cuda",
    ):
        self._model_name = model_name
        self._device = device
        self._model = None
        self._processor = None
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy-load model + processor from HuggingFace."""
        if self._initialized:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor

            logger.info(f"Loading Florence-2 model '{self._model_name}' on {self._device}...")

            self._processor = AutoProcessor.from_pretrained(
                self._model_name, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name, trust_remote_code=True
            )

            # Move to device
            if self._device != "cpu" and torch.cuda.is_available():
                self._model = self._model.to(self._device)
            else:
                self._device = "cpu"

            self._model.eval()
            self._initialized = True
            logger.info(f"Florence-2 loaded on {self._device}")

        except ImportError as e:
            logger.warning(f"Florence-2 dependencies not available: {e}")
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to load Florence-2: {e}", exc_info=True)
            self._initialized = True

    def _run_task(self, image_bytes: bytes, task: str, prompt: str = "") -> str:
        """Run a Florence-2 task on JPEG bytes. Blocking."""
        import torch
        from PIL import Image

        self._ensure_initialized()
        if self._model is None or self._processor is None:
            return ""

        # Decode JPEG → PIL
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Build input
        text_input = task if not prompt else f"{task} {prompt}"
        inputs = self._processor(text=text_input, images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=3,
            )

        # Decode output
        result = self._processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        # Post-process (Florence-2 specific)
        parsed = self._processor.post_process_generation(
            result, task=task, image_size=image.size
        )

        # Extract text from parsed result
        if isinstance(parsed, dict):
            # Florence-2 returns {task: result} — grab the value
            text = parsed.get(task, "")
            if isinstance(text, str):
                return text.strip()
            # Some tasks return structured data
            return str(text)

        return str(parsed).strip()

    def caption(self, image_bytes: bytes) -> str:
        """JPEG bytes → short caption. Blocking — run in thread pool."""
        return self._run_task(image_bytes, "<CAPTION>")

    def describe(self, image_bytes: bytes) -> str:
        """JPEG bytes → detailed description. Blocking — run in thread pool."""
        return self._run_task(image_bytes, "<MORE_DETAILED_CAPTION>")

    @property
    def available(self) -> bool:
        """Check if Florence-2 is available."""
        self._ensure_initialized()
        return self._model is not None
