"""Structured skill registry for the conversation engine.

Skills are named actions the LLM can invoke via tags in its output.
Each skill has a schema (for prompt construction) and an async handler.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    """A single executable skill."""

    name: str
    description: str
    parameters: Dict[str, str] = field(default_factory=dict)
    handler: Optional[Callable[..., Coroutine]] = None


class SkillRegistry:
    """Registry of available skills.

    Provides prompt formatting so the LLM knows what skills exist,
    tag parsing to extract skill invocations from LLM output, and
    dispatch to execute the matching handler.
    """

    # Matches [skill_name] or [skill_name:arg] in LLM output
    TAG_RE = re.compile(r"\[(\w+)(?::([^\]]*))?\]")

    def __init__(self):
        self._skills: Dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill
        logger.info(f"Skill registered: {skill.name}")

    def get(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    @property
    def names(self) -> List[str]:
        return list(self._skills.keys())

    def list_for_prompt(self) -> str:
        """Format all skills as a block for the LLM system prompt."""
        if not self._skills:
            return ""

        lines = ["Available skills (use [skill_name] or [skill_name:argument] tags):"]
        for skill in self._skills.values():
            params = ", ".join(f"{k}: {v}" for k, v in skill.parameters.items())
            tag_example = f"[{skill.name}:...]" if skill.parameters else f"[{skill.name}]"
            lines.append(f"  - {tag_example} — {skill.description}")
            if params:
                lines.append(f"    Parameters: {params}")
        return "\n".join(lines)

    def parse_tags(self, text: str) -> Tuple[str, List[Tuple[str, Optional[str]]]]:
        """Extract skill tags from LLM output.

        Returns:
            (clean_text, [(skill_name, arg_or_None), ...])
            Tags are returned in the order they appear.
            clean_text has the tags stripped out.
        """
        invocations: List[Tuple[str, Optional[str]]] = []
        for match in self.TAG_RE.finditer(text):
            name = match.group(1).lower()
            arg = match.group(2) if match.group(2) else None
            if name in self._skills:
                invocations.append((name, arg))

        clean = self.TAG_RE.sub("", text).strip()
        # Collapse multiple whitespace
        clean = re.sub(r"  +", " ", clean)
        return clean, invocations

    async def execute(self, skill_name: str, **kwargs) -> Any:
        """Look up a skill by name and call its handler."""
        skill = self._skills.get(skill_name)
        if skill is None:
            logger.warning(f"Unknown skill: {skill_name}")
            return None
        if skill.handler is None:
            logger.warning(f"Skill '{skill_name}' has no handler")
            return None
        try:
            return await skill.handler(**kwargs)
        except Exception as e:
            logger.error(f"Skill '{skill_name}' execution failed: {e}", exc_info=True)
            return None
