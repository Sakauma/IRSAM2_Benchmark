from __future__ import annotations

from ..registry import Registry
from .base import IdentityPrompt, PromptGenerator
from .heuristic_physics import HeuristicPhysicsPrompt


PromptFactory: Registry[PromptGenerator] = Registry("prompt")

PromptFactory.register("identity")(IdentityPrompt)
PromptFactory.register("heuristic_physics")(HeuristicPhysicsPrompt)

__all__ = ["HeuristicPhysicsPrompt", "IdentityPrompt", "PromptFactory", "PromptGenerator"]

