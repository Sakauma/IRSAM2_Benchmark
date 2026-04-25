from __future__ import annotations

from ..registry import Registry
from .base import IdentityPrior, PriorModule
from .physics import PriorFusion


PriorFactory: Registry[PriorModule] = Registry("prior")

PriorFactory.register("identity")(IdentityPrior)
PriorFactory.register("prior_fusion")(PriorFusion)

__all__ = ["IdentityPrior", "PriorFactory", "PriorFusion", "PriorModule"]

