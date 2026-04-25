from __future__ import annotations

from typing import Any, Callable, Dict, Generic, TypeVar


T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, name: str):
        self.name = name
        self._items: Dict[str, Callable[..., T]] = {}

    def register(self, key: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def decorator(factory: Callable[..., T]) -> Callable[..., T]:
            if key in self._items:
                raise KeyError(f"{self.name} registry already contains {key!r}.")
            self._items[key] = factory
            return factory

        return decorator

    def get(self, key: str) -> Callable[..., T]:
        try:
            return self._items[key]
        except KeyError as exc:
            available = ", ".join(sorted(self._items)) or "<empty>"
            raise KeyError(f"Unknown {self.name} component {key!r}. Available: {available}") from exc

    def build(self, config: Dict[str, Any] | None = None, **kwargs: Any) -> T:
        payload = dict(config or {})
        name = str(payload.pop("name", "identity"))
        payload.update(kwargs)
        return self.get(name)(**payload)

    def keys(self) -> list[str]:
        return sorted(self._items)

