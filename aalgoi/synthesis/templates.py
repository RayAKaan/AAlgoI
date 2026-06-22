from __future__ import annotations

from typing import Any


class TemplateManager:
    def __init__(self) -> None:
        self._templates: dict[str, str] = {}

    def register(self, name: str, template: str) -> None:
        self._templates[name] = template

    def render(self, name: str, **kwargs: Any) -> str:
        if name not in self._templates:
            raise KeyError(f"Unknown template: {name}")
        return self._templates[name].format(**kwargs)

    def list_templates(self) -> list[str]:
        return list(self._templates.keys())
