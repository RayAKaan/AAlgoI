import os
from typing import Optional

import requests


OLLAMA_DEFAULT_URL = "http://localhost:11434"


class OllamaClient:
    """
    Minimal HTTP client for Ollama API.
    Safe defaults: low temperature, top-p, short timeout.
    """

    def __init__(self, base_url: str = None, model: str = "llama2"):
        self.base_url = base_url or os.environ.get(
            "OLLAMA_URL", OLLAMA_DEFAULT_URL
        )
        self.model = model
        self._test_connection()

    def _test_connection(self):
        try:
            resp = requests.get(
                f"{self.base_url}/api/tags",
                timeout=2
            )
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(
                f"Ollama not found at {self.base_url}. "
                f"Start with: ollama serve"
            ) from e

    def generate(
        self,
        prompt: str,
        system: str = None,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 2048,
        timeout: int = 30,
    ) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            },
            "stream": False,
        }

        if system:
            payload["system"] = system

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}") from e

    def is_model_available(self, model: str) -> bool:
        try:
            resp = requests.get(
                f"{self.base_url}/api/tags",
                timeout=2,
            )
            resp.raise_for_status()
            models = resp.json().get("models", [])
            return any(m.get("name") == model for m in models)
        except Exception:
            return False
