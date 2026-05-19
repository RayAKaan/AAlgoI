
import json
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple


SELECTION_PROMPT = """You are an algorithm selection expert. Given this context:

Data: size={size}, type={data_type}, sorted={is_sorted}, nearly_sorted={is_nearly_sorted}
System: cpu_free={cpu_free:.0%}, mem_free={mem_free:.0%}, cpu_count={cpu_count}
Constraints: time_budget={time_budget}ms, priority={priority}

Available algorithms:
{algorithms}

Select the optimal algorithm. Return ONLY valid JSON:
{{"algorithm": "name", "reason": "brief reason"}}
"""

EXPLANATION_PROMPT = """Explain why the algorithm "{chosen}" was selected for this context:

Data: size={size}, type={data_type}, sorted={is_sorted}, nearly_sorted={is_nearly_sorted}
System: cpu_free={cpu_free:.0%}, mem_free={mem_free:.0%}
Priority: {priority}

Result: time={time_ms:.1f}ms, quality={quality:.2f}, success={success}

Return ONLY valid JSON:
{{"explanation": "your explanation", "confidence": 0.0-1.0, "suggestion": "optional improvement"}}
"""

PARAM_PROMPT = """Suggest optimal parameters for algorithm "{algo_name}" given:

Data: size={size}, type={data_type}
System: cpu_free={cpu_free:.0%}, mem_free={mem_free:.0%}
Priority: {priority}

Known parameters: {known_params}

Return ONLY valid JSON with parameter names and values.
"""


class LLMAdapter:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.backend = self.config.get("backend", "ollama")
        self.model = self.config.get("model", "phi3:mini")
        self.ollama_url = self.config.get("ollama_url", "http://localhost:11434")
        self.llamacpp_url = self.config.get("llamacpp_url", "http://localhost:8080")
        self.max_tokens = self.config.get("max_tokens", 256)
        self.temperature = self.config.get("temperature", 0.1)
        self.enabled = self.config.get("enabled", False)
        self._available = None

    def check_available(self) -> bool:
        if self._available is not None:
            return self._available

        if not self.enabled:
            self._available = False
            return False

        try:
            import requests
            if self.backend == "ollama":
                r = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
                self._available = r.status_code == 200
            else:
                r = requests.get(f"{self.llamacpp_url}/health", timeout=2)
                self._available = r.status_code == 200
        except Exception:
            self._available = False

        return self._available

    def analyze_context(self, context: Dict) -> Dict[str, Any]:
        if not self.enabled:
            return {"algorithm": None, "reason": "LLM disabled"}

        if not self.check_available():
            return {"algorithm": None, "reason": "LLM not available"}

        fp = context.get("features", {})
        dp = context.get("data_profile", {})
        env = context.get("environment", {})
        cons = context.get("constraints", {})

        cpu = fp.get("cpu_free", 0.5)
        mem = fp.get("mem_free_ratio", 0.5)
        cpu_count = fp.get("cpu_count", 1)

        task_type = context.get("task_type", "sorting")
        algo_descriptions = self._get_algo_descriptions(task_type)

        prompt = SELECTION_PROMPT.format(
            size=dp.get("size", "unknown"),
            data_type=dp.get("type", "unknown"),
            is_sorted=dp.get("patterns", {}).get("is_sorted", False),
            is_nearly_sorted=dp.get("patterns", {}).get("is_nearly_sorted", False),
            cpu_free=cpu,
            mem_free=mem,
            cpu_count=cpu_count,
            time_budget=cons.get("time_budget_ms", 500),
            priority=cons.get("priority", "balanced"),
            algorithms=algo_descriptions
        )

        response = self._query(prompt)
        return self._parse_selection(response)

    def explain_decision(self, context: Dict, chosen: str, result_metrics: Dict) -> str:
        if not self.enabled or not self.check_available():
            return "LLM explanation unavailable"

        fp = context.get("features", {})
        dp = context.get("data_profile", {})

        prompt = EXPLANATION_PROMPT.format(
            chosen=chosen,
            size=dp.get("size", "unknown"),
            data_type=dp.get("type", "unknown"),
            is_sorted=dp.get("patterns", {}).get("is_sorted", False),
            is_nearly_sorted=dp.get("patterns", {}).get("is_nearly_sorted", False),
            cpu_free=fp.get("cpu_free", 0.5),
            mem_free=fp.get("mem_free_ratio", 0.5),
            priority=context.get("constraints", {}).get("priority", "balanced"),
            time_ms=result_metrics.get("wall_time_ms", 0),
            quality=result_metrics.get("quality_score", 0),
            success=result_metrics.get("success", False)
        )

        response = self._query(prompt)
        try:
            return json.loads(response).get("explanation", response)
        except (json.JSONDecodeError, ValueError):
            return response

    def suggest_parameters(self, context: Dict, algo_name: str) -> Dict:
        if not self.enabled or not self.check_available():
            return {}

        fp = context.get("features", {})
        dp = context.get("data_profile", {})

        known = self._get_known_params(algo_name)

        prompt = PARAM_PROMPT.format(
            algo_name=algo_name,
            size=dp.get("size", "unknown"),
            data_type=dp.get("type", "unknown"),
            cpu_free=fp.get("cpu_free", 0.5),
            mem_free=fp.get("mem_free_ratio", 0.5),
            priority=context.get("constraints", {}).get("priority", "balanced"),
            known_params=json.dumps(known)
        )

        response = self._query(prompt)
        return self._try_parse_json(response)

    def _query(self, prompt: str) -> str:
        try:
            import requests
            if self.backend == "ollama":
                return self._query_ollama(prompt)
            else:
                return self._query_llamacpp(prompt)
        except (ImportError, Exception) as e:
            return json.dumps({"error": str(e)})

    def _query_ollama(self, prompt: str) -> str:
        import requests
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": self.max_tokens,
                "temperature": self.temperature
            }
        }
        r = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=30)
        if r.status_code == 200:
            return r.json().get("response", "")
        return json.dumps({"error": f"Ollama error: {r.status_code}"})

    def _query_llamacpp(self, prompt: str) -> str:
        import requests
        payload = {
            "prompt": prompt,
            "n_predict": self.max_tokens,
            "temperature": self.temperature
        }
        r = requests.post(f"{self.llamacpp_url}/completion", json=payload, timeout=30)
        if r.status_code == 200:
            return r.json().get("content", "")
        return json.dumps({"error": f"llama.cpp error: {r.status_code}"})

    def _parse_selection(self, response: str) -> Dict:
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict) and "algorithm" in parsed:
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

        for line in response.strip().split('\n'):
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                try:
                    parsed = json.loads(line)
                    if isinstance(parsed, dict) and "algorithm" in parsed:
                        return parsed
                except json.JSONDecodeError:
                    continue

        return {"algorithm": None, "reason": response[:200]}

    def _try_parse_json(self, text: str) -> Dict:
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            for line in text.strip().split('\n'):
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError:
                        continue
        return {"raw": text[:200]}

    def _get_algo_descriptions(self, task_type: str) -> str:
        descriptions = {
            "sorting": [
                "quicksort - O(n log n) avg, in-place, fast random data, bad nearly-sorted",
                "insertion_sort - O(n²) worst, O(n) best, great tiny/nearly-sorted data",
                "merge_sort - O(n log n) guaranteed, stable, uses O(n) memory",
                "timsort - O(n log n), hybrid, great real-world/nearly-sorted, Python built-in",
                "radix_sort - O(nk) linear for integers, needs small digit range",
                "heap_sort - O(n log n) in-place, O(1) space, good memory-constrained"
            ],
            "image_processing": [
                "gaussian_blur - O(n), fast noise reduction, blurs edges",
                "median_filter - O(n), great salt-pepper noise, preserves edges",
                "bilateral_filter - O(n), edge-preserving denoise, slower",
                "sobel_edge - O(n), fast edge detection",
                "clahe - O(n), adaptive contrast enhancement, good low-contrast images"
            ],
            "ml": [
                "kmeans - O(n*k*i), simple clustering, needs k specified",
                "dbscan - O(n log n), density clustering, detects outliers",
                "random_forest - ensemble classifier, high accuracy, slower",
                "linear_regression - O(n*d²), fast regression, interpretable"
            ]
        }
        algos = descriptions.get(task_type, descriptions["sorting"])
        return "\n".join(f"  - {a}" for a in algos)

    def _get_known_params(self, algo_name: str) -> Dict:
        known = {
            "quicksort": {"in_place": False},
            "timsort": {},
            "merge_sort": {},
            "insertion_sort": {},
            "heap_sort": {},
            "radix_sort": {},
            "gaussian_blur": {"sigma": 1.0},
            "median_filter": {"filter_size": 3},
            "bilateral_filter": {"sigma_color": 0.1, "sigma_spatial": 1.0},
            "sobel_edge": {},
            "clahe": {"clip_limit": 0.03},
            "kmeans": {"n_clusters": 3, "max_iter": 100},
            "dbscan": {"eps": 0.5, "min_samples": 5},
            "random_forest": {"n_estimators": 100, "max_depth": 10},
            "linear_regression": {"fit_intercept": True}
        }
        return known.get(algo_name, {})

    def get_stats(self) -> Dict:
        return {
            "enabled": self.enabled,
            "available": self._available,
            "backend": self.backend,
            "model": self.model
        }
