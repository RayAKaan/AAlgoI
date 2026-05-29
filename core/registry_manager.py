"""
Dynamic Algorithm Registry
Plugin system for runtime algorithm registration.
"""

import importlib.util
import os
import logging
from typing import Dict, List, Optional

from algorithms.base import Algorithm

logger = logging.getLogger(__name__)


class DynamicRegistry:
    """
    Manages dynamic algorithm registration and discovery.
    Users can register custom algorithms that behave like native ones.
    """

    def __init__(self, base_registry: Optional[Dict[str, Algorithm]] = None):
        self.algorithms: Dict[str, Algorithm] = base_registry or {}

    def register(self, algorithm: Algorithm) -> bool:
        """Register a new algorithm for the solver to use."""
        if not isinstance(algorithm, Algorithm):
            raise TypeError("Must be instance of Algorithm base class")

        self.algorithms[algorithm.name] = algorithm
        logger.info("Registered algorithm: %s", algorithm.name)
        return True

    def register_from_code(self, name: str, code: str) -> bool:
        """
        Register an algorithm from source code string.
        Dynamically loads the code and extracts Algorithm subclasses.
        """
        import importlib.util as _util
        import sys as _sys
        import tempfile as _tf

        with _tf.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False, encoding='utf-8'
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            spec = _util.spec_from_file_location(f"dynamic_{name}", tmp_path)
            module = _util.module_from_spec(spec)
            _sys.modules[f"dynamic_{name}"] = module
            spec.loader.exec_module(module)

            registered = False
            for attr_name in dir(module):
                obj = getattr(module, attr_name)
                if isinstance(obj, type) and issubclass(obj, Algorithm) and obj != Algorithm:
                    instance = obj()
                    self.register(instance)
                    registered = True

            return registered
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def register_from_file(self, file_path: str, class_name: Optional[str] = None) -> List[str]:
        """
        Load an algorithm from a .py file.
        Useful for sharing algorithms without PyPI packages.
        Returns list of registered algorithm names.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Algorithm file not found: {file_path}")

        spec = importlib.util.spec_from_file_location("custom_module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        registered = []
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and issubclass(obj, Algorithm) and obj != Algorithm:
                if class_name and name != class_name:
                    continue
                instance = obj()
                self.register(instance)
                registered.append(instance.name)

        if not registered:
            raise ValueError(f"No Algorithm subclass found in {file_path}")

        return registered

    def list_algorithms(self) -> List[str]:
        """Return all registered algorithm names."""
        return list(self.algorithms.keys())

    def get_algorithm(self, name: str) -> Optional[Algorithm]:
        """Retrieve algorithm by name."""
        return self.algorithms.get(name)

    def unregister(self, name: str) -> bool:
        """Remove an algorithm from the registry."""
        if name in self.algorithms:
            del self.algorithms[name]
            logger.info("Unregistered algorithm: %s", name)
            return True
        return False
