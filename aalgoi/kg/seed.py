from aalgoi.algorithms.registry import AlgorithmRegistry, get_registry
from aalgoi.kg.graph import KnowledgeGraph


def seed_from_registry(kg: KnowledgeGraph, registry: AlgorithmRegistry | None = None) -> KnowledgeGraph:
    if registry is None:
        registry = get_registry()
    for name in registry.get_names():
        spec = registry.get_spec(name)
        if spec is not None:
            kg.add_algorithm(spec)
    return kg
