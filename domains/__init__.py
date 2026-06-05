"""
Domain-specific adapters for the AAlgoI universal solver.

Each sub-module in this package provides domain-specific:
  - Problem type definitions and constraints
  - Input/output schema validation
  - Custom correctness oracles
  - Domain-specific configuration defaults

Available domains:
  - sorting
  - pathfinding
  - optimization
  - ml (classification, regression, clustering)
  - nlp
  - image_processing
  - scheduling
  - routing

Usage:
    from domains.sorting import SortingDomain
    domain = SortingDomain()
    spec = domain.create_problem_spec(data)
    result = domain.validate(output)
"""

from typing import Any, Dict, List

DOMAIN_REGISTRY: dict[str, Any] = {}

def register_domain(name: str, domain_module):
    DOMAIN_REGISTRY[name] = domain_module

def get_domain(name: str):
    return DOMAIN_REGISTRY.get(name)

def list_domains() -> list[str]:
    return list(DOMAIN_REGISTRY.keys())
