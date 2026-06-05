from typing import Dict, List, Set
from aalgoi.core.problem_spec import ProblemType
import logging

logger = logging.getLogger(__name__)

DOMAIN_ALGORITHMS: Dict[ProblemType, Set[str]] = {}

def _build_domain_map(registry: Dict) -> None:
    DOMAIN_ALGORITHMS.clear()
    for name, algo in registry.items():
        types = getattr(algo, 'problem_types', None) or getattr(algo, 'tags', [])
        if not types:
            continue
        for pt_str in types:
            pt_str_upper = pt_str.upper()
            try:
                pt = ProblemType(pt_str_upper) if pt_str_upper in [t.value.upper() for t in ProblemType] else ProblemType(pt_str_upper)
            except (ValueError, AttributeError):
                continue
            DOMAIN_ALGORITHMS.setdefault(pt, set()).add(name)

    compact = {k.name: len(v) for k, v in DOMAIN_ALGORITHMS.items()}
    logger.info("Domain router built: %s", compact)


def get_algorithms_for_domain(problem_type: ProblemType) -> List[str]:
    return list(DOMAIN_ALGORITHMS.get(problem_type, set()))


def get_broad_domain(pt: ProblemType) -> ProblemType:
    broad_map = {
        ProblemType.CLASSIFICATION: ProblemType.ML,
        ProblemType.REGRESSION: ProblemType.ML,
        ProblemType.CLUSTERING: ProblemType.ML,
        ProblemType.SEARCH: ProblemType.SORTING,
        ProblemType.ROUTING: ProblemType.PATHFINDING,
        ProblemType.SCHEDULING: ProblemType.OPTIMIZATION,
        ProblemType.TRANSFORMATION: ProblemType.SORTING,
        ProblemType.GENERATION: ProblemType.NLP,
        ProblemType.DECISION: ProblemType.OPTIMIZATION,
        ProblemType.COMPUTER_VISION: ProblemType.IMAGE_PROCESSING,
    }
    return broad_map.get(pt, pt)


def build_candidate_mask(algo_names: List[str], problem_type: ProblemType) -> List[int]:
    domain_algos = get_algorithms_for_domain(problem_type)
    if not domain_algos:
        broad = get_broad_domain(problem_type)
        domain_algos = get_algorithms_for_domain(broad)
    return [i for i, n in enumerate(algo_names) if n in domain_algos]
