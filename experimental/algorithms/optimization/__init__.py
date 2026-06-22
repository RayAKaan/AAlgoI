from .optimization_algos import (
    AntColony,
    GeneticAlgorithm,
    GreedyKnapsack,
    HillClimbing,
    ParticleSwarm,
    SimulatedAnnealing,
)

ParticleSwarmOptimization = ParticleSwarm
AntColonyOptimization     = AntColony

__all__ = [
    'GreedyKnapsack', 'SimulatedAnnealing', 'GeneticAlgorithm',
    'HillClimbing', 'ParticleSwarm', 'AntColony',
    'ParticleSwarmOptimization', 'AntColonyOptimization',
]
