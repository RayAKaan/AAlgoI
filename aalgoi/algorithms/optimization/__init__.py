from .optimization_algos import (
    GreedyKnapsack, SimulatedAnnealing, GeneticAlgorithm,
    HillClimbing, ParticleSwarm, AntColony
)

ParticleSwarmOptimization = ParticleSwarm
AntColonyOptimization     = AntColony

__all__ = [
    'GreedyKnapsack', 'SimulatedAnnealing', 'GeneticAlgorithm',
    'HillClimbing', 'ParticleSwarm', 'AntColony',
    'ParticleSwarmOptimization', 'AntColonyOptimization',
]
