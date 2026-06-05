from aalgoi.algorithms.base import Algorithm
from aalgoi.algorithms.primitives import (
    Primitive, PRIMITIVES, IteratePrimitive, MapPrimitive, FilterPrimitive,
    ReducePrimitive, ScanPrimitive, PartitionPrimitive, BinarySearchPrimitive,
    LinearSearchPrimitive, GreedyPrimitive, DynamicProgrammingPrimitive,
    GradientDescentPrimitive, QuickSortPrimitive, MergeSortPrimitive,
    HeapSortPrimitive, BFSPrimitive, DFSPrimitive, InterpolationSearchPrimitive,
    TwoPointerPrimitive, SlidingWindowPrimitive, TopologicalSortPrimitive,
    UnionFindPrimitive, BacktrackingPrimitive, RandomSearchPrimitive,
    LongestCommonSubsequencePrimitive, RabinKarpPrimitive,
)

try:
    from aalgoi.algorithms.sorting import QuickSort, InsertionSort, MergeSort, TimSort, RadixSort, HeapSort
except ImportError:
    QuickSort = InsertionSort = MergeSort = TimSort = RadixSort = HeapSort = None

try:
    from aalgoi.algorithms.image_processing import GaussianBlur, MedianFilter, BilateralFilter, SobelEdgeDetection, CLAHE, CannyEdgeDetection, LaplacianEdgeDetection, NLMDenoising, MorphologyOperation
except ImportError:
    GaussianBlur = MedianFilter = BilateralFilter = SobelEdgeDetection = CLAHE = CannyEdgeDetection = LaplacianEdgeDetection = NLMDenoising = MorphologyOperation = None

try:
    from aalgoi.algorithms.ml import KMeansClustering, DBSCANClustering, RandomForestClassifier, LinearRegression
except ImportError:
    KMeansClustering = DBSCANClustering = RandomForestClassifier = LinearRegression = None

try:
    from aalgoi.algorithms.pathfinding import Dijkstra, AStar, BFSPathfinder
except ImportError:
    Dijkstra = AStar = BFSPathfinder = None

try:
    from aalgoi.algorithms.optimization import GreedyKnapsack, SimulatedAnnealing
except ImportError:
    GreedyKnapsack = SimulatedAnnealing = None

try:
    from aalgoi.algorithms.safety import IdentityAlgorithm, SafeSort, SafePath, SafeKnapsack
except ImportError:
    IdentityAlgorithm = SafeSort = SafePath = SafeKnapsack = None

__all__ = [
    "Algorithm", "Primitive", "PRIMITIVES",
    "IteratePrimitive", "MapPrimitive", "FilterPrimitive",
    "ReducePrimitive", "ScanPrimitive", "PartitionPrimitive",
    "BinarySearchPrimitive", "LinearSearchPrimitive", "GreedyPrimitive",
    "DynamicProgrammingPrimitive", "GradientDescentPrimitive",
    "QuickSortPrimitive", "MergeSortPrimitive", "HeapSortPrimitive",
    "BFSPrimitive", "DFSPrimitive", "InterpolationSearchPrimitive",
    "TwoPointerPrimitive", "SlidingWindowPrimitive", "TopologicalSortPrimitive",
    "UnionFindPrimitive", "BacktrackingPrimitive", "RandomSearchPrimitive",
    "LongestCommonSubsequencePrimitive", "RabinKarpPrimitive",
    "QuickSort", "InsertionSort", "MergeSort", "TimSort", "RadixSort", "HeapSort",
    "GaussianBlur", "MedianFilter", "BilateralFilter", "SobelEdgeDetection", "CLAHE",
    "CannyEdgeDetection", "LaplacianEdgeDetection", "NLMDenoising", "MorphologyOperation",
    "KMeansClustering", "DBSCANClustering", "RandomForestClassifier", "LinearRegression",
    "Dijkstra", "AStar", "BFSPathfinder",
    "GreedyKnapsack", "SimulatedAnnealing",
    "IdentityAlgorithm", "SafeSort", "SafePath", "SafeKnapsack",
]
