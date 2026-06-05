from aalgoi.algorithms.base import Algorithm
from aalgoi.algorithms.primitives import (
    PRIMITIVES,
    BacktrackingPrimitive,
    BFSPrimitive,
    BinarySearchPrimitive,
    DFSPrimitive,
    DynamicProgrammingPrimitive,
    FilterPrimitive,
    GradientDescentPrimitive,
    GreedyPrimitive,
    HeapSortPrimitive,
    InterpolationSearchPrimitive,
    IteratePrimitive,
    LinearSearchPrimitive,
    LongestCommonSubsequencePrimitive,
    MapPrimitive,
    MergeSortPrimitive,
    PartitionPrimitive,
    Primitive,
    QuickSortPrimitive,
    RabinKarpPrimitive,
    RandomSearchPrimitive,
    ReducePrimitive,
    ScanPrimitive,
    SlidingWindowPrimitive,
    TopologicalSortPrimitive,
    TwoPointerPrimitive,
    UnionFindPrimitive,
)

try:
    from aalgoi.algorithms.sorting import HeapSort, InsertionSort, MergeSort, QuickSort, RadixSort, TimSort
except ImportError:
    QuickSort = InsertionSort = MergeSort = TimSort = RadixSort = HeapSort = None

try:
    from aalgoi.algorithms.image_processing import (
        CLAHE,
        BilateralFilter,
        CannyEdgeDetection,
        GaussianBlur,
        LaplacianEdgeDetection,
        MedianFilter,
        MorphologyOperation,
        NLMDenoising,
        SobelEdgeDetection,
    )
except ImportError:
    GaussianBlur = MedianFilter = BilateralFilter = SobelEdgeDetection = CLAHE = CannyEdgeDetection = LaplacianEdgeDetection = NLMDenoising = MorphologyOperation = None

try:
    from aalgoi.algorithms.ml import DBSCANClustering, KMeansClustering, LinearRegression, RandomForestClassifier
except ImportError:
    KMeansClustering = DBSCANClustering = RandomForestClassifier = LinearRegression = None

try:
    from aalgoi.algorithms.pathfinding import AStar, BFSPathfinder, Dijkstra
except ImportError:
    Dijkstra = AStar = BFSPathfinder = None

try:
    from aalgoi.algorithms.optimization import GreedyKnapsack, SimulatedAnnealing
except ImportError:
    GreedyKnapsack = SimulatedAnnealing = None

try:
    from aalgoi.algorithms.safety import IdentityAlgorithm, SafeKnapsack, SafePath, SafeSort
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
