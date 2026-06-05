import torch
import torch.nn.functional as F
import time
import math
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

from aalgoi.core.mind.rl_mind import AlgorithmicMind
from aalgoi.core.mind.mind_state import build_data_profile
from aalgoi.core.mind.cognitive_actions import CognitiveAction, ActionParams, ActionResult
from aalgoi.core.mind.knowledge_graph import AlgorithmicKnowledgeGraph
from aalgoi.core.mind.model_config import MindConfig

_STATE_FEATURE_DIM = 64 + 8 + 128  # data_features + scalars + kg_neighborhood


def _build_full_state(data: Any) -> torch.Tensor:
    features = build_data_profile(data)
    full = torch.zeros(_STATE_FEATURE_DIM)
    full[:features.shape[0]] = features
    return full


def _pad_state_features(features: torch.Tensor) -> torch.Tensor:
    full = torch.zeros(_STATE_FEATURE_DIM)
    full[:features.shape[0]] = features
    return full


@dataclass
class IdealPath:
    problem_type: str
    domain: str
    optimization_goal: str
    hidden_structure: str
    ideal_actions: list[CognitiveAction]
    best_algorithm: str
    expected_iterations: int


def get_ideal_paths() -> list[IdealPath]:
    return [
        IdealPath("SORTING", "integers", "find", "total_order",
            [CognitiveAction.IDENTIFY_STRUCTURE, CognitiveAction.EXTRACT_CONSTRAINTS, CognitiveAction.ESTIMATE_COMPLEXITY, CognitiveAction.QUERY_ALGORITHMS, CognitiveAction.SELECT_ALGORITHM, CognitiveAction.TEST_EXAMPLES, CognitiveAction.ACCEPT_SOLUTION],
            "tim_sort", 7),
        IdealPath("SORTING_SMALL", "integers", "find", "total_order",
            [CognitiveAction.IDENTIFY_STRUCTURE, CognitiveAction.EXTRACT_CONSTRAINTS, CognitiveAction.QUERY_ALGORITHMS, CognitiveAction.SELECT_ALGORITHM, CognitiveAction.TEST_EXAMPLES, CognitiveAction.ACCEPT_SOLUTION],
            "insertion_sort", 6),
        IdealPath("SORTING_STABLE", "integers", "find", "total_order",
            [CognitiveAction.IDENTIFY_STRUCTURE, CognitiveAction.EXTRACT_CONSTRAINTS, CognitiveAction.QUERY_ALGORITHMS, CognitiveAction.SELECT_ALGORITHM, CognitiveAction.TEST_EXAMPLES, CognitiveAction.ACCEPT_SOLUTION],
            "merge_sort", 6),
        IdealPath("PATHFINDING_WEIGHTED", "graph", "minimize", "graph_connectivity",
            [CognitiveAction.IDENTIFY_STRUCTURE, CognitiveAction.FIND_INVARIANT, CognitiveAction.QUERY_ALGORITHMS, CognitiveAction.SELECT_ALGORITHM, CognitiveAction.TEST_EXAMPLES, CognitiveAction.ACCEPT_SOLUTION],
            "dijkstra", 6),
        IdealPath("PATHFINDING_UNWEIGHTED", "graph", "find", "graph_connectivity",
            [CognitiveAction.IDENTIFY_STRUCTURE, CognitiveAction.QUERY_ALGORITHMS, CognitiveAction.SELECT_ALGORITHM, CognitiveAction.TEST_EXAMPLES, CognitiveAction.ACCEPT_SOLUTION],
            "bfs", 5),
        IdealPath("PATHFINDING_NEGATIVE_WEIGHTS", "graph", "minimize", "optimal_substructure",
            [CognitiveAction.IDENTIFY_STRUCTURE, CognitiveAction.EXTRACT_CONSTRAINTS, CognitiveAction.QUERY_ALGORITHMS, CognitiveAction.SELECT_ALGORITHM, CognitiveAction.TEST_EXAMPLES, CognitiveAction.ACCEPT_SOLUTION],
            "bellman_ford", 6),
        IdealPath("SEARCH_SORTED", "array", "find", "monotonic_feasibility",
            [CognitiveAction.IDENTIFY_STRUCTURE, CognitiveAction.QUERY_PRINCIPLE, CognitiveAction.SELECT_ALGORITHM, CognitiveAction.TEST_EXAMPLES, CognitiveAction.ACCEPT_SOLUTION],
            "binary_search", 5),
        IdealPath("SEARCH_UNSORTED", "array", "find", "unknown",
            [CognitiveAction.IDENTIFY_STRUCTURE, CognitiveAction.QUERY_ALGORITHMS, CognitiveAction.SELECT_ALGORITHM, CognitiveAction.TEST_EXAMPLES, CognitiveAction.ACCEPT_SOLUTION],
            "linear_search", 5),
        IdealPath("DP_KNAPSACK", "numbers", "maximize", "optimal_substructure",
            [CognitiveAction.IDENTIFY_STRUCTURE, CognitiveAction.FIND_INVARIANT, CognitiveAction.QUERY_PRINCIPLE, CognitiveAction.QUERY_ALGORITHMS, CognitiveAction.SELECT_ALGORITHM, CognitiveAction.TEST_EXAMPLES, CognitiveAction.STRESS_TEST, CognitiveAction.ACCEPT_SOLUTION],
            "knapsack", 8),
        IdealPath("DP_LCS", "text", "find", "optimal_substructure",
            [CognitiveAction.IDENTIFY_STRUCTURE, CognitiveAction.QUERY_PRINCIPLE, CognitiveAction.QUERY_ALGORITHMS, CognitiveAction.SELECT_ALGORITHM, CognitiveAction.TEST_EXAMPLES, CognitiveAction.ACCEPT_SOLUTION],
            "lcs", 6),
        IdealPath("DP_LIS", "integers", "maximize", "optimal_substructure",
            [CognitiveAction.IDENTIFY_STRUCTURE, CognitiveAction.QUERY_PRINCIPLE, CognitiveAction.QUERY_ALGORITHMS, CognitiveAction.SELECT_ALGORITHM, CognitiveAction.TEST_EXAMPLES, CognitiveAction.ACCEPT_SOLUTION],
            "lis", 6),
        IdealPath("DP_COIN_CHANGE", "numbers", "minimize", "optimal_substructure",
            [CognitiveAction.IDENTIFY_STRUCTURE, CognitiveAction.FIND_INVARIANT, CognitiveAction.QUERY_PRINCIPLE, CognitiveAction.SELECT_ALGORITHM, CognitiveAction.TEST_EXAMPLES, CognitiveAction.ACCEPT_SOLUTION],
            "coin_change", 6),
        IdealPath("GREEDY_INTERVAL", "numbers", "maximize", "greedy_exchange",
            [CognitiveAction.IDENTIFY_STRUCTURE, CognitiveAction.QUERY_PRINCIPLE, CognitiveAction.QUERY_ALGORITHMS, CognitiveAction.SELECT_ALGORITHM, CognitiveAction.TEST_EXAMPLES, CognitiveAction.PROVE_CORRECTNESS, CognitiveAction.ACCEPT_SOLUTION],
            "activity_selection", 7),
        IdealPath("GREEDY_HUFFMAN", "numbers", "minimize", "greedy_exchange",
            [CognitiveAction.IDENTIFY_STRUCTURE, CognitiveAction.QUERY_PRINCIPLE, CognitiveAction.SELECT_ALGORITHM, CognitiveAction.TEST_EXAMPLES, CognitiveAction.ACCEPT_SOLUTION],
            "huffman", 5),
        IdealPath("GREEDY_MST", "graph", "minimize", "greedy_exchange",
            [CognitiveAction.IDENTIFY_STRUCTURE, CognitiveAction.QUERY_PRINCIPLE, CognitiveAction.QUERY_ALGORITHMS, CognitiveAction.SELECT_ALGORITHM, CognitiveAction.TEST_EXAMPLES, CognitiveAction.ACCEPT_SOLUTION],
            "minimum_spanning_tree", 6),
        IdealPath("BINARY_SEARCH_MIN_MAX", "numbers", "minimize", "monotonic_feasibility",
            [CognitiveAction.IDENTIFY_STRUCTURE, CognitiveAction.QUERY_PRINCIPLE, CognitiveAction.QUERY_ALGORITHMS, CognitiveAction.SELECT_ALGORITHM, CognitiveAction.TEST_EXAMPLES, CognitiveAction.ACCEPT_SOLUTION],
            "binary_search", 6),
        IdealPath("HASH_TWO_SUM", "integers", "find", "hashing_fingerprint",
            [CognitiveAction.IDENTIFY_STRUCTURE, CognitiveAction.QUERY_PRINCIPLE, CognitiveAction.SELECT_ALGORITHM, CognitiveAction.TEST_EXAMPLES, CognitiveAction.ACCEPT_SOLUTION],
            "hash_complement", 5),
        IdealPath("HASH_FREQUENCY", "integers", "find", "hashing_fingerprint",
            [CognitiveAction.IDENTIFY_STRUCTURE, CognitiveAction.QUERY_PRINCIPLE, CognitiveAction.SELECT_ALGORITHM, CognitiveAction.TEST_EXAMPLES, CognitiveAction.ACCEPT_SOLUTION],
            "hash_frequency", 5),
        IdealPath("GRAPH_TOPOLOGICAL", "graph", "find", "graph_connectivity",
            [CognitiveAction.IDENTIFY_STRUCTURE, CognitiveAction.QUERY_ALGORITHMS, CognitiveAction.SELECT_ALGORITHM, CognitiveAction.TEST_EXAMPLES, CognitiveAction.ACCEPT_SOLUTION],
            "topological_sort", 5),
        IdealPath("GRAPH_CONNECTED", "graph", "check", "graph_connectivity",
            [CognitiveAction.IDENTIFY_STRUCTURE, CognitiveAction.QUERY_ALGORITHMS, CognitiveAction.SELECT_ALGORITHM, CognitiveAction.TEST_EXAMPLES, CognitiveAction.ACCEPT_SOLUTION],
            "connected_components", 5),
        IdealPath("STRING_PATTERN", "text", "find", "hashing_fingerprint",
            [CognitiveAction.IDENTIFY_STRUCTURE, CognitiveAction.QUERY_ALGORITHMS, CognitiveAction.SELECT_ALGORITHM, CognitiveAction.TEST_EXAMPLES, CognitiveAction.ACCEPT_SOLUTION],
            "rabin_karp", 5),
        IdealPath("STRING_EDIT_DISTANCE", "text", "minimize", "optimal_substructure",
            [CognitiveAction.IDENTIFY_STRUCTURE, CognitiveAction.QUERY_PRINCIPLE, CognitiveAction.QUERY_ALGORITHMS, CognitiveAction.SELECT_ALGORITHM, CognitiveAction.TEST_EXAMPLES, CognitiveAction.ACCEPT_SOLUTION],
            "levenshtein", 6),
        IdealPath("ML_CLASSIFY", "feature_matrix", "find", "statistical_separation",
            [CognitiveAction.IDENTIFY_STRUCTURE, CognitiveAction.EXTRACT_CONSTRAINTS, CognitiveAction.QUERY_ALGORITHMS, CognitiveAction.SELECT_ALGORITHM, CognitiveAction.TEST_EXAMPLES, CognitiveAction.ACCEPT_SOLUTION],
            "random_forest_classifier", 6),
        IdealPath("ML_REGRESS", "feature_matrix", "minimize", "statistical_fitting",
            [CognitiveAction.IDENTIFY_STRUCTURE, CognitiveAction.EXTRACT_CONSTRAINTS, CognitiveAction.QUERY_ALGORITHMS, CognitiveAction.SELECT_ALGORITHM, CognitiveAction.TEST_EXAMPLES, CognitiveAction.ACCEPT_SOLUTION],
            "linear_regression", 6),
        IdealPath("CLUSTERING", "feature_matrix", "find", "statistical_separation",
            [CognitiveAction.IDENTIFY_STRUCTURE, CognitiveAction.EXTRACT_CONSTRAINTS, CognitiveAction.QUERY_ALGORITHMS, CognitiveAction.SELECT_ALGORITHM, CognitiveAction.TEST_EXAMPLES, CognitiveAction.ACCEPT_SOLUTION],
            "kmeans", 6),
        IdealPath("NLP_CLASSIFY", "text", "find", "sequential",
            [CognitiveAction.IDENTIFY_STRUCTURE, CognitiveAction.QUERY_ALGORITHMS, CognitiveAction.SELECT_ALGORITHM, CognitiveAction.TEST_EXAMPLES, CognitiveAction.ACCEPT_SOLUTION],
            "text_classifier", 5),
        IdealPath("TREE_BST", "tree", "find", "divide_conquer",
            [CognitiveAction.IDENTIFY_STRUCTURE, CognitiveAction.QUERY_ALGORITHMS, CognitiveAction.SELECT_ALGORITHM, CognitiveAction.TEST_EXAMPLES, CognitiveAction.ACCEPT_SOLUTION],
            "bst_operations", 5),
        IdealPath("TREE_TRIE", "tree", "find", "divide_conquer",
            [CognitiveAction.IDENTIFY_STRUCTURE, CognitiveAction.QUERY_ALGORITHMS, CognitiveAction.SELECT_ALGORITHM, CognitiveAction.TEST_EXAMPLES, CognitiveAction.ACCEPT_SOLUTION],
            "trie", 5),
        IdealPath("MATH_GCD", "numbers", "find", "divide_conquer",
            [CognitiveAction.IDENTIFY_STRUCTURE, CognitiveAction.QUERY_ALGORITHMS, CognitiveAction.SELECT_ALGORITHM, CognitiveAction.TEST_EXAMPLES, CognitiveAction.ACCEPT_SOLUTION],
            "gcd", 5),
        IdealPath("MATH_PRIMES", "numbers", "find", "amortized_invariant",
            [CognitiveAction.IDENTIFY_STRUCTURE, CognitiveAction.QUERY_ALGORITHMS, CognitiveAction.SELECT_ALGORITHM, CognitiveAction.TEST_EXAMPLES, CognitiveAction.ACCEPT_SOLUTION],
            "prime_sieve", 5),
        IdealPath("IMAGE_EDGE", "image", "find", "spatial",
            [CognitiveAction.IDENTIFY_STRUCTURE, CognitiveAction.QUERY_ALGORITHMS, CognitiveAction.SELECT_ALGORITHM, CognitiveAction.TEST_EXAMPLES, CognitiveAction.ACCEPT_SOLUTION],
            "edge_detection", 5),
        IdealPath("BACKTRACK_PERMUTATION", "numbers", "find", "constraint_satisfaction",
            [CognitiveAction.IDENTIFY_STRUCTURE, CognitiveAction.QUERY_PRINCIPLE, CognitiveAction.QUERY_ALGORITHMS, CognitiveAction.SELECT_ALGORITHM, CognitiveAction.TEST_EXAMPLES, CognitiveAction.ACCEPT_SOLUTION],
            "permutation", 6),
        IdealPath("BACKTRACK_SUBSET", "numbers", "find", "constraint_satisfaction",
            [CognitiveAction.IDENTIFY_STRUCTURE, CognitiveAction.QUERY_PRINCIPLE, CognitiveAction.SELECT_ALGORITHM, CognitiveAction.TEST_EXAMPLES, CognitiveAction.ACCEPT_SOLUTION],
            "subset_sum", 5),
    ]


@dataclass
class TrainingStep:
    problem_text: str
    data_features: torch.Tensor
    action_history: torch.Tensor
    action_mask: torch.Tensor
    target_action: CognitiveAction
    reward: float


@dataclass
class TrainingTrajectory:
    steps: list[TrainingStep] = field(default_factory=list)
    total_reward: float = 0.0


def generate_training_data(
    kg: AlgorithmicKnowledgeGraph,
    config: MindConfig,
    n_augmentations: int = 5,
) -> list[TrainingTrajectory]:
    ideal_paths = get_ideal_paths()
    trajectories = []

    for path in ideal_paths:
        for aug in range(n_augmentations):
            traj = _generate_single_trajectory(path, kg, config, aug)
            if traj and traj.steps:
                trajectories.append(traj)

    return trajectories


def _generate_single_trajectory(
    path: IdealPath,
    kg: AlgorithmicKnowledgeGraph,
    config: MindConfig,
    augmentation: int,
) -> TrainingTrajectory | None:
    traj = TrainingTrajectory()

    problem_text = _generate_problem_text(path, augmentation)
    data = _generate_problem_data(path, augmentation)
    data_features = build_data_profile(data)

    history_len = config.state_history_len
    n_actions = config.n_cognitive_actions
    action_history = torch.zeros(history_len, n_actions)
    action_mask = torch.ones(n_actions, dtype=torch.bool)

    for step_idx, target_action in enumerate(path.ideal_actions):
        if step_idx == len(path.ideal_actions) - 1:
            reward = 2.0
        elif target_action == CognitiveAction.TEST_EXAMPLES:
            reward = 0.5
        elif target_action in (CognitiveAction.IDENTIFY_STRUCTURE, CognitiveAction.QUERY_PRINCIPLE):
            reward = 0.3
        else:
            reward = 0.2

        full_features = _pad_state_features(data_features)

        step = TrainingStep(
            problem_text=problem_text,
            data_features=full_features,
            action_history=action_history.clone(),
            action_mask=action_mask.clone(),
            target_action=target_action,
            reward=reward,
        )
        traj.steps.append(step)
        traj.total_reward += reward

        action_history[:-1] = action_history[1:].clone()
        action_history[-1] = 0
        action_history[-1, target_action.value] = 1.0

    return traj


def _generate_problem_text(path: IdealPath, aug: int) -> str:
    templates = {
        "SORTING": ["Sort the array of integers", "Arrange the elements in ascending order", "Return a sorted version of the input array", "Order the numbers from smallest to largest", "Sort the given list of numbers"],
        "SORTING_SMALL": ["Sort a small array of at most 20 integers", "Arrange a short list of numbers", "Sort the few elements in the array", "Order the small collection of integers", "Sort the tiny array efficiently"],
        "SORTING_STABLE": ["Sort the array while preserving relative order of equal elements", "Stably sort the integers", "Sort with a stable sorting algorithm", "Arrange elements keeping equal items in original order", "Perform a stable sort on the array"],
        "PATHFINDING_WEIGHTED": ["Find the shortest path in a weighted graph", "Compute the minimum cost path between nodes", "Find the least expensive route through the graph", "Determine the shortest weighted path", "Calculate the optimal path in a weighted network"],
        "PATHFINDING_UNWEIGHTED": ["Find the shortest path in an unweighted graph", "Compute minimum hops between nodes", "Find the path with fewest edges", "Determine shortest route in unweighted network", "Find minimum steps between two nodes"],
        "PATHFINDING_NEGATIVE_WEIGHTS": ["Find shortest path with negative edge weights", "Compute shortest path allowing negative weights", "Handle negative edges in shortest path calculation", "Find minimum path with possible negative costs", "Shortest path where edges can be negative"],
        "SEARCH_SORTED": ["Find the target in a sorted array", "Locate the element in an ordered list", "Search for the value in a sorted sequence", "Determine if target exists in sorted array", "Find the position of target in sorted data"],
        "SEARCH_UNSORTED": ["Find the target in an unsorted array", "Locate the element in an arbitrary list", "Search for the value in a collection", "Determine if target exists in the array", "Find whether the element is present"],
        "DP_KNAPSACK": ["Select items with maximum value within capacity", "Choose items to maximize value without exceeding weight limit", "Solve the 0/1 knapsack problem", "Pack maximum value into limited capacity", "Find optimal item selection given weight constraint"],
        "DP_LCS": ["Find the longest common subsequence of two strings", "Compute the LCS of two sequences", "Determine the longest shared subsequence", "Find the longest subsequence common to both strings", "Calculate the maximum common subsequence length"],
        "DP_LIS": ["Find the longest increasing subsequence", "Compute the maximum length increasing subsequence", "Determine the longest strictly increasing subsequence", "Find the longest subsequence where each element is larger", "Calculate the LIS of the array"],
        "DP_COIN_CHANGE": ["Find minimum coins to make the amount", "Compute fewest coins needed for target sum", "Determine minimum number of coins for amount", "Find the least coins to form the value", "Calculate minimum coin count for the target"],
        "GREEDY_INTERVAL": ["Select maximum number of non-overlapping intervals", "Choose the most intervals without overlap", "Find maximum set of compatible activities", "Schedule the most activities without conflict", "Select maximum non-overlapping ranges"],
        "GREEDY_HUFFMAN": ["Build optimal prefix code for characters", "Construct Huffman encoding tree", "Create minimum cost prefix code", "Generate optimal compression codes", "Build Huffman tree for minimum encoding length"],
        "GREEDY_MST": ["Find minimum spanning tree of the graph", "Compute the MST with minimum total weight", "Determine minimum cost spanning tree", "Find the spanning tree with least total edge weight", "Calculate the minimum spanning tree"],
        "BINARY_SEARCH_MIN_MAX": ["Find minimum maximum value in split array", "Minimize the maximum element across partitions", "Find optimal split minimizing largest sum", "Determine minimum possible maximum", "Binary search for minimum feasible maximum"],
        "HASH_TWO_SUM": ["Find two numbers that sum to target", "Locate pair of elements adding to target", "Find indices of two elements with target sum", "Determine which two numbers equal the target", "Find the complement pair for target sum"],
        "HASH_FREQUENCY": ["Find elements appearing more than n/3 times", "Identify majority elements in array", "Find frequent elements above threshold", "Determine which values appear most often", "Find elements with high frequency"],
        "GRAPH_TOPOLOGICAL": ["Find topological ordering of DAG", "Compute topological sort of directed graph", "Determine valid ordering of nodes", "Sort nodes in dependency order", "Find linear ordering respecting all edges"],
        "GRAPH_CONNECTED": ["Find connected components of graph", "Determine which nodes are connected", "Compute all connected components", "Find groups of connected nodes", "Identify separate connected regions"],
        "STRING_PATTERN": ["Find pattern occurrences in text", "Locate all matches of pattern in string", "Search for substring pattern", "Find all positions where pattern appears", "Determine pattern matches in the text"],
        "STRING_EDIT_DISTANCE": ["Compute edit distance between two strings", "Find minimum edits to transform one string to another", "Calculate Levenshtein distance", "Determine minimum operations to convert strings", "Find edit distance between the strings"],
        "ML_CLASSIFY": ["Classify data points into categories", "Assign labels to feature vectors", "Train a classifier on the data", "Predict classes for input features", "Build classification model from data"],
        "ML_REGRESS": ["Predict continuous values from features", "Fit a regression model to the data", "Estimate target values from input features", "Train regression on feature matrix", "Build regression model for prediction"],
        "CLUSTERING": ["Group data points into clusters", "Partition data into k groups", "Find natural groupings in the data", "Cluster the feature vectors", "Segment data into similar groups"],
        "NLP_CLASSIFY": ["Classify text documents into categories", "Assign labels to text inputs", "Determine category for each text", "Build text classifier", "Predict text labels from content"],
        "TREE_BST": ["Insert and search in binary search tree", "Perform BST operations", "Manage binary search tree data structure", "Build and query BST", "Implement BST insert and find"],
        "TREE_TRIE": ["Build prefix tree for strings", "Implement trie for word lookup", "Create prefix matching data structure", "Build trie for autocomplete", "Implement trie operations"],
        "MATH_GCD": ["Compute greatest common divisor", "Find GCD of two numbers", "Calculate GCD using Euclidean algorithm", "Determine greatest common factor", "Find the largest common divisor"],
        "MATH_PRIMES": ["Find all prime numbers up to n", "Generate primes using sieve", "Sieve of Eratosthenes for primes", "List all primes below limit", "Compute prime numbers efficiently"],
        "IMAGE_EDGE": ["Detect edges in image", "Find edges using gradient detection", "Apply edge detection filter", "Identify edges in the image", "Compute edge map from image"],
        "BACKTRACK_PERMUTATION": ["Generate all permutations", "Find all arrangements of elements", "List every possible ordering", "Compute all permutations of array", "Generate complete set of permutations"],
        "BACKTRACK_SUBSET": ["Find subset that sums to target", "Determine if subset with target sum exists", "Search for subset with given total", "Find elements adding to target value", "Check if subset sum equals target"],
    }
    options = templates.get(path.problem_type, [
        f"Solve the {path.problem_type.lower()} problem",
        f"Find the solution for {path.domain} {path.optimization_goal}",
        f"Compute the {path.hidden_structure} result",
        f"Determine the answer for {path.problem_type.lower()}",
        f"Process the {path.domain} input",
    ])
    return options[aug % len(options)]


def _generate_problem_data(path: IdealPath, aug: int) -> Any:
    seed = hash(path.problem_type + str(aug)) % (2**31)
    rng = random.Random(seed)

    sizes = [10, 50, 100, 500, 1000]
    n = sizes[aug % len(sizes)]

    if path.domain in ("integers", "numbers"):
        return {"nums": [rng.randint(-100, 100) for _ in range(n)]}
    elif path.domain == "array":
        return {"nums": sorted([rng.randint(0, 1000) for _ in range(n)])}
    elif path.domain == "graph":
        nodes = min(n, 20)
        edges = []
        for _ in range(nodes * 2):
            u, v = rng.randint(0, nodes-1), rng.randint(0, nodes-1)
            if u != v:
                w = rng.randint(1, 10)
                edges.append([u, v, w])
        return {"edges": edges, "n": nodes}
    elif path.domain == "text":
        words = ["abc", "def", "ghi", "jkl", "mno", "pqr", "stu", "vwx", "yz"]
        return {"s": "".join(rng.choice(words) for _ in range(n // 3))}
    elif path.domain == "feature_matrix":
        d = min(10, n)
        return {"X": [[rng.random() for _ in range(d)] for _ in range(n)],
                "y": [rng.randint(0, 2) for _ in range(n)]}
    else:
        return {"nums": [rng.randint(0, 100) for _ in range(n)]}


class BootstrapTrainer:
    def __init__(
        self,
        mind: AlgorithmicMind,
        kg: AlgorithmicKnowledgeGraph,
        config: MindConfig | None = None,
    ) -> None:
        self.mind = mind
        self.kg = kg
        self.config = config or MindConfig()

    def train(
        self,
        n_epochs: int = 10,
        n_augmentations: int = 5,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        log_interval: int = 100,
    ) -> dict:
        trajectories = generate_training_data(
            self.kg, self.config, n_augmentations
        )

        if not trajectories:
            return {"error": "No training data generated"}

        all_steps = []
        for traj in trajectories:
            all_steps.extend(traj.steps)

        stats = {
            "n_trajectories": len(trajectories),
            "n_steps": len(all_steps),
            "n_epochs": n_epochs,
            "phase1_loss": [],
            "phase2_reward": [],
        }

        optimizer = torch.optim.AdamW(
            self.mind.parameters(), lr=learning_rate
        )

        for epoch in range(n_epochs):
            random.shuffle(all_steps)

            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(all_steps), batch_size):
                batch = all_steps[i:i + batch_size]
                if not batch:
                    continue

                loss = self._behavioral_cloning_loss(batch)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.mind.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            stats["phase1_loss"].append(avg_loss)

        ppo_optimizer = torch.optim.AdamW(
            self.mind.parameters(), lr=learning_rate * 0.1
        )
        for epoch in range(n_epochs // 2):
            random.shuffle(all_steps)

            total_policy_loss = 0.0
            total_value_loss = 0.0
            n_batches = 0

            for i in range(0, len(all_steps), batch_size):
                batch = all_steps[i:i + batch_size]
                if not batch:
                    continue

                policy_loss = torch.tensor(0.0)
                value_loss = torch.tensor(0.0)

                for step in batch:
                    problem_tokens = torch.zeros(1, 32, dtype=torch.long)
                    state_features = step.data_features.unsqueeze(0)
                    action_history = step.action_history.unsqueeze(0)
                    action_mask = step.action_mask.unsqueeze(0)

                    action_probs, value, fused = self.mind.forward(
                        problem_tokens, state_features, action_history, action_mask
                    )

                    target = torch.tensor([step.target_action.value])
                    log_probs = torch.log(action_probs.squeeze(0) + 1e-8)
                    pl = F.nll_loss(log_probs.unsqueeze(0), target)
                    policy_loss = policy_loss + pl

                    target_val = torch.tensor([step.reward], dtype=torch.float32)
                    vl = F.mse_loss(value.squeeze(), target_val)
                    value_loss = value_loss + vl

                loss = policy_loss + 0.5 * value_loss

                ppo_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.mind.parameters(), 1.0)
                ppo_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                n_batches += 1

            avg_reward = sum(t.total_reward for t in trajectories) / max(len(trajectories), 1)
            stats["phase2_reward"].append(avg_reward)

        return stats

    def _behavioral_cloning_loss(
        self,
        batch: list[TrainingStep],
    ) -> torch.Tensor:
        total_loss = torch.tensor(0.0)

        for step in batch:
            problem_tokens = torch.zeros(1, 32, dtype=torch.long)
            state_features = step.data_features.unsqueeze(0)
            action_history = step.action_history.unsqueeze(0)
            action_mask = step.action_mask.unsqueeze(0)

            action_probs, value, fused = self.mind.forward(
                problem_tokens, state_features, action_history, action_mask
            )

            target = torch.tensor([step.target_action.value])
            log_probs = torch.log(action_probs.squeeze(0) + 1e-8)
            loss = F.nll_loss(log_probs.unsqueeze(0), target)
            total_loss = total_loss + loss

        return total_loss / len(batch)

    def evaluate(self, n_samples: int = 100) -> dict:
        ideal_paths = get_ideal_paths()
        correct = 0
        total = 0

        self.mind.eval()

        with torch.no_grad():
            for path in ideal_paths:
                for aug in range(max(1, n_samples // len(ideal_paths))):
                    problem_text = _generate_problem_text(path, aug)
                    data = _generate_problem_data(path, aug)
                    data_features = _build_full_state(data)

                    n_actions = self.config.n_cognitive_actions
                    history_len = self.config.state_history_len
                    action_history = torch.zeros(history_len, n_actions)
                    action_mask = torch.ones(n_actions, dtype=torch.bool)

                    predicted_actions = []
                    for _ in range(path.expected_iterations + 3):
                        problem_tokens = torch.zeros(1, 32, dtype=torch.long)
                        state_features = data_features.unsqueeze(0)
                        action_history_tensor = action_history.unsqueeze(0)
                        action_mask_tensor = action_mask.unsqueeze(0)

                        action_probs, _, _ = self.mind.forward(
                            problem_tokens, state_features,
                            action_history_tensor, action_mask_tensor,
                        )

                        predicted = torch.argmax(action_probs.squeeze(0)).item()
                        try:
                            action = CognitiveAction(predicted)
                        except ValueError:
                            break
                        predicted_actions.append(action)

                        action_history[:-1] = action_history[1:].clone()
                        action_history[-1] = 0
                        action_history[-1, action.value] = 1.0

                        if action == CognitiveAction.ACCEPT_SOLUTION:
                            break

                    if CognitiveAction.SELECT_ALGORITHM in predicted_actions:
                        correct += 1

                    total += 1

        return {
            "accuracy": correct / max(total, 1),
            "correct": correct,
            "total": total,
        }
