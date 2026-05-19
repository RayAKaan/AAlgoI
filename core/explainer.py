import time
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from core.problem_spec import ProblemSpec
from algorithms.primitives import PRIMITIVES

logger = logging.getLogger(__name__)

EXPLANATION_TEMPLATES = {
    "quicksort": {
        "summary": "Quicksort is a divide-and-conquer sorting algorithm that selects a pivot, partitions elements around it, and recursively sorts the sub-arrays.",
        "complexity": "O(n log n) average, O(n²) worst case",
        "steps": [
            "Choose a pivot element from the array",
            "Partition: rearrange so elements < pivot come before, elements > pivot come after",
            "Recursively apply quicksort to the left and right sub-arrays"
        ],
        "best_for": "General-purpose sorting with good average-case performance"
    },
    "mergesort": {
        "summary": "Mergesort is a stable divide-and-conquer sorting algorithm that splits the array, recursively sorts each half, and merges them.",
        "complexity": "O(n log n) guaranteed",
        "steps": [
            "Divide the array into two equal halves",
            "Recursively sort each half",
            "Merge the two sorted halves into a single sorted array"
        ],
        "best_for": "Stable sorting and large datasets where consistent performance is needed"
    },
    "heapsort": {
        "summary": "Heapsort builds a max-heap from the data, then repeatedly extracts the maximum element to build the sorted array.",
        "complexity": "O(n log n) guaranteed, O(1) auxiliary space",
        "steps": [
            "Build a max-heap from the input array",
            "Repeatedly swap the root (maximum) with the last element",
            "Heapify the reduced heap to restore the heap property"
        ],
        "best_for": "In-place sorting with guaranteed O(n log n) performance"
    },
    "binary_search": {
        "summary": "Binary search finds an element in a sorted array by repeatedly dividing the search interval in half.",
        "complexity": "O(log n)",
        "steps": [
            "Compare the target value with the middle element",
            "If equal, return the index",
            "If target < middle, search the left half; otherwise search the right half",
            "Repeat until the element is found or the interval is empty"
        ],
        "best_for": "Fast search on sorted data"
    },
    "linear_search": {
        "summary": "Linear search sequentially checks each element of the list until a match is found.",
        "complexity": "O(n)",
        "steps": [
            "Traverse the array from the first element",
            "Compare each element with the target",
            "Return the index if found, -1 if not found after checking all elements"
        ],
        "best_for": "Small datasets or unsorted data"
    },
    "interpolation_search": {
        "summary": "Interpolation search estimates the position of the target using the value distribution, like looking up a word in a dictionary.",
        "complexity": "O(log log n) average for uniform data, O(n) worst case",
        "steps": [
            "Estimate the target position using linear interpolation",
            "Compare and narrow the search range accordingly",
            "Repeat until found or range is empty"
        ],
        "best_for": "Uniformly distributed sorted data"
    },
    "dfs": {
        "summary": "Depth-First Search explores a graph by going as deep as possible along each branch before backtracking.",
        "complexity": "O(V + E)",
        "steps": [
            "Start at the root node and mark it visited",
            "Recursively visit each unvisited neighbor",
            "Backtrack when no unvisited neighbors remain"
        ],
        "best_for": "Path finding, cycle detection, topological ordering"
    },
    "bfs": {
        "summary": "Breadth-First Search explores a graph level by level, visiting all neighbors before moving to the next depth.",
        "complexity": "O(V + E)",
        "steps": [
            "Start at the root node and add it to a queue",
            "Dequeue a node and visit all its unvisited neighbors",
            "Enqueue each unvisited neighbor and continue until the queue is empty"
        ],
        "best_for": "Shortest path in unweighted graphs, level-order traversal"
    },
    "greedy": {
        "summary": "A greedy algorithm makes the locally optimal choice at each step, hoping to find the global optimum.",
        "complexity": "Depends on the problem (typically O(n log n))",
        "steps": [
            "Initialize with an empty solution",
            "At each step, make the best immediate choice",
            "Check if the choice leads to a valid solution",
            "Repeat until a complete solution is found"
        ],
        "best_for": "Problems with optimal substructure where local choices lead to global optima"
    },
    "dynamic_programming": {
        "summary": "Dynamic programming solves complex problems by breaking them into overlapping subproblems and storing results to avoid recomputation.",
        "complexity": "Depends on subproblem count (typically O(n²) or O(n*m))",
        "steps": [
            "Define the state and recurrence relation",
            "Initialize base cases",
            "Fill the DP table bottom-up (or use memoization top-down)",
            "Extract the final answer from the table"
        ],
        "best_for": "Problems with overlapping subproblems and optimal substructure"
    },
    "backtracking": {
        "summary": "Backtracking incrementally builds candidates and abandons them as soon as they are determined to be invalid.",
        "complexity": "O(2ⁿ) in worst case",
        "steps": [
            "Start with an empty solution",
            "Extend the solution incrementally",
            "If the current solution is invalid, backtrack (undo the last step)",
            "If complete, record the solution and continue searching"
        ],
        "best_for": "Constraint satisfaction, permutations, combinations"
    },
    "topological_sort": {
        "summary": "Topological sort orders nodes in a directed acyclic graph so every edge goes from earlier to later nodes.",
        "complexity": "O(V + E)",
        "steps": [
            "Compute in-degree for each vertex",
            "Add all vertices with in-degree 0 to a queue",
            "While the queue is not empty, dequeue a vertex, add it to result, reduce in-degree of its neighbors"
        ],
        "best_for": "Dependency resolution, task scheduling"
    },
    "union_find": {
        "summary": "Union-Find tracks elements partitioned into disjoint sets, supporting efficient union and find operations.",
        "complexity": "Near O(1) with path compression and union by rank",
        "steps": [
            "Initialize each element as its own set",
            "Find: determine which set an element belongs to",
            "Union: merge two sets together"
        ],
        "best_for": "Connectivity queries, Kruskal's MST, cycle detection in graphs"
    },
    "rabin_karp": {
        "summary": "Rabin-Karp uses rolling hash to find a pattern in a text, comparing hash values before doing character-by-character checks.",
        "complexity": "O(n + m) average, O(n*m) worst case",
        "steps": [
            "Compute the hash of the pattern and the first window of text",
            "Slide the window across the text, updating the rolling hash",
            "When hashes match, verify with character comparison"
        ],
        "best_for": "Multiple pattern matching, plagiarism detection"
    },
    "two_pointer": {
        "summary": "Two-pointer uses two indices to traverse a data structure, often from opposite ends, to find pairs or satisfy conditions.",
        "complexity": "O(n)",
        "steps": [
            "Initialize two pointers at opposite ends (or at the start)",
            "Move pointers based on comparison results",
            "Stop when pointers meet or condition is satisfied"
        ],
        "best_for": "Sorted array pair search, palindrome checking"
    },
    "sliding_window": {
        "summary": "Sliding window maintains a contiguous subarray of fixed or variable size, updating as it moves across the data.",
        "complexity": "O(n)",
        "steps": [
            "Initialize the window with the first k elements",
            "Slide the window one element at a time, updating the result",
            "Track the optimal window state across all positions"
        ],
        "best_for": "Subarray/substring problems, streaming data"
    },
    "partition": {
        "summary": "Partition divides an array around a pivot, used as a building block for quicksort and selection algorithms.",
        "complexity": "O(n)",
        "steps": [
            "Select a pivot element",
            "Rearrange elements so those less than pivot come first",
            "Elements equal to pivot come next, then those greater than pivot"
        ],
        "best_for": "Quicksort building block, quickselect"
    },
    "map": {
        "summary": "Map applies a transformation function to each element of a collection, producing a new collection.",
        "complexity": "O(n)",
        "steps": [
            "Iterate through each element",
            "Apply the transformation function",
            "Collect transformed elements into the output"
        ],
        "best_for": "Element-wise transformations"
    },
    "filter": {
        "summary": "Filter selects elements from a collection that satisfy a given predicate.",
        "complexity": "O(n)",
        "steps": [
            "Iterate through each element",
            "Apply the predicate function",
            "Collect elements for which the predicate returns True"
        ],
        "best_for": "Selection and filtering operations"
    },
    "reduce": {
        "summary": "Reduce combines all elements of a collection into a single value using a combining function.",
        "complexity": "O(n)",
        "steps": [
            "Start with an initial value (or the first element)",
            "Apply the combining function to accumulate results",
            "Return the final accumulated value"
        ],
        "best_for": "Aggregation operations like sum, product, min, max"
    },
    "lcs": {
        "summary": "Longest Common Subsequence finds the longest sequence that appears in the same order in both inputs.",
        "complexity": "O(n*m)",
        "steps": [
            "Build a DP table where dp[i][j] = LCS of prefixes a[:i] and b[:j]",
            "If characters match, dp[i][j] = dp[i-1][j-1] + 1",
            "Otherwise, dp[i][j] = max(dp[i-1][j], dp[i][j-1])",
            "Return dp[n][m] as the LCS length"
        ],
        "best_for": "Sequence comparison, diff tools, bioinformatics"
    }
}


@dataclass
class Explanation:
    algorithm_name: str
    summary: str
    complexity: str
    steps: List[str]
    best_for: str
    source: str
    detail_level: str
    execution_time_ms: float = 0.0


class Explainer:
    def __init__(self, llm_client: Optional[Any] = None,
                 default_detail: str = "short"):
        self.llm_client = llm_client
        self.default_detail = default_detail

    def explain(self, algorithm_name: str,
                detail: Optional[str] = None,
                problem_spec: Optional[ProblemSpec] = None,
                context: Optional[Dict] = None) -> Explanation:
        start = time.perf_counter()
        detail = detail or self.default_detail

        if detail == "short":
            result = self._template_explain(algorithm_name, context)
        elif detail == "detailed" and self.llm_client:
            result = self._llm_explain(algorithm_name, problem_spec, context)
        else:
            result = self._template_explain(algorithm_name, context)

        result.execution_time_ms = (time.perf_counter() - start) * 1000
        return result

    def explain_pipeline(self, pipeline_names: List[str],
                         detail: Optional[str] = None,
                         problem_spec: Optional[ProblemSpec] = None) -> List[Explanation]:
        return [
            self.explain(name, detail=detail, problem_spec=problem_spec)
            for name in pipeline_names
        ]

    def _template_explain(self, algorithm_name: str,
                          context: Optional[Dict] = None) -> Explanation:
        normalized = algorithm_name.lower().replace("_primitive", "").replace("_", "_")
        template = EXPLANATION_TEMPLATES.get(normalized)

        if template is None:
            template = EXPLANATION_TEMPLATES.get(
                next((k for k in EXPLANATION_TEMPLATES if k in normalized), None),
                {
                    "summary": f"{algorithm_name} is a custom algorithm that processes data according to its specific logic.",
                    "complexity": "Unknown (not profiled)",
                    "steps": [
                        f"Execute {algorithm_name} on the input data",
                        "Return the processed result"
                    ],
                    "best_for": "Custom use cases"
                }
            )

        explanation = Explanation(
            algorithm_name=algorithm_name,
            summary=template["summary"],
            complexity=template["complexity"],
            steps=list(template["steps"]),
            best_for=template["best_for"],
            source="template",
            detail_level="short"
        )

        if context:
            explanation.summary += f" (Context: {context.get('task', 'general')})"

        return explanation

    def _llm_explain(self, algorithm_name: str,
                     problem_spec: Optional[ProblemSpec] = None,
                     context: Optional[Dict] = None) -> Explanation:
        if not self.llm_client:
            return self._template_explain(algorithm_name, context)

        try:
            prompt = self._build_explain_prompt(algorithm_name, problem_spec, context)
            response = self.llm_client.generate(prompt)
            content = response.get("explanation", "") if isinstance(response, dict) else str(response)

            return Explanation(
                algorithm_name=algorithm_name,
                summary=content,
                complexity="See explanation",
                steps=[],
                best_for="",
                source="llm",
                detail_level="detailed"
            )
        except Exception as e:
            logger.warning(f"LLM explanation failed for {algorithm_name}: {e}")
            return self._template_explain(algorithm_name, context)

    def _build_explain_prompt(self, algorithm_name: str,
                              problem_spec: Optional[ProblemSpec] = None,
                              context: Optional[Dict] = None) -> str:
        parts = [f"Explain the {algorithm_name} algorithm in detail."]
        if problem_spec:
            parts.append(f"Problem: {problem_spec.to_dict()}")
        if context:
            parts.append(f"Context: {json.dumps(context)}")
        parts.append("Include: how it works, time/space complexity, and when to use it.")
        return "\n".join(parts)

    def list_available_explanations(self) -> List[str]:
        return sorted(EXPLANATION_TEMPLATES.keys())
