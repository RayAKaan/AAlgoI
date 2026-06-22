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
    "heap_sort": {
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
    },

    "linear_regression": {
        "summary": "Linear regression models the relationship between features and target as a linear function, minimizing mean squared error.",
        "complexity": "O(n*d\u00b2) training, O(d) prediction",
        "steps": [
            "Compute X^T X matrix",
            "Solve normal equations (X^T X)^-1 X^T y",
            "Return coefficients and intercept"
        ],
        "best_for": "Small datasets with linear relationships, interpretable baseline"
    },
    "ridge": {
        "summary": "Ridge regression adds L2 regularization to linear regression, preventing overfitting on multicollinear data.",
        "complexity": "O(n*d\u00b2) training",
        "steps": [
            "Add penalty term \u03b1 * ||w||\u00b2 to loss",
            "Solve regularized normal equations",
            "Return shrunken coefficients"
        ],
        "best_for": "Data with correlated features or more features than samples"
    },
    "lasso": {
        "summary": "Lasso regression uses L1 regularization, driving some coefficients to zero for automatic feature selection.",
        "complexity": "O(n*d\u00b2) training (iterative)",
        "steps": [
            "Add penalty term \u03b1 * ||w||\u2081 to loss",
            "Optimize via coordinate descent",
            "Return sparse coefficient vector"
        ],
        "best_for": "Feature selection, sparse solutions"
    },
    "logistic_regression": {
        "summary": "Logistic regression models the probability of a binary outcome using the logistic function.",
        "complexity": "O(n*d) per iteration",
        "steps": [
            "Initialize weights",
            "Compute predicted probabilities via sigmoid",
            "Update weights via gradient descent",
            "Repeat until convergence"
        ],
        "best_for": "Binary classification with probabilistic output, baseline classifier"
    },
    "knn": {
        "summary": "K-Nearest Neighbors classifies a point by majority vote of its k nearest neighbors in feature space.",
        "complexity": "O(n*d) prediction (lazy learner, no training)",
        "steps": [
            "Store all training samples",
            "For each query, compute distance to all stored points",
            "Return majority class of k nearest neighbors"
        ],
        "best_for": "Small datasets with well-defined decision boundaries"
    },
    "svm": {
        "summary": "Support Vector Machine finds the hyperplane that maximally separates classes using kernel tricks for nonlinear boundaries.",
        "complexity": "O(n\u00b2*d) training, O(d) prediction",
        "steps": [
            "Map input to feature space via kernel",
            "Find support vectors at margin boundary",
            "Maximize margin between classes"
        ],
        "best_for": "High-dimensional data with clear separation margins, small-to-medium datasets"
    },
    "gaussian_nb": {
        "summary": "Gaussian Naive Bayes assumes features are independent and normally distributed per class. Extremely fast.",
        "complexity": "O(n*d) training, O(d) prediction",
        "steps": [
            "Compute mean and variance per feature per class",
            "Apply Bayes' theorem for prediction",
            "Return class with highest posterior probability"
        ],
        "best_for": "High-dimensional data, real-time prediction, when features are roughly independent"
    },
    "random_forest_classification": {
        "summary": "Random Forest builds multiple decision trees on bootstrapped samples and averages their predictions.",
        "complexity": "O(n*m*log(n)) training, O(log n) prediction",
        "steps": [
            "Bootstrap sample the data",
            "Train a decision tree on each sample with random feature subset",
            "Average predictions across all trees"
        ],
        "best_for": "Tabular data with nonlinear relationships, robust default classifier"
    },
    "xgboost_classification": {
        "summary": "XGBoost is a gradient boosting framework that builds trees sequentially, each correcting errors of the previous ensemble.",
        "complexity": "O(n*m*log(n)) per tree",
        "steps": [
            "Initialize with constant prediction",
            "Compute residuals from current ensemble",
            "Fit a tree to predict residuals",
            "Add tree to ensemble with shrinkage",
            "Repeat for specified number of trees"
        ],
        "best_for": "Tabular data where prediction accuracy is the top priority"
    },
    "kmeans": {
        "summary": "K-Means partitions data into k clusters by minimizing within-cluster variance.",
        "complexity": "O(n*k*i) where i is iterations",
        "steps": [
            "Initialize k centroids randomly",
            "Assign each point to nearest centroid",
            "Update centroids as cluster means",
            "Repeat until convergence"
        ],
        "best_for": "Spherical clusters, large datasets, fast prototyping"
    },
    "dbscan": {
        "summary": "DBSCAN groups together points that are closely packed together, marking points in low-density regions as outliers.",
        "complexity": "O(n*log(n)) with spatial indexing",
        "steps": [
            "For each point, count neighbors within \u03b5 radius",
            "Points with \u2265min_samples neighbors are core points",
            "Expand clusters from core points",
            "Mark isolated points as noise"
        ],
        "best_for": "Arbitrary-shaped clusters, noise detection, outlier detection"
    },
    "gmm": {
        "summary": "Gaussian Mixture Model represents data as a weighted sum of Gaussian distributions, fitted via Expectation-Maximization.",
        "complexity": "O(n*k*d) per iteration",
        "steps": [
            "Initialize k Gaussian components",
            "E-step: compute probability of each point belonging to each component",
            "M-step: update component parameters",
            "Repeat until convergence"
        ],
        "best_for": "Soft clustering, density estimation, overlapping clusters"
    },
    "pca": {
        "summary": "Principal Component Analysis finds orthogonal directions of maximum variance.",
        "complexity": "O(n*d\u00b2) training",
        "steps": [
            "Center the data",
            "Compute covariance matrix",
            "Extract eigenvectors with largest eigenvalues",
            "Project data onto top k components"
        ],
        "best_for": "Visualization, noise reduction, feature extraction, preprocessing"
    },

    "word2vec_trainer": {
        "summary": "Trains Word2Vec embeddings by predicting context words, learning semantic relationships between words.",
        "complexity": "O(V * E) where V=vocabulary size, E=epochs",
        "steps": [
            "Initialize random embeddings for each word",
            "For each word, predict surrounding context words",
            "Update embeddings via backpropagation",
            "Result: words with similar meanings have similar vectors"
        ],
        "best_for": "Training custom embeddings for domain-specific vocabulary"
    },

    "frequency_arithmetic": {
        "summary": "Performs word arithmetic using word frequencies, finding words with similar frequency patterns.",
        "complexity": "O(N) where N=corpus size",
        "steps": [
            "Count word frequencies in corpus",
            "Add/subtract frequency values per the operation",
            "Find words with closest resulting frequency"
        ],
        "best_for": "Simple word analogies on small corpora (Lab 1)"
    },

    "word_vector_arithmetic": {
        "summary": "Performs semantic word arithmetic using pre-trained embeddings like GloVe.",
        "complexity": "O(V) for nearest neighbor search",
        "steps": [
            "Retrieve embeddings for each word in the operation",
            "Add positive vectors, subtract negative vectors",
            "Find nearest neighbors to the result vector"
        ],
        "best_for": "Word analogies (king - man + woman = queen)"
    },

    "embedding_visualization": {
        "summary": "Projects high-dimensional word embeddings into 2D/3D space for visualization.",
        "complexity": "O(V * D\u00b2) for PCA, O(V * iter) for t-SNE",
        "steps": [
            "Load or train word embeddings",
            "Apply PCA or t-SNE dimensionality reduction",
            "Project to 2D or 3D coordinates",
            "Visualize clusters and relationships"
        ],
        "best_for": "Understanding semantic relationships visually"
    },

    "sentiment_analysis": {
        "summary": "Classifies text as positive, negative, or neutral using transformer models.",
        "complexity": "O(L) where L=text length",
        "steps": [
            "Tokenize input text",
            "Pass through fine-tuned transformer (DistilBERT/BERT)",
            "Output sentiment label and confidence score"
        ],
        "best_for": "Analyzing opinions, reviews, social media sentiment"
    },

    "text_summarization": {
        "summary": "Condenses long documents into shorter summaries using seq2seq transformers.",
        "complexity": "O(L\u00b2) due to transformer attention",
        "steps": [
            "Encode input text with encoder (BART/T5)",
            "Decode summary with decoder",
            "Apply length constraints (min/max)"
        ],
        "best_for": "Document summarization, article condensation"
    },

    "rag_retrieval": {
        "summary": "Retrieves relevant passages from documents using semantic similarity.",
        "complexity": "O(N * D) where N=passages, D=embedding dimension",
        "steps": [
            "Split document into passages",
            "Encode passages and query with sentence transformers",
            "Compute cosine similarity",
            "Return top-k most similar passages"
        ],
        "best_for": "Document QA, chatbots, information retrieval (Lab 10)"
    },

    "semantic_search": {
        "summary": "Finds semantically similar texts in a corpus using embedding similarity.",
        "complexity": "O(N * D) where N=corpus size, D=embedding dimension",
        "steps": [
            "Encode all documents in corpus",
            "Encode query",
            "Compute cosine similarities",
            "Return top-k results"
        ],
        "best_for": "Document matching, duplicate detection, semantic retrieval"
    },

    "prompt_enrichment": {
        "summary": "Enhances prompts by adding semantically related terms from embeddings.",
        "complexity": "O(V) for similarity search",
        "steps": [
            "Identify seed word from prompt",
            "Find similar words using embeddings",
            "Append related terms to the prompt"
        ],
        "best_for": "Improving prompt quality for generative AI (Lab 4)"
    },

    "creative_generation": {
        "summary": "Generates creative sentences/stories using word embeddings and templates.",
        "complexity": "O(V) for word similarity",
        "steps": [
            "Find similar words to seed word",
            "Select from template sentences",
            "Combine into paragraph"
        ],
        "best_for": "Creative writing, story generation (Lab 5)"
    },

    "word_expansion": {
        "summary": "Expands a word into related terms and concepts across multiple levels.",
        "complexity": "O(D * V) where D=depth, V=vocabulary",
        "steps": [
            "Find similar words to the input",
            "For each similar word, find more related words",
            "Repeat to specified depth",
            "Collect all terms"
        ],
        "best_for": "Keyword expansion, query broadening, brainstorming"
    }
}
