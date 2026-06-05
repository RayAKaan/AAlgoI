"""
core/__init__.py

Lazy exports. Nothing loads until accessed.
"""
import sys

_LAZY = {
    # Problem types
    "ProblemSpec":              ("aalgoi.core.problem_spec",       "ProblemSpec"),
    "ProblemType":              ("aalgoi.core.problem_spec",       "ProblemType"),
    "Objective":                ("aalgoi.core.problem_spec",       "Objective"),
    "Constraint":               ("aalgoi.core.problem_spec",       "Constraint"),
    "ProblemLibrary":           ("aalgoi.core.problem_library",    "ProblemLibrary"),
    # Meta controller
    "MetaController":           ("aalgoi.core.meta_controller",    "MetaController"),
    "UniversalMetaController":  ("aalgoi.core.meta_controller",    "UniversalMetaController"),
    # Solvers
    "QuestionParser":           ("aalgoi.core.question_parser",    "QuestionParser"),
    "SmartSolver":              ("aalgoi.core.smart_solver",       "SmartSolver"),
    # Explanation
    "Explainer":                ("aalgoi.core.explainer",          "Explainer"),
    "Explanation":              ("aalgoi.core.explainer",          "Explanation"),
    # Benchmark
    "Benchmarker":              ("aalgoi.core.benchmarker",        "Benchmarker"),
    # Validation
    "LearningValidator":        ("aalgoi.core.validator",          "LearningValidator"),
    "PipelineValidator":        ("aalgoi.core.validator",          "PipelineValidator"),
    "ValidationResult":         ("aalgoi.core.validator",          "ValidationResult"),
    # Algorithm
    "AlgorithmSynthesizer":     ("aalgoi.core.algorithm_synthesizer", "AlgorithmSynthesizer"),
    "SynthesisResult":          ("aalgoi.core.algorithm_synthesizer", "SynthesisResult"),
    "AlgorithmMarketplace":     ("aalgoi.core.algorithm_marketplace", "AlgorithmMarketplace"),
    "AlgorithmMetadata":        ("aalgoi.core.algorithm_marketplace", "AlgorithmMetadata"),
    "AlgorithmEmbedder":        ("aalgoi.core.algorithm_embedder", "AlgorithmEmbedder"),
    # Graph & Knowledge
    "AlgorithmKnowledgeGraph":  ("aalgoi.core.knowledge_graph",    "AlgorithmKnowledgeGraph"),
    "VectorKnowledgeBase":      ("aalgoi.core.knowledge_base",     "VectorKnowledgeBase"),
    "KnowledgeBase":            ("aalgoi.core.knowledge_base",     "KnowledgeBase"),
    # Context & Composition
    "ContextEngine":            ("aalgoi.core.context_engine",     "ContextEngine"),
    "DynamicCompositor":        ("aalgoi.core.compositor",         "DynamicCompositor"),
    # Performance & Metrics
    "PerformanceTracker":       ("aalgoi.core.performance_tracker","PerformanceTracker"),
    "DecisionLog":              ("aalgoi.core.decision_log",       "DecisionLog"),
    "Decision":                 ("aalgoi.core.decision_log",       "Decision"),
    # Bandit & Drift
    "UCB1Bandit":               ("aalgoi.core.bandit",             "UCB1Bandit"),
    "DriftDetector":            ("aalgoi.core.drift_detector",     "DriftDetector"),
    # Pipeline
    "PipelineGraph":            ("aalgoi.core.pipeline_graph",     "PipelineGraph"),
    # Federation
    "FederatedKnowledgeSync":   ("aalgoi.core.federated_sync",     "FederatedKnowledgeSync"),
    # LLM
    "LLMAdapter":               ("aalgoi.core.llm_adapter",        "LLMAdapter"),
    # Registry
    "DynamicRegistry":          ("aalgoi.core.registry_manager",   "DynamicRegistry"),
    "GitHubRegistrySync":       ("aalgoi.core.registry_sync",      "GitHubRegistrySync"),
    "TokenManager":             ("aalgoi.core.token_manager",      "TokenManager"),
    # Optimization
    "ASTOptimizer":             ("aalgoi.core.ast_optimizer",      "ASTOptimizer"),
    # RL
    "PPOAgent":                 ("aalgoi.core.rl.agents.selection_agent", "PPOAgent"),
    "RewardShaper":             ("aalgoi.core.rl.reward_shaper",   "RewardShaper"),
    "AdaptiveRewardShaper":     ("aalgoi.core.rl.reward_shaper",   "AdaptiveRewardShaper"),
    "CurriculumRewardShaper":   ("aalgoi.core.rl.reward_shaper",   "CurriculumRewardShaper"),
}


def __getattr__(name: str):
    if name in _LAZY:
        mod_path, attr = _LAZY[name]
        import importlib
        mod = importlib.import_module(mod_path)
        obj = getattr(mod, attr)
        setattr(sys.modules[__name__], name, obj)
        return obj
    raise AttributeError(f"module 'core' has no attribute '{name}'")


__all__ = list(_LAZY.keys())
