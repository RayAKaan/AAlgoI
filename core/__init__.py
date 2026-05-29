"""
core/__init__.py

Lazy exports. Nothing loads until accessed.
"""
import sys

_LAZY = {
    # Problem types
    "ProblemSpec":              ("core.problem_spec",       "ProblemSpec"),
    "ProblemType":              ("core.problem_spec",       "ProblemType"),
    "Objective":                ("core.problem_spec",       "Objective"),
    "Constraint":               ("core.problem_spec",       "Constraint"),
    "ProblemLibrary":           ("core.problem_library",    "ProblemLibrary"),
    # Meta controller
    "MetaController":           ("core.meta_controller",    "MetaController"),
    "UniversalMetaController":  ("core.meta_controller",    "UniversalMetaController"),
    # Solvers
    "QuestionParser":           ("core.question_parser",    "QuestionParser"),
    "SmartSolver":              ("core.smart_solver",       "SmartSolver"),
    # Explanation
    "Explainer":                ("core.explainer",          "Explainer"),
    "Explanation":              ("core.explainer",          "Explanation"),
    # Benchmark
    "Benchmarker":              ("core.benchmarker",        "Benchmarker"),
    # Validation
    "LearningValidator":        ("core.validator",          "LearningValidator"),
    "PipelineValidator":        ("core.validator",          "PipelineValidator"),
    "ValidationResult":         ("core.validator",          "ValidationResult"),
    # Algorithm
    "AlgorithmSynthesizer":     ("core.algorithm_synthesizer", "AlgorithmSynthesizer"),
    "SynthesisResult":          ("core.algorithm_synthesizer", "SynthesisResult"),
    "AlgorithmMarketplace":     ("core.algorithm_marketplace", "AlgorithmMarketplace"),
    "AlgorithmMetadata":        ("core.algorithm_marketplace", "AlgorithmMetadata"),
    "AlgorithmEmbedder":        ("core.algorithm_embedder", "AlgorithmEmbedder"),
    # Graph & Knowledge
    "AlgorithmKnowledgeGraph":  ("core.knowledge_graph",    "AlgorithmKnowledgeGraph"),
    "VectorKnowledgeBase":      ("core.knowledge_base",     "VectorKnowledgeBase"),
    "KnowledgeBase":            ("core.knowledge_base",     "KnowledgeBase"),
    # Context & Composition
    "ContextEngine":            ("core.context_engine",     "ContextEngine"),
    "DynamicCompositor":        ("core.compositor",         "DynamicCompositor"),
    # Performance & Metrics
    "PerformanceTracker":       ("core.performance_tracker","PerformanceTracker"),
    "DecisionLog":              ("core.decision_log",       "DecisionLog"),
    "Decision":                 ("core.decision_log",       "Decision"),
    # Bandit & Drift
    "UCB1Bandit":               ("core.bandit",             "UCB1Bandit"),
    "DriftDetector":            ("core.drift_detector",     "DriftDetector"),
    # Pipeline
    "PipelineGraph":            ("core.pipeline_graph",     "PipelineGraph"),
    # Federation
    "FederatedKnowledgeSync":   ("core.federated_sync",     "FederatedKnowledgeSync"),
    # LLM
    "LLMAdapter":               ("core.llm_adapter",        "LLMAdapter"),
    # Registry
    "DynamicRegistry":          ("core.registry_manager",   "DynamicRegistry"),
    "GitHubRegistrySync":       ("core.registry_sync",      "GitHubRegistrySync"),
    "TokenManager":             ("core.token_manager",      "TokenManager"),
    # Optimization
    "ASTOptimizer":             ("core.ast_optimizer",      "ASTOptimizer"),
    # RL
    "PPOAgent":                 ("core.rl.agents.selection_agent", "PPOAgent"),
    "RewardShaper":             ("core.rl.reward_shaper",   "RewardShaper"),
    "AdaptiveRewardShaper":     ("core.rl.reward_shaper",   "AdaptiveRewardShaper"),
    "CurriculumRewardShaper":   ("core.rl.reward_shaper",   "CurriculumRewardShaper"),
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
