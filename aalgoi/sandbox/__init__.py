"""
aalgoi.sandbox — User-trainable RL sandbox for algorithm selection.

The sandbox gives users a dynamic, trainable RL policy that runs
alongside the production PPOAgent without touching it.

Usage:
    from aalgoi.sandbox import SandboxRL

    rl = SandboxRL()
    rl.curriculum()                          # train on 25 built-in problems
    rl.teach("classify churn", my_data)      # train on your own problem
    rl.save("my_model.rl")                   # persist checkpoint

    result = rl.solve("sort this", data)     # solve + learn from outcome

    # Multiple independent instances
    nlp_rl = SandboxRL(name="nlp", domains=["NLP"])
    opt_rl  = SandboxRL(name="opt", domains=["OPTIMIZATION"])

    # Fork from a trained instance
    specialist = rl.fork("specialist")
    specialist.teach("custom domain problem", data)

    # List all named instances
    SandboxRL.list()     # ["nlp", "opt", "specialist"]
    SandboxRL.get("nlp") # returns the nlp instance
"""

try:
    from .sandbox_rl import SandboxRL
    __all__ = ["SandboxRL"]
except ImportError as e:
    import warnings
    warnings.warn(
        f"aalgoi.sandbox requires PyTorch: pip install torch. Error: {e}",
        ImportWarning,
        stacklevel=2,
    )
    __all__ = []
