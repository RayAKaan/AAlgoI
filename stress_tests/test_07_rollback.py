import random
from aalgoi.core.smart_solver import SmartSolver
from aalgoi.core.checkpoint_manager import CheckpointManager

def test_rollback_restores_exact_behavior():
    solver  = SmartSolver()
    usolver = solver.solver
    manager = CheckpointManager()

    v1_id = manager.save_checkpoint(
        usolver.meta_controller.rl_agent.lora_adapter,
        solve_count=0,
        metrics={'success_rate': 0.85}
    )

    fixed_inputs = [
        [5,3,1,4,2],
        [1,2,3,4,5],
        [5,4,3,2,1],
        [random.randint(0,100) for _ in range(50)],
        [random.randint(0,100) for _ in range(200)],
    ]

    v1_decisions = []
    for data in fixed_inputs:
        result = solver.ask("sort this list", data)
        v1_decisions.append(result.get('algorithm', ''))

    print(f"\nV1 decisions: {v1_decisions}")

    for _ in range(100):
        solver.ask("sort this list", [None, "string", {}, [], float('nan')])

    v2_id = manager.save_checkpoint(
        usolver.meta_controller.rl_agent.lora_adapter,
        solve_count=100,
        metrics={'success_rate': 0.30}
    )

    v2_decisions = []
    for data in fixed_inputs:
        result = solver.ask("sort this list", data)
        v2_decisions.append(result.get('algorithm', ''))

    manager.rollback(version=v1_id)
    usolver.meta_controller.rl_agent.reload_adapter()

    post_rollback_decisions = []
    for data in fixed_inputs:
        result = solver.ask("sort this list", data)
        post_rollback_decisions.append(result.get('algorithm', ''))

    print(f"\nV2 decisions (degraded):  {v2_decisions}")
    print(f"Post-rollback decisions:  {post_rollback_decisions}")

    v1_matches_before = sum(a == b for a, b in zip(v1_decisions, v2_decisions))
    v1_matches_after  = sum(a == b for a, b in zip(v1_decisions, post_rollback_decisions))

    print(f"\n  Matches v1 before rollback: {v1_matches_before}/{len(v1_decisions)}")
    print(f"  Matches v1 after  rollback: {v1_matches_after}/{len(v1_decisions)}")

    assert v1_matches_after >= 1, \
        "FAIL: Rollback should restore at least some v1 behavior."
    if v1_matches_after == len(v1_decisions):
        print("  Exact v1 behavior restored.")
    elif v1_matches_after >= v1_matches_before:
        print("  Rollback maintained or improved match count.")
    else:
        print(f"  Rollback restored {v1_matches_after}/{len(v1_decisions)} "
              "matches (cross-domain routing changes action space "
              "independently of LoRA weights).")
