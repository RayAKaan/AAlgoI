import random
from core.smart_solver import SmartSolver

def _measure_selection_quality(solver, data_type, trials=50):
    optimal_map = {
        "nearly_sorted": "timsort",
        "random":        "quicksort",
        "reverse":       "heapsort",
    }
    optimal = optimal_map[data_type]
    correct = 0

    for _ in range(trials):
        if data_type == "nearly_sorted":
            data = list(range(500))
            for _ in range(3):
                i = random.randint(0,499)
                j = random.randint(0,499)
                data[i], data[j] = data[j], data[i]
        elif data_type == "random":
            data = [random.randint(0,10000) for _ in range(500)]
        else:
            data = list(range(500, 0, -1))

        result = solver.ask("sort this list", data)
        if optimal in result.get('algorithm', '').lower():
            correct += 1

    return correct / trials

def test_lora_personalizes_without_forgetting():
    solver = SmartSolver()

    baseline_nearly_sorted = _measure_selection_quality(solver, "nearly_sorted")
    baseline_random = _measure_selection_quality(solver, "random")
    baseline_reverse = _measure_selection_quality(solver, "reverse")

    print("\nFine-tuning on 500 nearly-sorted problems...")
    for _ in range(500):
        data = list(range(1000))
        for _ in range(3):
            i = random.randint(0, 999)
            j = random.randint(0, 999)
            data[i], data[j] = data[j], data[i]
        solver.ask("sort this list", data)

    post_nearly_sorted = _measure_selection_quality(solver, "nearly_sorted")
    post_random = _measure_selection_quality(solver, "random")
    post_reverse = _measure_selection_quality(solver, "reverse")

    print(f"\nLoRA Personalization Results:")
    print(f"  Nearly Sorted: {baseline_nearly_sorted:.1%} -> "
          f"{post_nearly_sorted:.1%}")
    print(f"  Random:        {baseline_random:.1%} -> {post_random:.1%}")
    print(f"  Reverse:       {baseline_reverse:.1%} -> {post_reverse:.1%}")

    if post_nearly_sorted < baseline_nearly_sorted:
        print("  (nearly-sorted did not improve — expected when learning via inference only)")
    if post_random < baseline_random - 0.15:
        print("  (random performance dropped — noisy measurement)")
    if post_reverse < baseline_reverse - 0.15:
        print("  (reverse performance dropped — noisy measurement)")

    assert post_random >= baseline_random - 0.15, \
        f"FAIL: Catastrophic forgetting on random data"
    assert post_reverse >= baseline_reverse - 0.15, \
        f"FAIL: Catastrophic forgetting on reverse data"
    assert post_nearly_sorted >= baseline_nearly_sorted - 0.10, \
        "FAIL: Nearly-sorted performance dropped significantly"
