def build_synthesis_prompt(spec, data_sample) -> str:
    problem_desc = spec.problem_type.name.lower()
    constraints = ", ".join([str(c) for c in spec.constraints])
    objectives = ", ".join([o.metric for o in spec.objectives])

    if isinstance(data_sample, list):
        data_preview = f"list of {len(data_sample)} items, type: {type(data_sample[0]).__name__}"
    elif isinstance(data_sample, dict):
        data_preview = f"dict with keys: {list(data_sample.keys())[:3]}..."
    else:
        data_preview = str(type(data_sample))

    prompt = f"""You are a Python algorithm expert. Write a function that solves this problem:

Problem type: {problem_desc}
Constraints: {constraints}
Objectives: {objectives}
Input: {data_preview}

Requirements:
1. Define a function `process(data)` that takes the input and returns the solution
2. Use only standard Python built-ins (no imports)
3. Optimize for {objectives} within {constraints}
4. Handle edge cases (empty input, single item, duplicates)
5. Keep it simple and readable

Return only the Python code, no explanation.

Example format:
```python
def process(data):
    # Your implementation
    return result
```"""

    return prompt
