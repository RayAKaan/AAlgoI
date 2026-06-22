from aalgoi.types import Example, ProblemTask

TASK_EXAMPLES: dict[ProblemTask, list[Example]] = {
    ProblemTask.SORT: [
        Example(input=[3, 1, 4, 1, 5], output=[1, 1, 3, 4, 5], description="Sort integers"),
        Example(input=[5, 4, 3, 2, 1], output=[1, 2, 3, 4, 5], description="Reverse sorted"),
    ],
    ProblemTask.BINARY_SEARCH: [
        Example(input={"data": [1, 3, 5, 7, 9], "target": 5}, output=2, description="Find middle element"),
        Example(input={"data": [1, 3, 5, 7, 9], "target": 4}, output=-1, description="Missing element"),
    ],
    ProblemTask.TWO_SUM: [
        Example(input={"data": [2, 7, 11, 15], "target": 9}, output=[0, 1], description="Classic two sum"),
    ],
    ProblemTask.GCD: [
        Example(input={"a": 12, "b": 8}, output=4, description="GCD of 12 and 8"),
        Example(input={"a": 17, "b": 5}, output=1, description="Coprime numbers"),
    ],
    ProblemTask.LCM: [
        Example(input={"a": 12, "b": 8}, output=24, description="LCM of 12 and 8"),
        Example(input={"a": 0, "b": 0}, output=0, description="Zero case"),
    ],
    ProblemTask.FIBONACCI: [
        Example(input={"n": 0}, output=0, description="F(0)"),
        Example(input={"n": 10}, output=55, description="F(10)"),
        Example(input={"n": 50}, output=12586269025, description="F(50) — verifies fast doubling"),
    ],
    ProblemTask.PALINDROME: [
        Example(input={"s": "racecar"}, output=True, description="Palindrome"),
        Example(input={"s": "hello"}, output=False, description="Not a palindrome"),
    ],
    ProblemTask.KMP: [
        Example(input={"text": "abcabcabc", "pattern": "abc"}, output=0, description="Pattern at start"),
        Example(input={"text": "aaaaa", "pattern": "ba"}, output=-1, description="Pattern not found"),
    ],
    ProblemTask.EDIT_DISTANCE: [
        Example(input={"a": "kitten", "b": "sitting"}, output=3, description="Classic example"),
    ],
    ProblemTask.LCS: [
        Example(input={"a": "abcde", "b": "ace"}, output=3, description="Classic LCS"),
    ],
    ProblemTask.KADANE: [
        Example(input={"data": [-2, 1, -3, 4, -1, 2, 1, -5, 4]}, output=6, description="Classic Kadane"),
        Example(input={"data": [-1, -2, -3]}, output=-1, description="All negative"),
        Example(input={"data": []}, output=0, description="Empty array"),
    ],
    ProblemTask.KNAPSACK_01: [
        Example(input={"items": [{"weight": 2, "value": 3}, {"weight": 3, "value": 4}, {"weight": 4, "value": 5}], "capacity": 5}, output={"max_value": 7}, description="Classic 0/1 knapsack"),
    ],
    ProblemTask.CYCLE_DETECTION: [
        Example(input={"graph": {0: [1], 1: [0]}}, output=True, description="Graph with cycle"),
    ],
}
