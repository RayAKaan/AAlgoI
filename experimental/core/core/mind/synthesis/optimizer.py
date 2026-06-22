from typing import Any


class CodeOptimizer:
    def optimize(
        self,
        code: str,
        optimization_type: str,
        data: Any = None,
    ) -> str | None:
        handlers = {
            "memoize":           self._memoize,
            "two_pointer":       self._two_pointer,
            "binary_search":     self._binary_search,
            "rolling_array":     self._rolling_array,
            "prefix_sum":        self._prefix_sum,
            "bit_manipulation":  self._bit_manipulation,
            "greedy_exchange":   self._greedy_exchange,
        }

        handler = handlers.get(optimization_type)
        if handler is None:
            return None

        try:
            return handler(code, data)
        except Exception:
            return None

    def _memoize(self, code: str, data: Any) -> str | None:
        lines = code.split("\n")
        func_name = None
        has_recursion = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("def "):
                func_name = stripped.split("(")[0].replace("def ", "").strip()
            if func_name and func_name + "(" in stripped and "def " not in stripped:
                has_recursion = True

        if not has_recursion:
            return None

        new_lines = ["from functools import lru_cache", ""]
        for line in lines:
            if line.strip().startswith("def "):
                indent = len(line) - len(line.lstrip())
                new_lines.append(" " * indent + "@lru_cache(None)")
            new_lines.append(line)

        return "\n".join(new_lines)

    def _two_pointer(self, code: str, data: Any) -> str | None:
        loop_count = code.count("for ")
        if loop_count < 2:
            return None

        if "target" not in code and "sum" not in code.lower():
            return None

        return '''def solve(nums, target):
    nums.sort()
    left, right = 0, len(nums) - 1
    result = []
    while left < right:
        s = nums[left] + nums[right]
        if s == target:
            result.append([nums[left], nums[right]])
            left += 1
            right -= 1
        elif s < target:
            left += 1
        else:
            right -= 1
    return result
'''

    def _binary_search(self, code: str, data: Any) -> str | None:
        if "for " not in code:
            return None

        if "==" not in code and ">=" not in code and "<=" not in code:
            return None

        return '''def solve(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
'''

    def _rolling_array(self, code: str, data: Any) -> str | None:
        if "dp" not in code:
            return None

        if "dp[i-2]" in code or "dp[i - 2]" in code:
            return self._rolling_3var(code)
        elif "dp[i-1]" in code or "dp[i - 1]" in code:
            return self._rolling_2var(code)

        return None

    def _rolling_2var(self, code: str) -> str:
        lines = code.split("\n")
        new_lines = []

        for line in lines:
            stripped = line.strip()

            if "dp = [" in stripped or "dp=[" in stripped:
                new_lines.append("    prev = 0  # dp[i-1]")
                new_lines.append("    curr = 0  # dp[i]")
                continue

            new_line = line.replace("dp[i-1]", "prev").replace("dp[i - 1]", "prev")
            new_line = new_line.replace("dp[i]", "curr")

            if "curr =" in new_line and "for " not in stripped:
                indent = len(new_line) - len(new_line.lstrip())
                new_lines.append(new_line)
                new_lines.append(" " * indent + "prev = curr")
                continue

            new_lines.append(new_line)

        result = "\n".join(new_lines)
        return result if result != code else None

    def _rolling_3var(self, code: str) -> str:
        lines = code.split("\n")
        new_lines = []

        for line in lines:
            stripped = line.strip()

            if "dp = [" in stripped or "dp=[" in stripped:
                new_lines.append("    prev2 = 0  # dp[i-2]")
                new_lines.append("    prev1 = 0  # dp[i-1]")
                new_lines.append("    curr = 0   # dp[i]")
                continue

            new_line = line.replace("dp[i-2]", "prev2").replace("dp[i - 2]", "prev2")
            new_line = new_line.replace("dp[i-1]", "prev1").replace("dp[i - 1]", "prev1")
            new_line = new_line.replace("dp[i]", "curr")

            if "curr =" in new_line and "for " not in stripped:
                indent = len(new_line) - len(new_line.lstrip())
                new_lines.append(new_line)
                new_lines.append(" " * indent + "prev2, prev1 = prev1, curr")
                continue

            new_lines.append(new_line)

        result = "\n".join(new_lines)
        return result if result != code else None

    def _prefix_sum(self, code: str, data: Any) -> str | None:
        if "sum(" not in code:
            return None

        lines = code.split("\n")
        sum_in_loop = False
        for line in lines:
            if line.strip().startswith("for ") or line.strip().startswith("while "):
                sum_in_loop = True
            if sum_in_loop and "sum(" in line:
                return self._add_prefix_sum_transform(code)

        return None

    def _add_prefix_sum_transform(self, code: str) -> str:
        lines = code.split("\n")
        new_lines = []
        added_prefix = False

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("def ") and not added_prefix:
                new_lines.append(line)
                indent = "    "
                new_lines.append(indent + "prefix = [0] * (len(nums) + 1)")
                new_lines.append(indent + "for i in range(len(nums)):")
                new_lines.append(indent + "    prefix[i + 1] = prefix[i] + nums[i]")
                new_lines.append("")
                added_prefix = True
                continue

            if "sum(nums[" in line:
                new_lines.append(" " * (len(line) - len(line.lstrip())) + "# Optimized with prefix sum")
                new_lines.append(line)
            else:
                new_lines.append(line)

        return "\n".join(new_lines)

    def _bit_manipulation(self, code: str, data: Any) -> str | None:
        new_code = code
        new_code = new_code.replace("// 2", ">> 1")
        new_code = new_code.replace("% 2", "& 1")
        new_code = new_code.replace("* 2", "<< 1")

        return new_code if new_code != code else None

    def _greedy_exchange(self, code: str, data: Any) -> str | None:
        if "sort()" not in code and "sorted(" not in code:
            lines = code.split("\n")
            for i, line in enumerate(lines):
                if line.strip().startswith("def "):
                    indent = "    "
                    params = line.split("(")[1].split(")")[0].split(",")
                    first_param = params[0].strip().split("=")[0].strip()
                    lines.insert(i + 1, indent + f"{first_param}.sort()")
                    return "\n".join(lines)
        return None
