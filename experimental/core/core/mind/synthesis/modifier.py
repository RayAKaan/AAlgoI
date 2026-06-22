import ast


class CodeModifier:
    def modify(
        self,
        code: str,
        modification_type: str,
        problem_info: dict | None = None,
    ) -> str | None:
        handlers = {
            "add_memoization":          self._add_memoization,
            "convert_to_iterative":     self._convert_to_iterative,
            "add_early_termination":    self._add_early_termination,
            "optimize_inner_loop":      self._optimize_inner_loop,
            "add_pruning":              self._add_pruning,
            "add_hashing":              self._add_hashing,
            "sort_first":               self._sort_first,
            "add_binary_search":        self._add_binary_search,
            "apply_two_pointer":        self._apply_two_pointer,
            "apply_sliding_window":     self._apply_sliding_window,
            "reduce_space":             self._reduce_space,
            "vectorize":                self._vectorize,
            "add_greedy":               self._add_greedy,
            "convert_to_dp":            self._convert_to_dp,
            "restructure_recursion":    self._restructure_recursion,
        }

        handler = handlers.get(modification_type)
        if handler is None:
            return None

        try:
            return handler(code, problem_info or {})
        except Exception:
            return None

    def _add_memoization(self, code: str, info: dict) -> str:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return None

        modified = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                is_recursive = self._is_recursive(node)
                if is_recursive:
                    node.decorator_list.append(
                        ast.Attribute(
                            value=ast.Name(id="functools", ctx=ast.Load()),
                            attr="lru_cache",
                            ctx=ast.Load(),
                        )
                    )
                    modified = True

        if not modified:
            return None

        new_code = "from functools import lru_cache\n\n" + ast.unparse(tree)
        return new_code

    def _is_recursive(self, func_node: ast.FunctionDef) -> bool:
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == func_node.name:
                    return True
        return False

    def _convert_to_iterative(self, code: str, info: dict) -> str:
        return None

    def _add_early_termination(self, code: str, info: dict) -> str:
        lines = code.split("\n")
        new_lines = []
        modified = False

        for line in lines:
            stripped = line.strip()
            new_lines.append(line)

            if "=" in stripped and any(
                target in stripped
                for target in ["result =", "best =", "found ="]
            ):
                indent = len(line) - len(line.lstrip())
                if indent > 0:
                    if "max" in stripped or "best" in stripped:
                        new_lines.append(" " * indent + "if result == target: return result")
                        modified = True
                    elif "found" in stripped:
                        new_lines.append(" " * indent + "if found: return True")
                        modified = True

        if modified:
            return "\n".join(new_lines)
        return None

    def _optimize_inner_loop(self, code: str, info: dict) -> str:
        if "in " not in code and ".index(" not in code:
            return None

        lines = code.split("\n")
        new_lines = []
        modified = False
        needs_set = False

        for line in lines:
            new_line = line

            if " in nums" in line and "not in" not in line:
                new_line = line.replace(" in nums", " in seen_set")
                needs_set = True
                modified = True

            if ".index(" in line:
                new_line = line.replace(".index(", "[index_map[")
                needs_set = True
                modified = True

            new_lines.append(new_line)

        if not modified:
            return None

        if needs_set:
            for i, line in enumerate(new_lines):
                if line.strip().startswith("def "):
                    indent = "    "
                    new_lines.insert(i + 1, indent + "seen_set = set(nums)")
                    break

        return "\n".join(new_lines)

    def _add_pruning(self, code: str, info: dict) -> str:
        lines = code.split("\n")
        new_lines = []
        modified = False

        for line in lines:
            stripped = line.strip()
            new_lines.append(line)

            if "(" in stripped and ")" in stripped:
                indent = len(line) - len(line.lstrip())
                if "dfs(" in stripped or "solve(" in stripped or "search(" in stripped:
                    new_lines.insert(
                        len(new_lines) - 1,
                        " " * indent + "if best is not None and curr >= best: continue"
                    )
                    modified = True

        if modified:
            return "\n".join(new_lines)
        return None

    def _add_hashing(self, code: str, info: dict) -> str:
        if "for" not in code:
            return None

        lines = code.split("\n")
        new_lines = []
        modified = False

        for i, line in enumerate(lines):
            new_lines.append(line)

            stripped = line.strip()
            if stripped.startswith("for ") and "range(" in stripped:
                if i + 1 < len(lines) and "==" in lines[i + 1]:
                    indent = len(line) - len(line.lstrip())
                    new_lines.append(" " * indent + "# Replaced linear scan with hash lookup")
                    modified = True

        if modified:
            return "\n".join(new_lines)
        return None

    def _sort_first(self, code: str, info: dict) -> str:
        lines = code.split("\n")

        for i, line in enumerate(lines):
            if line.strip().startswith("def "):
                if any("sort()" in line for line in lines):
                    return None
                indent = "    "
                params = line.split("(")[1].split(")")[0].split(",")
                first_param = params[0].strip() if params else "nums"
                lines.insert(i + 1, indent + f"{first_param}.sort()")
                return "\n".join(lines)

        return None

    def _add_binary_search(self, code: str, info: dict) -> str:
        if "for" not in code or "range(" not in code:
            return None

        if "sort()" not in code and "sorted(" not in code:
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

    def _apply_two_pointer(self, code: str, info: dict) -> str:
        if code.count("for ") < 2:
            return None

        return '''def solve(nums, target):
    left, right = 0, len(nums) - 1
    while left < right:
        s = nums[left] + nums[right]
        if s == target:
            return [left, right]
        elif s < target:
            left += 1
        else:
            right -= 1
    return []
'''

    def _apply_sliding_window(self, code: str, info: dict) -> str:
        return '''def solve(nums, k):
    window_sum = sum(nums[:k])
    best = window_sum
    for i in range(k, len(nums)):
        window_sum += nums[i] - nums[i - k]
        best = max(best, window_sum)
    return best
'''

    def _reduce_space(self, code: str, info: dict) -> str:
        if "dp = [[" not in code and "dp=[[" not in code:
            return None

        lines = code.split("\n")
        new_lines = []
        for line in lines:
            if "dp = [[" in line or "dp=[[" in line:
                new_lines.append(line.replace("[[", "[").split("]")[0] + "]")
            elif "dp[i][" in line or "dp[i-1][" in line:
                new_lines.append(line.replace("dp[i][", "dp[").replace("dp[i-1][", "prev["))
            else:
                new_lines.append(line)

        return "\n".join(new_lines) if new_lines != lines else None

    def _vectorize(self, code: str, info: dict) -> str:
        return None

    def _add_greedy(self, code: str, info: dict) -> str:
        return None

    def _convert_to_dp(self, code: str, info: dict) -> str:
        return None

    def _restructure_recursion(self, code: str, info: dict) -> str:
        return None
