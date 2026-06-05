import ast
import copy


class ASTOptimizer:
    """
    Three optimization passes on Python source:
    1. `@lru_cache` injection  — pure functions with >= 1 arg
    2. Listcomp conversion     — for loops building lists
    3. Loop fusion             — adjacent loops on same range
    """

    def optimize(self, source: str) -> str:
        if not source.strip():
            return source
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source

        modified = False

        new_tree = self._inject_lru_cache(tree)
        if new_tree is not None:
            tree = new_tree
            modified = True

        new_tree = self._convert_listcomps(tree)
        if new_tree is not None:
            tree = new_tree
            modified = True

        new_tree = self._fuse_loops(tree)
        if new_tree is not None:
            tree = new_tree
            modified = True

        if modified:
            ast.fix_missing_locations(tree)
            return ast.unparse(tree)
        return source

    # ── Pass 1: @lru_cache injection ─────────────────────────────────

    def _inject_lru_cache(self, tree: ast.Module) -> ast.Module | None:
        """Add @lru_cache to pure functions with >= 1 arg and no decorators."""
        tree = copy.deepcopy(tree)
        changed = False
        import_needed = False

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if self._is_pure_candidate(node):
                    # Check for existing decorators
                    names = [self._decorator_name(d) for d in node.decorator_list]
                    if "lru_cache" not in names:
                        decorator = ast.Name(id="lru_cache", ctx=ast.Load())
                        node.decorator_list.append(decorator)
                        changed = True
                        import_needed = True

        if not changed:
            return None

        if import_needed:
            self._ensure_functools_import(tree)

        return tree

    def _is_pure_candidate(self, node: ast.FunctionDef) -> bool:
        """Heuristic: no yields, no async, at least 1 arg, no self/cls."""
        if not node.args.args:
            return False
        # Skip methods and LLM entry points
        if node.args.args and node.args.args[0].arg in ("self", "cls"):
            return False
        if node.name == "process":
            return False
        if any(
            isinstance(n, (ast.Yield, ast.YieldFrom, ast.Await, ast.AsyncFor))
            for n in ast.walk(node)
        ):
            return False
        # Check for dangerous calls (IO, print, etc.)
        for n in ast.walk(node):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                if n.func.id in ("print", "input", "open", "exec", "eval"):
                    return False
        return True

    def _decorator_name(self, d: ast.expr) -> str:
        if isinstance(d, ast.Name):
            return d.id
        if isinstance(d, ast.Attribute):
            return d.attr
        if isinstance(d, ast.Call) and isinstance(d.func, ast.Name):
            return d.func.id
        return ""

    def _ensure_functools_import(self, tree: ast.Module):
        """Add 'from functools import lru_cache' if not present."""
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "functools":
                for alias in node.names:
                    if alias.name == "lru_cache":
                        return  # Already imported
                node.names.append(ast.alias(name="lru_cache"))
                return

        imp = ast.ImportFrom(
            module="functools",
            names=[ast.alias(name="lru_cache")],
            level=0,
        )
        tree.body.insert(0, imp)

    # ── Pass 2: Listcomp conversion ──────────────────────────────────

    def _convert_listcomps(self, tree: ast.Module) -> ast.Module | None:
        """
        Convert this pattern:
            result = []
            for x in data:
                result.append(f(x))
        into:
            result = [f(x) for x in data]
        """
        tree = copy.deepcopy(tree)
        changed = False

        for node in list(ast.walk(tree)):
            if not isinstance(node, (ast.FunctionDef, ast.Module)):
                continue

            body = node.body
            new_body = []
            i = 0
            while i < len(body):
                stmt = body[i]
                # Look for: result = []   (empty list init)
                if self._is_empty_list_init(stmt):
                    result_var = stmt.targets[0].id  # type: ignore
                    # Look ahead for: for x in data: result.append(...)
                    if (i + 1 < len(body)
                            and self._is_simple_for_append(body[i + 1], result_var)):
                        for_node = body[i + 1]
                        target = for_node.target  # type: ignore
                        iter_expr = for_node.iter  # type: ignore
                        call = for_node.body[0].value  # type: ignore

                        listcomp = ast.ListComp(
                            elt=call.args[0],
                            generators=[
                                ast.comprehension(
                                    target=target,
                                    iter=iter_expr,
                                    ifs=[],
                                    is_async=0,
                                )
                            ],
                        )
                        assign = ast.Assign(
                            targets=[ast.Name(id=result_var, ctx=ast.Store())],
                            value=listcomp,
                        )
                        new_body.append(assign)
                        i += 2
                        changed = True
                        continue
                new_body.append(stmt)
                i += 1

            if changed:
                node.body = new_body

        return tree if changed else None

    def _is_empty_list_init(self, node: ast.stmt) -> bool:
        return (isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.List)
                and len(node.value.elts) == 0)

    def _is_simple_for_append(
        self, node: ast.stmt, var: str
    ) -> bool:
        if not isinstance(node, ast.For):
            return False
        if len(node.body) != 1:
            return False
        body_stmt = node.body[0]
        if not isinstance(body_stmt, ast.Expr):
            return False
        call = body_stmt.value
        if not isinstance(call, ast.Call):
            return False
        func = call.func
        if not (isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == var
                and func.attr == "append"
                and len(call.args) == 1):
            return False
        return True

    # ── Pass 3: Loop fusion ──────────────────────────────────────────

    def _fuse_loops(self, tree: ast.Module) -> ast.Module | None:
        """
        Fuse adjacent for-loops over the same iterable:
            for x in data:  A(x)
            for x in data:  B(x)
        =>
            for x in data:  A(x); B(x)
        """
        tree = copy.deepcopy(tree)
        changed = False

        for node in list(ast.walk(tree)):
            if not isinstance(node, (ast.FunctionDef, ast.Module)):
                continue

            body = node.body
            new_body = []
            i = 0
            while i < len(body):
                current = body[i]
                if (isinstance(current, ast.For)
                        and self._is_simple_for(current)):
                    fused = False
                    # Peek at next statement
                    if (i + 1 < len(body)
                            and isinstance(body[i + 1], ast.For)
                            and self._same_iterable(current, body[i + 1])):
                        # Fuse: merge body of second into first
                        for stmt in body[i + 1].body:
                            current.body.append(stmt)
                        i += 2
                        changed = True
                        fused = True
                    new_body.append(current)
                else:
                    new_body.append(current)
                    i += 1
                    continue

                if not fused:
                    i += 1

            if changed:
                node.body = new_body

        return tree if changed else None

    def _is_simple_for(self, node: ast.For) -> bool:
        return not node.orelse  # No else

    def _same_iterable(self, a: ast.For, b: ast.For) -> bool:
        return (ast.dump(a.target) == ast.dump(b.target)
                and ast.dump(a.iter) == ast.dump(b.iter))
