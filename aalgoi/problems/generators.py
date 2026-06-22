from __future__ import annotations

import random
from typing import Any

from aalgoi.types import ProblemTask


def generate_example(task: ProblemTask, rng: random.Random | None = None) -> tuple[dict[str, Any], Any]:
    if rng is None:
        rng = random
    if task == ProblemTask.SORT:
        data = [rng.randint(-100, 100) for _ in range(rng.randint(3, 10))]
        return {"data": data}, sorted(data)
    elif task == ProblemTask.COUNTING_SORT:
        data = [rng.randint(0, 20) for _ in range(rng.randint(3, 10))]
        return {"data": data}, sorted(data)
    elif task == ProblemTask.LINEAR_SEARCH:
        data = rng.sample(range(-50, 50), rng.randint(3, 8))
        target = rng.choice(data) if rng.random() < 0.7 else 999
        try:
            idx = data.index(target)
        except ValueError:
            idx = -1
        return {"data": data, "target": target}, idx
    elif task == ProblemTask.BINARY_SEARCH:
        data = sorted(rng.sample(range(-50, 50), rng.randint(3, 8)))
        target = rng.choice(data) if rng.random() < 0.7 else 999
        try:
            idx = data.index(target)
        except ValueError:
            idx = -1
        return {"data": data, "target": target}, idx
    elif task == ProblemTask.TWO_SUM:
        n = rng.randint(4, 8)
        data = rng.sample(range(-20, 20), n)
        target = data[0] + data[1]
        return {"data": data, "target": target}, [0, 1]
    elif task == ProblemTask.LOWER_BOUND:
        data = sorted(rng.sample(range(-50, 50), rng.randint(3, 8)))
        target = rng.choice(data) if rng.random() < 0.7 else data[-1] + 10
        expected = next((i for i, v in enumerate(data) if v >= target), len(data))
        return {"data": data, "target": target}, expected
    elif task == ProblemTask.GCD:
        a, b = rng.randint(1, 100), rng.randint(1, 100)
        import math
        return {"a": a, "b": b}, math.gcd(a, b)
    elif task == ProblemTask.LCM:
        a, b = rng.randint(1, 50), rng.randint(1, 50)
        import math
        return {"a": a, "b": b}, a // math.gcd(a, b) * b
    elif task == ProblemTask.IS_PRIME:
        n = rng.choice([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 4, 6, 8, 9, 10, 12, 14, 15])
        return {"n": n}, n > 1 and all(n % i != 0 for i in range(2, int(n ** 0.5) + 1))
    elif task == ProblemTask.SIEVE:
        n = rng.randint(10, 50)
        sieve_arr = [True] * (n + 1)
        sieve_arr[0] = sieve_arr[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if sieve_arr[i]:
                for j in range(i * i, n + 1, i):
                    sieve_arr[j] = False
        return {"n": n}, [i for i, v in enumerate(sieve_arr) if v]
    elif task == ProblemTask.FAST_EXPONENTIATION:
        base_val = rng.randint(2, 10)
        exp = rng.randint(0, 10)
        return {"base": base_val, "exp": exp}, base_val ** exp
    elif task == ProblemTask.FIBONACCI:
        n = rng.randint(0, 30)
        def fib(k: int) -> int:
            if k < 2:
                return k
            a, b = 0, 1
            for _ in range(k - 1):
                a, b = b, a + b
            return b
        return {"n": n}, fib(n)
    elif task == ProblemTask.PALINDROME:
        s = "".join(rng.choice("abc") for _ in range(rng.randint(2, 5)))
        s2 = s + s[::-1] if rng.random() < 0.5 else s + "x"
        return {"s": s2}, s2 == s2[::-1]
    elif task == ProblemTask.ANAGRAM:
        base = "".join(rng.choice("abc") for _ in range(rng.randint(2, 4)))
        if rng.random() < 0.5:
            other = "".join(rng.sample(base, len(base)))
        else:
            other = "".join(rng.choice("abc") for _ in range(len(base)))
        return {"s1": base, "s2": other}, sorted(base) == sorted(other)
    elif task == ProblemTask.KMP:
        text = "abcabcabc" * rng.randint(1, 3)
        pattern = "abc" * rng.randint(1, 2)
        idx = text.find(pattern)
        return {"text": text, "pattern": pattern}, idx
    elif task == ProblemTask.RABIN_KARP:
        text = "xyzxyzxyz" * rng.randint(1, 3)
        pattern = "xyz" * rng.randint(1, 2)
        idx = text.find(pattern)
        return {"text": text, "pattern": pattern}, idx
    elif task == ProblemTask.EDIT_DISTANCE:
        s1 = "".join(rng.choice("abc") for _ in range(rng.randint(2, 4)))
        s2 = "".join(rng.choice("abc") for _ in range(rng.randint(2, 4)))
        n, m = len(s1), len(s2)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        return {"s1": s1, "s2": s2}, dp[n][m]
    elif task == ProblemTask.LCS:
        s1 = "".join(rng.choice("abc") for _ in range(rng.randint(2, 5)))
        s2 = "".join(rng.choice("abc") for _ in range(rng.randint(2, 5)))
        n, m = len(s1), len(s2)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return {"s1": s1, "s2": s2}, dp[n][m]
    elif task == ProblemTask.BFS:
        n_nodes = rng.randint(4, 7)
        adj = {i: [] for i in range(n_nodes)}
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if rng.random() < 0.4:
                    adj[i].append(j)
                    adj[j].append(i)
        start = 0
        visited = []
        queue = [start]
        seen = {start}
        while queue:
            v = queue.pop(0)
            visited.append(v)
            for nb in sorted(adj[v]):
                if nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
        return {"graph": adj, "start": start}, visited
    elif task == ProblemTask.DFS:
        n_nodes = rng.randint(4, 7)
        adj = {i: [] for i in range(n_nodes)}
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if rng.random() < 0.4:
                    adj[i].append(j)
                    adj[j].append(i)
        start = 0
        visited = []
        stack = [start]
        seen = {start}
        while stack:
            v = stack.pop()
            visited.append(v)
            for nb in sorted(adj[v], reverse=True):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        return {"graph": adj, "start": start}, visited
    elif task == ProblemTask.SHORTEST_PATH_UNWEIGHTED:
        n_nodes = rng.randint(4, 6)
        adj = {i: [] for i in range(n_nodes)}
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if rng.random() < 0.4:
                    adj[i].append(j)
                    adj[j].append(i)
        start, end = 0, n_nodes - 1
        dist = {start: 0}
        q = [start]
        while q:
            v = q.pop(0)
            for nb in adj.get(v, []):
                if nb not in dist:
                    dist[nb] = dist[v] + 1
                    q.append(nb)
        return {"graph": adj, "start": start, "end": end}, dist.get(end, -1)
    elif task == ProblemTask.SHORTEST_PATH_WEIGHTED:
        n_nodes = rng.randint(4, 6)
        adj = {i: {} for i in range(n_nodes)}
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if rng.random() < 0.4:
                    w = rng.randint(1, 10)
                    adj[i][j] = w
                    adj[j][i] = w
        start, end = 0, n_nodes - 1
        import heapq
        pq = [(0, start)]
        dist = {start: 0}
        while pq:
            d, v = heapq.heappop(pq)
            if d > dist.get(v, float('inf')):
                continue
            for nb, w in adj.get(v, {}).items():
                nd = d + w
                if nd < dist.get(nb, float('inf')):
                    dist[nb] = nd
                    heapq.heappush(pq, (nd, nb))
        return {"graph": adj, "start": start, "end": end}, dist.get(end, -1)
    elif task == ProblemTask.SHORTEST_PATH_NEGATIVE:
        n_nodes = rng.randint(4, 5)
        adj = {i: {} for i in range(n_nodes)}
        edges = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if rng.random() < 0.4:
                    w = rng.randint(-5, 10)
                    if w != 0:
                        adj[i][j] = w
                        edges.append((i, j, w))
        start, end = 0, n_nodes - 1
        dist = {i: float('inf') for i in range(n_nodes)}
        dist[start] = 0
        for _ in range(n_nodes - 1):
            for u, v, w in edges:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
        return {"graph": adj, "start": start, "end": end}, dist.get(end, -1) if dist.get(end) != float('inf') else -1
    elif task == ProblemTask.TOPOLOGICAL_SORT:
        n_nodes = rng.randint(4, 6)
        nodes = list(range(n_nodes))
        edges = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if rng.random() < 0.5:
                    edges.append((i, j))
        adj = {i: [] for i in range(n_nodes)}
        for u, v in edges:
            adj[u].append(v)
        visited_set = set()
        temp = set()
        order = []
        def dfs(v: int) -> bool:
            if v in temp:
                return False
            if v in visited_set:
                return True
            temp.add(v)
            for nb in adj.get(v, []):
                if not dfs(nb):
                    return False
            temp.remove(v)
            visited_set.add(v)
            order.append(v)
            return True
        for v in nodes:
            dfs(v)
        return {"graph": adj}, order[::-1]
    elif task == ProblemTask.CYCLE_DETECTION:
        n_nodes = rng.randint(4, 6)
        adj = {i: [] for i in range(n_nodes)}
        if rng.random() < 0.5:
            for i in range(n_nodes - 1):
                adj[i].append(i + 1)
            adj[n_nodes - 1].append(0)
        else:
            for i in range(n_nodes - 1):
                adj[i].append(i + 1)
        has_cycle = False
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {i: WHITE for i in range(n_nodes)}
        def dfs_cycle(v: int) -> bool:
            color[v] = GRAY
            for nb in adj.get(v, []):
                if color[nb] == GRAY:
                    return True
                if color[nb] == WHITE and dfs_cycle(nb):
                    return True
            color[v] = BLACK
            return False
        for v in range(n_nodes):
            if color[v] == WHITE and dfs_cycle(v):
                has_cycle = True
                break
        return {"graph": adj}, has_cycle
    elif task == ProblemTask.CONNECTED_COMPONENTS:
        n_nodes = rng.randint(4, 7)
        adj = {i: [] for i in range(n_nodes)}
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if rng.random() < 0.3:
                    adj[i].append(j)
                    adj[j].append(i)
        comps = []
        unvisited = set(range(n_nodes))
        while unvisited:
            start_v = unvisited.pop()
            stack_cc = [start_v]
            comp = []
            while stack_cc:
                v = stack_cc.pop()
                comp.append(v)
                for nb in adj.get(v, []):
                    if nb in unvisited:
                        unvisited.remove(nb)
                        stack_cc.append(nb)
            comps.append(sorted(comp))
        return {"graph": adj}, comps
    elif task == ProblemTask.MST:
        n_nodes = rng.randint(4, 6)
        edges = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if rng.random() < 0.5:
                    w = rng.randint(1, 10)
                    edges.append((w, i, j))
        edges.sort()
        parent_uf = list(range(n_nodes))
        def find_uf(x: int) -> int:
            while parent_uf[x] != x:
                parent_uf[x] = parent_uf[parent_uf[x]]
                x = parent_uf[x]
            return x
        def union_uf(x: int, y: int) -> None:
            rx, ry = find_uf(x), find_uf(y)
            if rx != ry:
                parent_uf[rx] = ry
        mst_weight = 0
        for w, u, v in edges:
            if find_uf(u) != find_uf(v):
                union_uf(u, v)
                mst_weight += w
        return {"graph": edges}, mst_weight
    elif task == ProblemTask.MAX_FLOW:
        n_nodes = 4
        edges_list = [
            (0, 1, 10),
            (0, 2, 5),
            (1, 2, 3),
            (1, 3, 8),
            (2, 3, 6),
        ]
        source, sink = 0, 3
        capacity = {i: {} for i in range(n_nodes)}
        for u, v, c in edges_list:
            capacity[u][v] = c
        flow = 0
        residual = {i: {j: c for j, c in cap.items()} for i, cap in capacity.items()}
        for i in range(n_nodes):
            for j in range(n_nodes):
                if j not in residual.get(i, {}):
                    if i not in residual:
                        residual[i] = {}
                    residual[i][j] = 0
        from collections import deque
        while True:
            parent = {i: -1 for i in range(n_nodes)}
            parent[source] = source
            q = deque([source])
            while q:
                u = q.popleft()
                for v in range(n_nodes):
                    if parent[v] == -1 and residual.get(u, {}).get(v, 0) > 0:
                        parent[v] = u
                        q.append(v)
            if parent[sink] == -1:
                break
            add_flow = float('inf')
            v = sink
            while v != source:
                u = parent[v]
                add_flow = min(add_flow, residual.get(u, {}).get(v, 0))
                v = u
            v = sink
            while v != source:
                u = parent[v]
                residual[u][v] -= add_flow
                residual[v][u] = residual[v].get(u, 0) + add_flow
                v = u
            flow += add_flow
        return {"graph": edges_list, "source": source, "sink": sink}, flow
    elif task == ProblemTask.KADANE:
        data = [rng.randint(-10, 10) for _ in range(rng.randint(3, 8))]
        best = cur = data[0]
        for x in data[1:]:
            cur = max(x, cur + x)
            best = max(best, cur)
        return {"data": data}, best
    elif task == ProblemTask.KNAPSACK_01:
        n = rng.randint(3, 6)
        items = [{"weight": rng.randint(1, 10), "value": rng.randint(1, 20)} for _ in range(n)]
        capacity = rng.randint(5, 20)
        return {"items": items, "capacity": capacity}, None
    elif task == ProblemTask.KNAPSACK_FRACTIONAL:
        n = rng.randint(3, 5)
        items = [{"weight": rng.randint(1, 10), "value": rng.randint(1, 20)} for _ in range(n)]
        capacity = rng.randint(5, 20)
        return {"items": items, "capacity": capacity}, None
    elif task == ProblemTask.COIN_CHANGE:
        coins = rng.sample([1, 2, 5, 10, 25], rng.randint(2, 4))
        amount = rng.randint(3, 15)
        INF = 10 ** 9
        dp = [INF] * (amount + 1)
        dp[0] = 0
        for i in range(1, amount + 1):
            for c in coins:
                if i >= c:
                    dp[i] = min(dp[i], dp[i - c] + 1)
        expected = dp[amount] if dp[amount] != INF else -1
        return {"coins": coins, "amount": amount}, expected
    elif task == ProblemTask.LIS:
        n = rng.randint(4, 8)
        data = [rng.randint(-10, 20) for _ in range(n)]
        import bisect
        tails = []
        for x in data:
            i = bisect.bisect_left(tails, x)
            if i == len(tails):
                tails.append(x)
            else:
                tails[i] = x
        return {"data": data}, len(tails)
    elif task == ProblemTask.CLASSIFICATION:
        return {"X_train": [[0], [1], [10], [11]], "y_train": [0, 0, 1, 1], "X_test": [[0.2], [10.5]]}, [0, 1]
    elif task == ProblemTask.REGRESSION:
        return {"X_train": [[0], [1], [2]], "y_train": [0, 2, 4], "X_test": [[3], [4]]}, [6, 8]
    elif task == ProblemTask.CLUSTERING:
        return {"data": [[0, 0], [0, 1], [9, 9], [9, 8]], "n_clusters": 2}, None
    elif task == ProblemTask.DIMENSIONALITY_REDUCTION:
        return {"data": [[1, 2, 3], [2, 3, 4], [5, 6, 7]], "n_components": 2}, None
    elif task == ProblemTask.ANOMALY_DETECTION:
        return {"data": [[0, 0], [0, 1], [1, 0], [20, 20]], "X_test": [[0, 0], [20, 20]]}, None
    elif task == ProblemTask.SENTIMENT_ANALYSIS:
        return {"text": "I love this product"}, None
    elif task == ProblemTask.TEXT_SUMMARIZATION:
        return {"text": "short text"}, "short text"
    elif task == ProblemTask.IMAGE_BLUR:
        return {"image": [[0, 1], [1, 0]], "sigma": 1.0}, None
    elif task == ProblemTask.EDGE_DETECTION:
        return {"image": [[0, 1], [1, 0]], "edge": True}, None
    else:
        return {"data": [1, 2, 3]}, [1, 2, 3]
