import networkx as nx
from typing import List, Tuple, Set

def get_shell_components_with_collapse_nodes(G: nx.Graph, k: int) -> List[Tuple[Set[int], Set[int]]]:
    core = nx.core_number(G)
    shell_nodes = {u for u in G.nodes if core[u] == k - 1}

    if not shell_nodes:
        return []

    k_minus_1_core = {u for u in G.nodes if core[u] >= k - 1}
    components = list(nx.connected_components(G.subgraph(shell_nodes)))
    results = []

    for comp in components:
        collapse_nodes = set()
        for u in comp:
            neighbors = [v for v in G.neighbors(u) if v in k_minus_1_core]
            if len(neighbors) == k - 1:
                collapse_nodes.add(u)
        results.append((comp, collapse_nodes))

    return results


def estimate_complete_conversion_cost(G: nx.Graph, component: Set[int], collapse_nodes: Set[int], k: int) -> int:
    core = nx.core_number(G)
    k_core = {u for u in G.nodes if core[u] >= k}
    n = len(collapse_nodes)

    if n == 0:
        return 0

    cost = n // 2
    if n % 2 == 1:
        if component | k_core:
            cost += 1
        else:
            raise RuntimeError("No valid target for the remaining collapse node")

    return cost


def select_components_dp_under_budget(component_info: List[Tuple[Set[int], Set[int], int]], budget: int) -> List[Tuple[Set[int], Set[int]]]:
    n = len(component_info)
    dp = [[0] * (budget + 1) for _ in range(n + 1)]

    values = [len(c) for c, _, _ in component_info]
    costs = [cost for _, _, cost in component_info]

    for i in range(1, n + 1):
        for w in range(budget + 1):
            if costs[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - costs[i - 1]] + values[i - 1])
            else:
                dp[i][w] = dp[i - 1][w]

    selected = []
    w = budget
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected.append((component_info[i - 1][0], component_info[i - 1][1]))
            w -= costs[i - 1]

    return selected


def apply_complete_conversion(G: nx.Graph, component: Set[int], collapse_nodes: Set[int], k: int) -> List[Tuple[int, int]]:
    if not collapse_nodes:
        return []

    added_edges = []
    core = nx.core_number(G)
    k_core = {u for u in G.nodes if core[u] >= k}
    used = set()
    nodes = list(collapse_nodes)

    for i in range(len(nodes)):
        if nodes[i] in used:
            continue
        u = nodes[i]
        for j in range(i + 1, len(nodes)):
            v = nodes[j]
            if v not in used and not G.has_edge(u, v):
                added_edges.append((u, v))
                used.update([u, v])
                break

    leftovers = [u for u in nodes if u not in used]
    for u in leftovers:
        targets = sorted(v for v in (component | k_core) - {u} if not G.has_edge(u, v))
        if targets:
            added_edges.append((u, targets[0]))
        else:
            raise RuntimeError(f"No valid target for node {u}")

    return added_edges


def apply_component_conversions(G: nx.Graph, components: List[Tuple[Set[int], Set[int]]], k: int) -> Tuple[nx.Graph, List[Tuple[int, int]]]:
    G_reinforced = G.copy()
    added_edges = []

    for comp, collapse in components:
        try:
            new_edges = apply_complete_conversion(G_reinforced, comp, collapse, k)
            for u, v in new_edges:
                if not G_reinforced.has_edge(u, v):
                    G_reinforced.add_edge(u, v)
                    added_edges.append((u, v))
        except RuntimeError as e:
            print(f"Skipping component: {e}")
            continue

    return G_reinforced, added_edges


def fastcm_plus_reinforce(G: nx.Graph, k: int = None, budget: int = 10) -> Tuple[nx.Graph, List[Tuple[int, int]]]:
    if k is None:
        core_vals = nx.core_number(G)
        if not core_vals:
            return G.copy(), []
        k = max(core_vals.values()) + 1

    core = nx.core_number(G)
    shell = {u for u, c in core.items() if c == k - 1}
    if not shell:
        return G.copy(), []

    try:
        components = get_shell_components_with_collapse_nodes(G, k)
    except Exception:
        return G.copy(), []

    if not components:
        return G.copy(), []

    component_info = []
    for comp, collapse in components:
        try:
            cost = estimate_complete_conversion_cost(G, comp, collapse, k)
            component_info.append((comp, collapse, cost))
        except RuntimeError:
            continue

    if not component_info:
        return G.copy(), []

    try:
        selected = select_components_dp_under_budget(component_info, budget)
    except Exception:
        return G.copy(), []

    if not selected:
        return G.copy(), []

    try:
        return apply_component_conversions(G, selected, k)
    except Exception:
        return G.copy(), []


def fastcm_reinforce(G: nx.Graph, budget: int = 10) -> nx.Graph:
    G_reinforced, _ = fastcm_plus_reinforce(G, budget=budget)
    return G_reinforced


def fastcm_plus(G: nx.Graph, k: int, budget: int) -> Tuple[nx.Graph, List[Tuple[int, int]]]:
    return fastcm_plus_reinforce(G, k=k, budget=budget)
