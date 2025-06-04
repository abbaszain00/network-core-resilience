import networkx as nx
from typing import List, Tuple, Set, Dict

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

def apply_complete_conversion(G: nx.Graph, component: Set[int], collapse_nodes: Set[int], k: int) -> List[Tuple[int, int]]:
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

def compute_onion_layers(G: nx.Graph, k: int) -> Dict[int, int]:
    core = nx.core_number(G)
    k_minus_1_core_nodes = {u for u in G.nodes if core[u] >= k - 1}

    if not k_minus_1_core_nodes:
        return {}

    subgraph = G.subgraph(k_minus_1_core_nodes).copy()
    layers = {}
    current_layer = 0

    while subgraph.number_of_nodes() > 0:
        layer_nodes = [u for u in subgraph.nodes if subgraph.degree(u) < k - 1]

        if not layer_nodes:
            for u in subgraph.nodes:
                layers[u] = current_layer
            break

        for u in layer_nodes:
            layers[u] = current_layer
            subgraph.remove_node(u)

        current_layer += 1

    return layers

def calculate_requisite_degree(u, j, layers, G, component_nodes, k):
    core = nx.core_number(G)
    k_core = {v for v in G.nodes if core[v] >= k}
    higher = {v for v in component_nodes if layers.get(v, float('inf')) >= j}
    H = k_core | higher
    return max(0, k - sum(1 for v in G.neighbors(u) if v in H))

def calculate_anchor_gain(u, j, layers, G, component_nodes, k):
    benefit = 0
    layer_j_nodes = {v for v in component_nodes if layers.get(v, -1) == j}

    for v in G.neighbors(u):
        if v in layer_j_nodes and calculate_requisite_degree(v, j, layers, G, component_nodes, k) > 0:
            benefit += 1

    cost = calculate_requisite_degree(u, j, layers, G, component_nodes, k)
    return benefit - cost

def apply_partial_conversion(G: nx.Graph, component_nodes: Set[int], k: int, budget: int, layers: Dict[int, int]) -> Tuple[List[Tuple[int, int]], Set[int]]:
    if not component_nodes or budget <= 0:
        return [], set()

    max_layer = max(layers.get(u, 0) for u in component_nodes)

    for j in range(1, max_layer + 1):
        try:
            edges, anchors = [], set()
            upper = {u for u in component_nodes if layers.get(u, -1) >= j}
            lower = {u for u in component_nodes if layers.get(u, -1) < j}
            L_j = {u for u in component_nodes if layers.get(u, -1) == j}

            reqs = {u: calculate_requisite_degree(u, j, layers, G, component_nodes, k) for u in L_j}
            total = sum(reqs.values())

            while True:
                best = None
                best_gain = -1
                for u in lower:
                    if u in anchors:
                        continue
                    gain = calculate_anchor_gain(u, j, layers, G, component_nodes, k)
                    if gain > best_gain:
                        best, best_gain = u, gain
                if best is None or best_gain < 0:
                    break
                anchors.add(best)
                lower.remove(best)
                for v in G.neighbors(best):
                    if v in reqs and reqs[v] > 0:
                        reqs[v] -= 1
                        total -= 1

            if total > 2 * budget:
                continue

            core = nx.core_number(G)
            k_core = {u for u in G.nodes if core[u] >= k}
            H = anchors | upper | k_core

            needs = [u for u in H if u in component_nodes and sum(1 for v in G.neighbors(u) if v in H) < k]
            q = needs[:]

            while len(q) >= 2 and len(edges) < budget:
                u, v = q[0], q[1]
                if not G.has_edge(u, v):
                    edges.append((u, v))
                    du = sum(1 for x in G.neighbors(u) if x in H) + sum(1 for a, b in edges if a == u or b == u)
                    dv = sum(1 for x in G.neighbors(v) if x in H) + sum(1 for a, b in edges if a == v or b == v)
                    if du >= k:
                        q.remove(u)
                    if dv >= k and v in q:
                        q.remove(v)
                    continue
                q.pop(0)

            for u in q:
                if len(edges) >= budget:
                    break
                targets = [v for v in H if v != u and not G.has_edge(u, v)]
                if targets:
                    edges.append((u, targets[0]))

            if len(edges) <= budget:
                return edges, anchors | upper

        except Exception as e:
            continue

    return [], set()

def fastcm_plus_reinforce(G: nx.Graph, k: int = None, budget: int = 10) -> Tuple[nx.Graph, List[Tuple[int, int]]]:
    if G.number_of_nodes() == 0:
        return G.copy(), []

    if k is None:
        core_vals = nx.core_number(G)
        if not core_vals:
            return G.copy(), []
        k = max(core_vals.values()) + 1

    core = nx.core_number(G)
    shell = {u for u, c in core.items() if c == k - 1}
    if not shell:
        return G.copy(), []

    components = get_shell_components_with_collapse_nodes(G, k)
    if not components:
        return G.copy(), []

    onion_layers = compute_onion_layers(G, k)

    options = []
    for comp, collapse in components:
        try:
            cost = estimate_complete_conversion_cost(G, comp, collapse, k)
            if cost <= budget:
                options.append({'type': 'complete', 'component': comp, 'collapse': collapse, 'cost': cost, 'followers': set(comp)})
        except Exception:
            pass
        try:
            edges, followers = apply_partial_conversion(G, comp, k, budget, onion_layers)
            if edges and len(edges) <= budget:
                options.append({'type': 'partial', 'component': comp, 'collapse': set(), 'cost': len(edges), 'followers': followers, 'edges': edges})
        except Exception:
            pass

    dp_table = [set() for _ in range(budget + 1)]
    selection = [None for _ in range(budget + 1)]

    for opt in options:
        cost = opt['cost']
        followers = opt['followers']
        for b in range(budget, cost - 1, -1):
            combined = dp_table[b - cost] | followers
            if len(combined) > len(dp_table[b]):
                dp_table[b] = combined
                if selection[b - cost] is None:
                    selection[b] = [opt]
                else:
                    selection[b] = selection[b - cost] + [opt]

    best_b = max(range(budget + 1), key=lambda b: len(dp_table[b]))
    selected_options = selection[best_b] if selection[best_b] else []

    G_new = G.copy()
    added_edges = []
    for opt in selected_options:
        if opt['type'] == 'complete':
            try:
                edges = apply_complete_conversion(G_new, opt['component'], opt['collapse'], k)
                for u, v in edges:
                    if len(added_edges) >= budget:
                        break
                    if not G_new.has_edge(u, v):
                        G_new.add_edge(u, v)
                        added_edges.append((u, v))
            except Exception:
                continue
        elif opt['type'] == 'partial':
            for u, v in opt['edges']:
                if len(added_edges) >= budget:
                    break
                if not G_new.has_edge(u, v):
                    G_new.add_edge(u, v)
                    added_edges.append((u, v))

    # Use leftover budget greedily
    remaining = budget - len(added_edges)
    if remaining > 0:
        core = nx.core_number(G_new)
        k_core = {u for u in G_new.nodes if core[u] >= k}
        candidates = [u for u in G_new.nodes if core[u] == k - 1]
        for u in candidates:
            if remaining <= 0:
                break
            for v in k_core:
                if not G_new.has_edge(u, v):
                    G_new.add_edge(u, v)
                    added_edges.append((u, v))
                    remaining -= 1
                    break

    return G_new, added_edges
