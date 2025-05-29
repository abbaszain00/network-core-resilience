import networkx as nx
from typing import List, Tuple, Set

def get_shell_components_with_collapse_nodes(G: nx.Graph, k: int) -> List[Tuple[Set[int], Set[int]]]:
    """
    Identifies connected components in the (k-1)-shell and the collapse nodes in each,
    using the definition from the FastCM+ paper.
    """
    core = nx.core_number(G)
    shell_nodes = [u for u in G.nodes if core[u] == k - 1]
    shell_subgraph = G.subgraph(shell_nodes)
    components = list(nx.connected_components(shell_subgraph))

    result = []
    for comp in components:
        collapse_nodes = set()
        for u in comp:
            neighbors_in_k_minus_1_core = [v for v in G.neighbors(u) if core[v] >= k - 1]
            if len(neighbors_in_k_minus_1_core) == k - 1:
                collapse_nodes.add(u)
        result.append((comp, collapse_nodes))

    return result


def estimate_complete_conversion_cost(
    G: nx.Graph,
    component_nodes: Set[int],
    collapse_nodes: Set[int],
    k: int
) -> int:
    """
    Estimates the number of edges required to completely convert all collapse nodes
    in a component into the k-core using the FastCM+ complete conversion strategy.
    """
    core = nx.core_number(G)
    k_core_nodes = {u for u in G.nodes if core[u] >= k}

    collapse_count = len(collapse_nodes)
    if collapse_count == 0:
        return 0

    cost = collapse_count // 2
    if collapse_count % 2 == 1:
        target_set = component_nodes | k_core_nodes
        if target_set:
            cost += 1
        else:
            raise RuntimeError("No valid connection target for remaining collapse node")

    return cost


def select_components_dp_under_budget(
    component_info: List[Tuple[Set[int], Set[int], int]],
    budget: int
) -> List[Tuple[Set[int], Set[int]]]:
    """
    Selects the best subset of components to convert within the given edge budget using dynamic programming.
    Each component has:
        - cost = estimated edge cost to convert (not collapse node count)
        - value = number of promoted nodes (component size)
    """
    n = len(component_info)
    dp = [[0] * (budget + 1) for _ in range(n + 1)]

    values = [len(comp_nodes) for comp_nodes, _, _ in component_info]
    costs = [cost for _, _, cost in component_info]

    for i in range(1, n + 1):
        for w in range(budget + 1):
            if costs[i - 1] <= w:
                dp[i][w] = max(
                    dp[i - 1][w],
                    dp[i - 1][w - costs[i - 1]] + values[i - 1]
                )
            else:
                dp[i][w] = dp[i - 1][w]

    selected = []
    w = budget
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected.append((component_info[i - 1][0], component_info[i - 1][1]))
            w -= costs[i - 1]

    return selected


def apply_component_conversions(
    G: nx.Graph,
    selected_components: List[Tuple[Set[int], Set[int]]],
    k: int
) -> Tuple[nx.Graph, List[Tuple[int, int]]]:
    """
    Applies complete conversions to selected components by adding the minimum
    number of edges required to promote all collapse nodes into the k-core.
    Follows Algorithm 1 from the FastCM+ paper.
    """
    G_reinforced = G.copy()
    added_edges = []

    core = nx.core_number(G)
    k_core_nodes = {u for u in G.nodes if core[u] >= k}

    for comp_nodes, collapse_nodes in selected_components:
        collapse_nodes = sorted(collapse_nodes)

        for i in range(0, len(collapse_nodes) - 1, 2):
            u, v = collapse_nodes[i], collapse_nodes[i + 1]
            if not G_reinforced.has_edge(u, v):
                G_reinforced.add_edge(u, v)
                added_edges.append((u, v))

        if len(collapse_nodes) % 2 == 1:
            last = collapse_nodes[-1]
            target_set = comp_nodes | k_core_nodes
            target_set = [v for v in target_set if v != last and not G_reinforced.has_edge(last, v)]

            if target_set:
                v = sorted(target_set)[0]
                G_reinforced.add_edge(last, v)
                added_edges.append((last, v))
            else:
                raise RuntimeError(f"No valid target found for collapse node {last}")

    return G_reinforced, added_edges

def fastcm_plus_reinforce(G: nx.Graph, k: int = None, budget: int = 10) -> Tuple[nx.Graph, List[Tuple[int, int]]]: # type: ignore
    """
    Main FastCM+ algorithm that orchestrates all components.
    
    Parameters:
        G: Input graph
        k: Target k-core to maximize (if None, uses max_core + 1)
        budget: Number of edges that can be added
        
    Returns:
        Tuple of (reinforced_graph, added_edges)
    """
    # Determine target k-core
    if k is None:
        current_cores = nx.core_number(G)
        if not current_cores:
            return G.copy(), []
        max_core = max(current_cores.values())
        k = max_core + 1  # Try to expand to next level
    
    # Check if (k-1)-shell exists
    core_numbers = nx.core_number(G)
    k_minus_1_shell = {node for node, core_val in core_numbers.items() if core_val == k - 1}
    
    if not k_minus_1_shell:
        # No (k-1)-shell exists, cannot improve k-core
        return G.copy(), []
    
    # Step 1: Get (k-1)-shell components with collapse nodes
    try:
        components_info = get_shell_components_with_collapse_nodes(G, k)
    except Exception:
        return G.copy(), []
    
    if not components_info:
        return G.copy(), []
    
    # Step 2: Calculate conversion costs for each component
    component_costs = []
    for comp_nodes, collapse_nodes in components_info:
        try:
            cost = estimate_complete_conversion_cost(G, comp_nodes, collapse_nodes, k)
            component_costs.append((comp_nodes, collapse_nodes, cost))
        except RuntimeError:
            # Skip components that can't be converted
            continue
    
    if not component_costs:
        return G.copy(), []
    
    # Step 3: Select best components using dynamic programming
    try:
        selected_components = select_components_dp_under_budget(component_costs, budget)
    except Exception:
        return G.copy(), []
    
    if not selected_components:
        return G.copy(), []
    
    # Step 4: Apply conversions to selected components
    try:
        G_reinforced, added_edges = apply_component_conversions(G, selected_components, k)
        return G_reinforced, added_edges
    except Exception:
        return G.copy(), []


def fastcm_reinforce(G: nx.Graph, budget: int = 10) -> nx.Graph:
    """
    Simplified interface for FastCM+ that just returns the reinforced graph.
    Matches the interface pattern of MRKC for easy comparison.
    
    Parameters:
        G: Input graph
        budget: Number of edges that can be added
        
    Returns:
        Reinforced graph
    """
    G_reinforced, _ = fastcm_plus_reinforce(G, budget=budget)
    return G_reinforced


# For backwards compatibility and testing
def fastcm_plus(G: nx.Graph, k: int, budget: int) -> Tuple[nx.Graph, List[Tuple[int, int]]]:
    """
    Alternative interface that requires explicit k parameter.
    """
    return fastcm_plus_reinforce(G, k=k, budget=budget)
