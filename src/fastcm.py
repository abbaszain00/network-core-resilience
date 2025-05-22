from typing import List, Tuple, Set
import networkx as nx

def get_shell_components_with_collapse_nodes(G: nx.Graph, k: int) -> List[Tuple[Set[int], Set[int]]]:
    """
    Identifies connected components in the (k-1)-shell and the collapse nodes in each.

    Parameters:
        G (networkx.Graph): Original graph
        k (int): The target core level for reinforcement

    Returns:
        List of tuples: (component_nodes, collapse_nodes_within_component)
    """
    core = nx.core_number(G)
    
    # Step 1: Get (k-1)-shell nodes
    shell_nodes = [u for u in G.nodes if core[u] == k - 1]
    shell_subgraph = G.subgraph(shell_nodes)

    # Step 2: Find connected components in the shell
    components = list(nx.connected_components(shell_subgraph))

    result = []

    for comp in components:
        collapse_nodes = set()
        for u in comp:
            shell_neighbors = [v for v in G.neighbors(u) if core.get(v, 0) == k - 1]
            if len(shell_neighbors) == k - 1:
                collapse_nodes.add(u)

        result.append((comp, collapse_nodes))

    return result
