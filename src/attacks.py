import networkx as nx

def degree_based_attack(G, num_nodes=1):
    """
    Removes the top `num_nodes` highest-degree nodes from the graph.
    Returns a new graph (copy) with those nodes removed.
    """
    G_copy = G.copy()
    
    # Sort nodes by degree (descending)
    nodes_sorted = sorted(G_copy.degree, key=lambda x: x[1], reverse=True)
    
    # Pick top nodes
    to_remove = [node for node, _ in nodes_sorted[:num_nodes]]
    
    G_copy.remove_nodes_from(to_remove)
    print("Degree-based nodes removed:", to_remove)

    return G_copy

def kcore_based_attack(G, num_nodes=1):
    """
    Removes the top `num_nodes` nodes with the highest k-core values.
    If multiple nodes have the same core value, uses degree as a tiebreaker.
    """
    G_copy = G.copy()
    
    # Get core numbers (dict: node -> k-core value)
    core_dict = nx.core_number(G_copy)
    
    # Convert to list of (node, core_value) pairs
    core_items = list(core_dict.items())
    
    # Sort by core value (descending), then by degree as tiebreaker
    core_items.sort(key=lambda x: (x[1], G_copy.degree[x[0]]), reverse=True)
    
    # Extract top nodes
    to_remove = [node for node, _ in core_items[:num_nodes]]
    
    G_copy.remove_nodes_from(to_remove)
    print("K-core-based nodes removed:", to_remove)

    return G_copy
