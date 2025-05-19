import networkx as nx
import itertools

def mrkc_reinforce(G, budget=10):
    """
    Reinforces the graph by adding `budget` edges between unconnected nodes
    that have the most common neighbors (simple proxy for boosting resilience).
    
    Parameters:
        G (networkx.Graph): The original graph
        budget (int): Number of edges to add

    Returns:
        G_reinforced (networkx.Graph): Modified graph
        added_edges (list of tuple): Edges that were added
    """
    G_reinforced = G.copy()
    added_edges = []

    # Step 1: Get all unconnected node pairs
    potential_edges = [
        (u, v) for u, v in itertools.combinations(G.nodes(), 2)
        if not G.has_edge(u, v)
    ]

    # Step 2: Score each pair by number of common neighbors
    edge_scores = []
    for u, v in potential_edges:
        score = len(list(nx.common_neighbors(G, u, v)))
        edge_scores.append(((u, v), score))

    # Step 3: Sort by score (descending) and add top edges
    edge_scores.sort(key=lambda x: x[1], reverse=True)
    for (u, v), _ in edge_scores[:budget]:
        G_reinforced.add_edge(u, v)
        added_edges.append((u, v))

    return G_reinforced, added_edges
