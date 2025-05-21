import networkx as nx
import itertools
from src import attacks, metrics
import random

def core_strength(G, u, core=None):
    """
    Computes the core strength of node u based on the MRKC paper definition:
    CS(u, G) = |{v in N(u) : core(v) >= core(u)}| - core(u) + 1
    """
    if core is None:
        core = nx.core_number(G)
    ku = core.get(u, 0)
    neighbors = G.neighbors(u)
    delta_ge = [v for v in neighbors if core.get(v, 0) >= ku]
    return len(delta_ge) - ku + 1

def compute_core_influence(G, core_numbers=None):
    """
    Compute the Core Influence of all nodes as defined in the MRKC paper.

    Parameters:
        G (networkx.Graph): The graph
        core_numbers (dict, optional): Pre-computed core numbers

    Returns:
        dict: Mapping of nodes to their core influence values
    """
    if core_numbers is None:
        core_numbers = nx.core_number(G)
    
    # Initialize core influence values to 1.0 for all nodes
    core_influence = {node: 1.0 for node in G.nodes()}
    
    # Identify V_delta: nodes with neighbors having higher core number
    V_delta = set()
    for u in G.nodes():
        if any(core_numbers[v] > core_numbers[u] for v in G.neighbors(u)):
            V_delta.add(u)
    
    # Process nodes in increasing order of core number
    for k in range(1, max(core_numbers.values()) + 1):
        nodes_with_core_k = [node for node, core in core_numbers.items() if core == k]
        
        for node in nodes_with_core_k:
            if node in V_delta:
                # Δ=(v): neighbors with equal core
                neighbors_same_core = sum(1 for neighbor in G.neighbors(node) 
                                          if core_numbers[neighbor] == k)
                delta_factor = 1.0 - (neighbors_same_core / k) if k != 0 else 0.0

                # Δ>(v): neighbors with higher core
                higher_neighbors = [neighbor for neighbor in G.neighbors(node)
                                    if core_numbers[neighbor] > k]
                
                if higher_neighbors:
                    contribution = delta_factor * core_influence[node] / len(higher_neighbors)
                    for neighbor in higher_neighbors:
                        core_influence[neighbor] += contribution

    return core_influence




def simulate_retention_score(G, edge, num_nodes_to_remove=3):
    """
    Simulates the effect of adding an edge and performing an attack,
    returning the number of original top-k-core nodes retained in the top core.

    Returns -1 if edge already exists.
    """
    u, v = edge
    if G.has_edge(u, v):
        return -1  # already connected

    G_temp = G.copy()
    G_temp.add_edge(u, v)

    G_attacked = attacks.degree_based_attack(G_temp, num_nodes=num_nodes_to_remove)

    retained_count, _ = metrics.retained_top_kcore_members(G, G_attacked)
    return retained_count


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

    # Step 2: Score edges by resilience (simulated top-k-core retention)
    edge_scores = []
    for u, v in potential_edges:
        score = simulate_retention_score(G, (u, v), num_nodes_to_remove=3)
        edge_scores.append(((u, v), score))


    # Step 3: Sort by score (descending) and add top edges
    edge_scores.sort(key=lambda x: x[1], reverse=True)
    for (u, v), _ in edge_scores[:budget]:
        G_reinforced.add_edge(u, v)
        added_edges.append((u, v))

    return G_reinforced, added_edges
