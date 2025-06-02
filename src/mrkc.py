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

    # Identify V_delta: nodes that rely on neighbors with higher core number
    V_delta = set()
    for u in G.nodes():
        same_core_neighbors = sum(1 for v in G.neighbors(u) if core_numbers[v] == core_numbers[u])
        if same_core_neighbors < core_numbers[u]:
            V_delta.add(u)

    # Process nodes in increasing order of core number
    for k in range(1, max(core_numbers.values()) + 1):
        nodes_with_core_k = [node for node, core in core_numbers.items() if core == k]

        for node in nodes_with_core_k:
            if node in V_delta:
                neighbors_same_core = sum(
                    1 for neighbor in G.neighbors(node) if core_numbers[neighbor] == k
                )
                delta_factor = 1.0 - (neighbors_same_core / k) if k != 0 else 0.0

                higher_neighbors = [
                    neighbor for neighbor in G.neighbors(node) if core_numbers[neighbor] > k
                ]

                if higher_neighbors:
                    contribution = (
                        delta_factor * core_influence[node] / len(higher_neighbors)
                    )
                    for neighbor in higher_neighbors:
                        core_influence[neighbor] += contribution

    return core_influence


def generate_candidate_edges(G, core_numbers=None):
    """
    Generate only edges that don't change any node's core number.
    This is a critical constraint from the MRKC paper.
    """
    if core_numbers is None:
        core_numbers = nx.core_number(G)

    candidate_edges = []

    for u, v in itertools.combinations(G.nodes(), 2):
        if not G.has_edge(u, v):
            G_test = G.copy()
            G_test.add_edge(u, v)
            new_core = nx.core_number(G_test)
            if all(new_core[node] == core_numbers[node] for node in G.nodes()):
                candidate_edges.append((u, v))

    return candidate_edges


def mrkc_reinforce(G, budget=10):
    """
    Implements the full MRKC edge selection strategy using
    core strength and core influence for scoring as defined in the paper.

    Parameters:
        G (networkx.Graph): Original graph
        budget (int): Number of edges to add

    Returns:
        G_reinforced (networkx.Graph): Graph after adding reinforcement edges
        added_edges (list): List of edges that were added
    """
    G_reinforced = G.copy()
    added_edges = []

    # Calculate core numbers, core strength, and core influence
    core = nx.core_number(G)
    cs = {u: core_strength(G, u, core=core) for u in G.nodes()}
    ci = compute_core_influence(G, core_numbers=core)

    # Get only candidate edges that preserve core numbers
    potential_edges = generate_candidate_edges(G, core_numbers=core)

    # Helper to avoid division by zero
    def safe_div(numerator, denominator):
        return float('inf') if denominator == 0 else numerator / denominator

    # Score edges using the paper's priority formula
    edge_scores = []
    for u, v in potential_edges:
        if core[u] < core[v]:
            priority = safe_div(ci[u], cs[u])
        elif core[u] > core[v]:
            priority = safe_div(ci[v], cs[v])
        else:  # core[u] == core[v]
            priority = safe_div(ci[u], cs[u]) + safe_div(ci[v], cs[v])

        edge_scores.append(((u, v), priority))

    # Select top edges
    edge_scores.sort(key=lambda x: x[1], reverse=True)
    for (u, v), _ in edge_scores[:budget]:
        G_reinforced.add_edge(u, v)
        added_edges.append((u, v))

    return G_reinforced, added_edges
