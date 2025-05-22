import networkx as nx
from collections import Counter

def average_core_number(G):
    """
    Returns the average k-core number of all nodes in graph G.
    """
    if G.number_of_nodes() == 0:
        return 0.0
    
    core_dict = nx.core_number(G)
    avg = sum(core_dict.values()) / len(core_dict)
    return avg

def max_core_number(G):
    """
    Returns the highest core number in the graph.
    """
    if G.number_of_nodes() == 0:
        return 0
    core_dict = nx.core_number(G)
    return max(core_dict.values())
def core_distribution(G):
    """
    Returns a Counter of how many nodes belong to each k-core value.
    """
    if G.number_of_nodes() == 0:
        return Counter()
    core_dict = nx.core_number(G)
    return Counter(core_dict.values())

def retained_top_kcore_members(G_before, G_after):
    """
    Returns how many of the original top-k-core nodes are still in the same
    top-k-core after attack.

    Parameters:
        G_before (networkx.Graph): Graph before attack
        G_after (networkx.Graph): Graph after attack

    Returns:
        (int, float): Count and percentage of original top-core nodes still in top-core after attack
    """
    if G_before.number_of_nodes() == 0 or G_after.number_of_nodes() == 0:
        return 0, 0.0

    core_before = nx.core_number(G_before)
    core_after = nx.core_number(G_after)

    max_before = max(core_before.values())
    max_after = max(core_after.values())

    original_top_nodes = {n for n, k in core_before.items() if k == max_before}
    retained_in_top = {n for n in original_top_nodes if n in core_after and core_after[n] == max_after}

    return len(retained_in_top), len(retained_in_top) / len(original_top_nodes)


