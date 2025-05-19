import networkx as nx

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

from collections import Counter

def core_distribution(G):
    """
    Returns a Counter of how many nodes belong to each k-core value.
    """
    if G.number_of_nodes() == 0:
        return Counter()
    core_dict = nx.core_number(G)
    return Counter(core_dict.values())
