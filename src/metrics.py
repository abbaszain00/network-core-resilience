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
