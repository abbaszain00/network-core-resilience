import networkx as nx

def get_test_graph(type="scale_free", n=30, seed=42):
    """
    Returns a synthetic test graph of the given type.

    Parameters:
        type (str): One of "random", "scale_free", or "small_world"
        n (int): Number of nodes
        seed (int): Random seed for reproducibility

    Returns:
        networkx.Graph: Generated graph
    """
    if type == "random":
        return nx.erdos_renyi_graph(n=n, p=0.2, seed=seed)
    elif type == "small_world":
        return nx.watts_strogatz_graph(n=n, k=4, p=0.1, seed=seed)
    elif type == "scale_free":
        return nx.barabasi_albert_graph(n=n, m=2, seed=seed)
    else:
        raise ValueError(f"Unknown graph type: {type}")
