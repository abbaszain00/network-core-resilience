import networkx as nx

def generate_random_graph(n=10, p=0.3, seed=None):
    """Generate an Erdős-Rényi random graph."""
    return nx.erdos_renyi_graph(n=n, p=p, seed=seed)

def generate_scale_free_graph(n=10, m=2):
    """Generate a Barabási–Albert scale-free graph."""
    return nx.barabasi_albert_graph(n=n, m=m)

def generate_small_world_graph(n=10, k=4, p=0.1, seed=None):
    """Generate a Watts-Strogatz small-world graph."""
    return nx.watts_strogatz_graph(n=n, k=k, p=p, seed=seed)
