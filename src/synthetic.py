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


def generate_networks_batch(sizes=[100, 500, 1000], seed=42):
    """Generate multiple synthetic networks for experiments."""
    networks = {}
    types = ["random", "scale_free", "small_world"]
    
    for size in sizes:
        for net_type in types:
            name = f"{net_type}_{size}"
            networks[name] = get_test_graph(net_type, n=size, seed=seed)
    
    return networks


def get_parameter_variants(base_size=500, seed=42):
    """Create networks with different parameters for testing sensitivity."""
    networks = {}
    
    # Random graphs with different densities
    for p in [0.1, 0.2, 0.3]:
        name = f"random_{base_size}_dense{int(p*10)}"
        networks[name] = nx.erdos_renyi_graph(n=base_size, p=p, seed=seed)
    
    # Scale-free with different attachment rates
    for m in [1, 2, 3]:
        name = f"scalefree_{base_size}_m{m}"
        networks[name] = nx.barabasi_albert_graph(n=base_size, m=m, seed=seed)
    
    # Small-world with different rewiring
    for p in [0.05, 0.1, 0.2]:
        name = f"smallworld_{base_size}_rewire{int(p*100)}"
        networks[name] = nx.watts_strogatz_graph(n=base_size, k=4, p=p, seed=seed)
    
    return networks


def get_all_synthetic(max_nodes=5000, include_variants=False):
    """Load all synthetic networks for experiments."""
    networks = {}
    
    # Standard sizes
    sizes = [s for s in [100, 500, 1000, 2000] if s <= max_nodes]
    networks.update(generate_networks_batch(sizes))
    
    if include_variants:
        networks.update(get_parameter_variants(min(500, max_nodes)))
    
    return networks


def network_stats(G):
    """Basic network statistics."""
    if G.number_of_nodes() == 0:
        return {}
    
    core_nums = nx.core_number(G)
    degrees = dict(G.degree())
    
    return {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'density': round(nx.density(G), 3),
        'max_core': max(core_nums.values()) if core_nums else 0,
        'avg_degree': round(sum(degrees.values()) / len(degrees), 1) if degrees else 0,
        'connected': nx.is_connected(G)
    }


def print_network_info(networks):
    """Print summary of networks."""
    print(f"Generated {len(networks)} synthetic networks:")
    print()
    
    # Group by type
    by_type = {}
    for name, G in networks.items():
        net_type = name.split('_')[0]
        if net_type not in by_type:
            by_type[net_type] = []
        by_type[net_type].append((name, G))
    
    for net_type in ['random', 'scale', 'scalefree', 'small', 'smallworld']:
        if net_type in by_type or any(net_type in key for key in by_type.keys()):
            # Find matching networks
            matches = []
            for key, nets in by_type.items():
                if net_type in key:
                    matches.extend(nets)
            
            if matches:
                print(f"{net_type.upper()} networks:")
                for name, G in sorted(matches, key=lambda x: x[1].number_of_nodes()):
                    stats = network_stats(G)
                    print(f"  {name}: {stats['nodes']} nodes, {stats['edges']} edges, "
                          f"k-core={stats['max_core']}")
                print()


if __name__ == "__main__":
    # Test basic generation
    test_net = get_test_graph("scale_free", 100)
    print(f"Test network: {test_net.number_of_nodes()} nodes")
    
    # Generate batch for experiments
    networks = get_all_synthetic(max_nodes=1000)
    print_network_info(networks)