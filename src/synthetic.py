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


def get_all_synthetic(max_nodes=5000, include_variants=False):
    """Load essential synthetic networks for controlled analysis."""
    networks = {}
    
    print("Loading essential synthetic networks...")
    
    # Essential networks only - one of each major topology class
    essential_configs = [
        ('random', 500),      # Erdős-Rényi baseline
        ('scale_free', 500),  # Barabási-Albert 
        ('small_world', 500), # Watts-Strogatz
    ]
    
    # Add one larger network for scale testing
    if max_nodes >= 1000:
        essential_configs.append(('scale_free', 1000))  # Most real networks are scale-free
    
    for net_type, size in essential_configs:
        if size <= max_nodes:
            name = f"{net_type}_{size}"
            try:
                networks[name] = get_test_graph(net_type, n=size, seed=42)
                print(f"Generated {name}")
            except Exception as e:
                print(f"Failed to generate {name}: {e}")
    
    print(f"Generated {len(networks)} essential synthetic networks")
    
    # Warning if variants requested (not recommended for final evaluation)
    if include_variants:
        print("Warning: include_variants=True not supported in streamlined version")
        print("Use essential networks only for proper real/synthetic balance")
    
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
    print(f"Synthetic networks ({len(networks)} total):")
    
    for name, G in sorted(networks.items(), key=lambda x: x[1].number_of_nodes()):
        stats = network_stats(G)
        print(f"  {name}: {stats['nodes']} nodes, {stats['edges']} edges, k-core={stats['max_core']}")


if __name__ == "__main__":
    print("STREAMLINED SYNTHETIC NETWORK GENERATION")
    print("=" * 50)
    
    # Generate essential networks
    networks = get_all_synthetic(max_nodes=1000, include_variants=False)
    print_network_info(networks)
    
    print(f"\nBalance projection:")
    print(f"• Synthetic: {len(networks)} networks")
    print(f"• Expected real: ~5-6 networks")
    print(f"• Total: ~{len(networks) + 5} networks")
    print(f"• Real percentage: ~{5/(len(networks) + 5)*100:.0f}%")
    
    if 5/(len(networks) + 5) >= 0.6:
        print("✅ Will achieve 60%+ real balance")
    else:
        print("⚠️  May need more real networks")