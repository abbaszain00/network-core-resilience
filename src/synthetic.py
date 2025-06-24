import networkx as nx

def create_synthetic_network(network_type, n=500, seed=42):
    """Create a synthetic network of given type."""
    
    if network_type == "random":
        # Erdos-Renyi random graph
        return nx.erdos_renyi_graph(n=n, p=0.1, seed=seed)
    elif network_type == "scale_free":
        # Barabasi-Albert preferential attachment
        return nx.barabasi_albert_graph(n=n, m=3, seed=seed)
    elif network_type == "small_world":
        # Watts-Strogatz small world
        return nx.watts_strogatz_graph(n=n, k=6, p=0.3, seed=seed)
    else:
        raise ValueError(f"Unknown type: {network_type}")

def get_all_synthetic(max_nodes=5000, include_variants=False):
    """Get synthetic networks for testing."""
    networks = {}
    
    print("Creating synthetic networks...")
    
    # Basic synthetic networks for comparison
    types_to_create = [
        ("random", 500),
        ("scale_free", 500),
        ("small_world", 500)
    ]
    
    # Add larger version if we have space
    if max_nodes >= 1000:
        types_to_create.append(("scale_free", 1000))
    
    for net_type, size in types_to_create:
        if size <= max_nodes:
            try:
                G = create_synthetic_network(net_type, n=size)
                
                # Make sure it's connected
                if not nx.is_connected(G):
                    # Take largest component
                    largest = max(nx.connected_components(G), key=len)
                    G = G.subgraph(largest).copy()
                
                name = f"synthetic_{net_type}_{size}"
                networks[name] = G
                print(f"Created {name}: {G.number_of_nodes()} nodes")
                
            except Exception as e:
                print(f"Failed to create {net_type}: {e}")
    
    print(f"Created {len(networks)} synthetic networks")
    return networks

def get_basic_stats(G):
    """Get basic network statistics."""
    cores = nx.core_number(G)
    
    stats = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'density': round(nx.density(G), 4),
        'max_core': max(cores.values()) if cores else 0,
        'connected': nx.is_connected(G)
    }
    
    # Add clustering if not too big
    if G.number_of_nodes() <= 1000:
        try:
            stats['clustering'] = round(nx.average_clustering(G), 3)
        except:
            stats['clustering'] = 0
    
    return stats

def print_synthetic_summary(networks):
    """Print summary of synthetic networks."""
    print(f"\nSynthetic networks ({len(networks)} total):")
    
    for name, G in networks.items():
        stats = get_basic_stats(G)
        print(f"  {name}:")
        print(f"    {stats['nodes']} nodes, {stats['edges']} edges")
        print(f"    max k-core: {stats['max_core']}, density: {stats['density']}")
        if 'clustering' in stats:
            print(f"    clustering: {stats['clustering']}")

if __name__ == "__main__":
    print("Creating synthetic networks for testing...")
    networks = get_all_synthetic(max_nodes=1000)
    print_synthetic_summary(networks)
    
    # Simple check
    if len(networks) >= 3:
        print("Good - have basic synthetic types")
    else:
        print("Warning - missing some network types")