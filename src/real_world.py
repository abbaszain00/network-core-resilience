import os
import gzip
import urllib.request
import networkx as nx

def load_snap_graph(name):
    """
    Downloads and loads a SNAP graph by name (e.g., 'ca-GrQc').
    Returns a NetworkX Graph object.
    """
    base_url = "https://snap.stanford.edu/data"
    file_name = f"{name}.txt"
    gz_file = f"{file_name}.gz"
    local_dir = "data"
    os.makedirs(local_dir, exist_ok=True)
    
    gz_path = os.path.join(local_dir, gz_file)
    txt_path = os.path.join(local_dir, file_name)
    
    # Download if not already present
    if not os.path.exists(txt_path):
        if not os.path.exists(gz_path):
            print(f"Downloading {gz_file}...")
            urllib.request.urlretrieve(f"{base_url}/{gz_file}", gz_path)
        
        print(f"Unzipping {gz_file}...")
        with gzip.open(gz_path, "rt") as f_in, open(txt_path, "w") as f_out:
            f_out.writelines(f_in)
    
    print(f"Loading graph from {txt_path}...")
    G = nx.read_edgelist(txt_path, comments="#", nodetype=int)
    
    # Clean up
    G = G.to_undirected()
    G.remove_edges_from(nx.selfloop_edges(G))
    
    print(f"Loaded {name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def get_builtin_networks():
    """Load built-in NetworkX networks."""
    networks = {}
    
    networks['karate'] = nx.karate_club_graph()
    networks['florentine'] = nx.florentine_families_graph()
    
    # Try dolphins if available (not in all NetworkX versions)
    try:
        if hasattr(nx, 'dolphins_graph'):
            networks['dolphins'] = getattr(nx, 'dolphins_graph')()
    except:
        pass
    
    return networks


def get_snap_networks():
    """Load the main SNAP networks we want for experiments."""
    networks = {}
    
    # Networks from the papers
    datasets = ['ca-GrQc', 'ca-HepTh', 'email-Eu-core', 'facebook_combined']
    
    for name in datasets:
        try:
            networks[name] = load_snap_graph(name)
        except Exception as e:
            print(f"Couldn't load {name}: {e}")
    
    return networks


def get_more_snap_networks():
    """Load additional SNAP networks if we want more data."""
    networks = {}
    
    more_datasets = ['ca-AstroPh', 'ca-CondMat', 'email-Enron', 'p2p-Gnutella08']
    
    for name in more_datasets:
        try:
            G = load_snap_graph(name)
            # Skip if too big or too small
            if 100 <= G.number_of_nodes() <= 10000:
                networks[name] = G
            else:
                print(f"Skipping {name}: {G.number_of_nodes()} nodes")
        except Exception as e:
            print(f"Couldn't load {name}: {e}")
    
    return networks


def get_all_real_networks(max_nodes=5000, extra=False):
    """Get all real-world networks."""
    networks = {}
    
    # Always get built-ins
    networks.update(get_builtin_networks())
    
    # Try SNAP networks
    try:
        snap_nets = get_snap_networks()
        networks.update(snap_nets)
        print(f"Got {len(snap_nets)} SNAP networks")
    except:
        print("SNAP loading failed")
    
    if extra:
        try:
            more_nets = get_more_snap_networks()
            networks.update(more_nets)
        except:
            print("Extra networks failed")
    
    # Filter by size
    filtered = {}
    for name, G in networks.items():
        if G.number_of_nodes() <= max_nodes:
            filtered[name] = G
    
    return filtered


def basic_stats(G):
    """Get basic network stats."""
    if G.number_of_nodes() == 0:
        return "Empty"
    
    core_nums = nx.core_number(G)
    return {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'max_core': max(core_nums.values()) if core_nums else 0,
        'connected': nx.is_connected(G)
    }


def print_networks(networks):
    """Print info about loaded networks."""
    if not networks:
        print("No networks loaded")
        return
    
    print(f"Loaded {len(networks)} real networks:")
    print()
    
    builtin = ['karate', 'florentine', 'dolphins']
    
    for name, G in networks.items():
        stats = basic_stats(G)
        if isinstance(stats, str):  # Handle "Empty" case
            print(f"{name}: {stats}")
            continue
            
        source = "built-in" if name in builtin else "SNAP"
        nodes = stats.get('nodes', 0)
        max_core = stats.get('max_core', 0)
        print(f"{name} ({source}): {nodes} nodes, k-core={max_core}")


if __name__ == "__main__":
    print("Testing network loading...")
    
    # Test what we can load
    builtin = get_builtin_networks()
    print(f"Built-in: {list(builtin.keys())}")
    
    try:
        snap = get_snap_networks()
        print(f"SNAP: {list(snap.keys())}")
    except:
        print("SNAP failed")
    
    # Load everything
    all_nets = get_all_real_networks(max_nodes=10000)
    print()
    print_networks(all_nets)