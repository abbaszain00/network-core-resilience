#!/usr/bin/env python3
"""
Updated network loading with SNAP datasets.

Added some real SNAP networks to strengthen the dataset mix.
Now includes benchmark networks, synthetic networks, AND real datasets
from the SNAP repository for a more comprehensive evaluation.
"""

import sys
sys.path.append('src')
import networkx as nx

from real_world import get_all_real_networks
from synthetic import get_all_synthetic

def get_consistent_networks(max_network_size=5000, include_snap=True):
    """
    Load all networks consistently for evaluation.
    
    Now includes SNAP networks to strengthen the real-world dataset mix.
    Added include_snap flag in case SNAP download fails.
    """
    
    print(f"Loading networks consistently (max_size={max_network_size})...")
    
    networks = {}
    
    # Real networks - consistent parameters
    try:
        real_nets = get_all_real_networks(max_nodes=max_network_size)
        networks.update(real_nets)
        print(f"Loaded {len(real_nets)} built-in real networks")
    except Exception as e:
        print(f"Error loading built-in real networks: {e}")
    
    # SNAP networks - real datasets from Stanford repository
    if include_snap:
        try:
            from snap_loader import get_snap_networks
            snap_nets = get_snap_networks(max_nodes=max_network_size)
            networks.update(snap_nets)
            print(f"Loaded {len(snap_nets)} SNAP networks")
        except ImportError:
            print("SNAP loader not available - install requests if needed")
        except Exception as e:
            print(f"Could not load SNAP networks: {e}")
            print("Continuing with other networks...")
    
    # Synthetic networks - consistent parameters  
    try:
        synthetic_nets = get_all_synthetic(
            max_nodes=max_network_size, 
            include_variants=True
        )
        networks.update(synthetic_nets)
        print(f"Loaded {len(synthetic_nets)} synthetic networks")
    except Exception as e:
        print(f"Error loading synthetic networks: {e}")
    
    print(f"Total networks: {len(networks)}")
    
    # Show network breakdown for transparency
    if len(networks) > 0:
        builtin_count = len([n for n in networks.keys() if any(x in n for x in ['karate', 'florentine', 'davis', 'miserables', 'petersen'])])
        snap_count = len([n for n in networks.keys() if any(x in n for x in ['facebook', 'email', 'ca_'])])
        synthetic_count = len([n for n in networks.keys() if 'synthetic' in n])
        other_count = len(networks) - builtin_count - snap_count - synthetic_count
        
        print(f"   Built-in real: {builtin_count}")
        print(f"   SNAP real: {snap_count}")
        print(f"   Synthetic: {synthetic_count}")
        print(f"   Other: {other_count}")
    
    return networks

if __name__ == "__main__":
    print("TESTING UPDATED NETWORK LOADING WITH SNAP")
    print("=" * 50)
    
    # Test with SNAP networks
    print("\nTesting with SNAP networks:")
    networks = get_consistent_networks(max_network_size=5000, include_snap=True)
    
    print(f"\nNetwork summary:")
    for name, G in networks.items():
        cores = nx.core_number(G) if hasattr(G, 'nodes') else {}
        max_core = max(cores.values()) if cores else 0
        print(f"  {name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, max k-core: {max_core}")
    
    print(f"\nEnhanced network collection ready for evaluation!")
    print(f"Total: {len(networks)} networks")
    print(f"Mix of built-in real, SNAP real, and synthetic networks")