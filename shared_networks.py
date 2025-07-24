#!/usr/bin/env python3
"""
Network loading module for consistent dataset handling.
"""

import sys
sys.path.append('src')
import networkx as nx

from real_world import get_all_real_networks
from synthetic import get_all_synthetic

def get_consistent_networks(max_network_size=6000, include_snap=True):
    """
    Load networks from multiple sources for evaluation.
    
    Args:
        max_network_size: Maximum number of nodes to include
        include_snap: Whether to attempt loading SNAP datasets
    
    Returns:
        dict: Network name -> NetworkX graph
    """
    
    print(f"Loading networks (max_size={max_network_size})...")
    
    networks = {}
    
    # Built-in NetworkX networks
    try:
        real_nets = get_all_real_networks(max_nodes=max_network_size)
        networks.update(real_nets)
        print(f"Loaded {len(real_nets)} built-in networks")
    except Exception as e:
        print(f"Error loading built-in networks: {e}")
    
    # SNAP datasets
    if include_snap:
        try:
            from snap_loader import get_snap_networks
            snap_nets = get_snap_networks(max_nodes=max_network_size)
            networks.update(snap_nets)
            print(f"Loaded {len(snap_nets)} SNAP networks")
        except ImportError:
            print("SNAP loader unavailable")
        except Exception as e:
            print(f"SNAP loading failed: {e}")
    
    # Synthetic networks
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
    
    if len(networks) > 0:
        # Categorize networks
        builtin_real = [n for n in networks.keys() if any(x in n for x in 
                       ['karate', 'florentine', 'davis', 'miserables', 'petersen', 'house'])]
        
        # Updated SNAP network detection to include all current SNAP networks
        snap_real = [n for n in networks.keys() if any(x in n for x in 
                    ['facebook_ego', 'email_enron', 'ca_grqc', 'bitcoin_alpha', 
                     'bitcoin_otc', 'wiki_vote', 'email_eu_core',])]
        
        pure_synthetic = [n for n in networks.keys() if 'synthetic' in n]
        
        realistic_synthetic = [n for n in networks.keys() if n not in builtin_real 
                             and n not in snap_real and n not in pure_synthetic]
        
        print(f"   Built-in real: {len(builtin_real)}")
        print(f"   SNAP real: {len(snap_real)}")
        print(f"   Realistic synthetic: {len(realistic_synthetic)}")
        print(f"   Pure synthetic: {len(pure_synthetic)}")
        
        # Debug info to see which networks are in each category
        if len(snap_real) > 0:
            print(f"   SNAP networks: {snap_real}")
        if len(realistic_synthetic) > 0:
            print(f"   Realistic synthetic: {realistic_synthetic}")
        
        total_real = len(builtin_real) + len(snap_real)
        real_percentage = (total_real / len(networks)) * 100
        
        print(f"\nNetwork Balance: {total_real}/{len(networks)} real ({real_percentage:.1f}%)")
    
    return networks

if __name__ == "__main__":
    print("Testing network loading functionality")
    print("=" * 40)
    
    networks = get_consistent_networks(max_network_size=6000, include_snap=True)
    
    print(f"\nNetwork details:")
    for name, G in networks.items():
        cores = nx.core_number(G) if hasattr(G, 'nodes') else {}
        max_core = max(cores.values()) if cores else 0
        print(f"  {name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, max k-core: {max_core}")
    
    print(f"\nLoaded {len(networks)} networks total")