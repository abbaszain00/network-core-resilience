#!/usr/bin/env python3
"""
Simple fix: Create shared network loading function.
Both scripts use this to ensure identical networks.
"""

import sys
sys.path.append('src')

from real_world import get_all_real_networks
from synthetic import get_all_synthetic

def get_consistent_networks(max_network_size=5000):
    """
    Single function that both final_evaluation.py and failure_diagnostics.py use.
    Ensures identical network loading parameters.
    """
    
    print(f"Loading networks consistently (max_size={max_network_size})...")
    
    networks = {}
    
    # Real networks - consistent parameters
    try:
        real_nets = get_all_real_networks(max_nodes=max_network_size)
        networks.update(real_nets)
        print(f"✅ Loaded {len(real_nets)} real networks")
    except Exception as e:
        print(f"❌ Error loading real networks: {e}")
    
    # Synthetic networks - consistent parameters (THIS WAS THE ISSUE!)
    try:
        synthetic_nets = get_all_synthetic(
            max_nodes=max_network_size, 
            include_variants=True  # ← Ensure both scripts use this
        )
        networks.update(synthetic_nets)
        print(f"✅ Loaded {len(synthetic_nets)} synthetic networks")
    except Exception as e:
        print(f"❌ Error loading synthetic networks: {e}")
    
    print(f"📊 Total networks: {len(networks)}")
    return networks

# Usage in both scripts:
# Replace the network loading sections with:
# networks = get_consistent_networks(max_network_size=your_size)

if __name__ == "__main__":
    print("TESTING CONSISTENT NETWORK LOADING")
    print("=" * 40)
    
    # Test the function
    networks = get_consistent_networks(max_network_size=5000)
    
    print(f"\nNetwork summary:")
    for name, G in networks.items():
        print(f"  {name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    print(f"\n✅ Use this function in both evaluation scripts for consistency!")