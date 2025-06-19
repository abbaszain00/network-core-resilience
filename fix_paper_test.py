"""
Fix the paper test discrepancy - why does FastCM+ work in debug but not paper test?
"""

import networkx as nx
from src.fastcm import fastcm_plus_reinforce

def compare_test_setups():
    """Compare the exact networks and parameters from both tests."""
    
    print("COMPARING TEST SETUPS")
    print("="*50)
    
    # Test 1: Recreate EXACT paper test conditions
    print("\nTest 1: Paper Test Recreation")
    G_paper = nx.connected_watts_strogatz_graph(4941, 4, 0.1)  # Same as paper
    
    cores = nx.core_number(G_paper)
    max_core = max(cores.values())
    k_minus_1_shell = sum(1 for c in cores.values() if c == max_core - 1)
    
    print(f"  Network: {G_paper.number_of_nodes()} nodes, max_core={max_core}, (k-1)-shell={k_minus_1_shell}")
    
    # Test with paper's exact budget
    paper_budget = 5
    print(f"  Testing with paper budget = {paper_budget}")
    
    try:
        G_result, edges = fastcm_plus_reinforce(G_paper, paper_budget)
        print(f"  Result: {len(edges)} edges added")
        if len(edges) > 0:
            print(f"  ✅ FastCM+ WORKS in paper test recreation!")
        else:
            print(f"  ❌ FastCM+ fails in paper test recreation")
    except Exception as e:
        print(f"  ❌ FastCM+ crashed: {e}")
    
    # Test 2: Debug test exact recreation
    print(f"\nTest 2: Debug Test Recreation")
    G_debug = nx.connected_watts_strogatz_graph(4941, 4, 0.1)  # Same parameters
    
    cores = nx.core_number(G_debug)
    max_core = max(cores.values())
    k_minus_1_shell = sum(1 for c in cores.values() if c == max_core - 1)
    
    print(f"  Network: {G_debug.number_of_nodes()} nodes, max_core={max_core}, (k-1)-shell={k_minus_1_shell}")
    
    debug_budget = 3
    print(f"  Testing with debug budget = {debug_budget}")
    
    try:
        G_result, edges = fastcm_plus_reinforce(G_debug, debug_budget)
        print(f"  Result: {len(edges)} edges added")
        if len(edges) > 0:
            print(f"  ✅ FastCM+ WORKS in debug test recreation!")
        else:
            print(f"  ❌ FastCM+ fails in debug test recreation")
    except Exception as e:
        print(f"  ❌ FastCM+ crashed: {e}")
    
    # Test 3: Same network, different budgets
    print(f"\nTest 3: Budget Sensitivity on Same Network")
    budgets = [1, 2, 3, 4, 5, 6, 7, 8, 10]
    
    print(f"  Using the same network instance...")
    G_same = nx.connected_watts_strogatz_graph(4941, 4, 0.1)
    
    for budget in budgets:
        try:
            G_result, edges = fastcm_plus_reinforce(G_same, budget)
            status = "✅" if len(edges) > 0 else "❌"
            print(f"    Budget {budget:2d}: {len(edges):2d} edges {status}")
        except Exception as e:
            print(f"    Budget {budget:2d}: CRASH - {e}")

def test_exact_paper_network():
    """Test the exact network from your paper output."""
    
    print(f"\nTEST EXACT PAPER NETWORK CONDITIONS")
    print("="*50)
    
    # Your paper output showed:
    # Network: 4941 nodes, 9882 edges, Max k-core: 3, (k-1)-shell size: 46
    
    print("Creating network to match paper output exactly...")
    
    # Try different random seeds to get the exact same network
    for seed in range(10):
        print(f"\nTrying seed {seed}:")
        
        # Set seed for reproducibility
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        
        G = nx.connected_watts_strogatz_graph(4941, 4, 0.1)
        
        cores = nx.core_number(G)
        max_core = max(cores.values())
        k_minus_1_shell = sum(1 for c in cores.values() if c == max_core - 1)
        
        print(f"  Network: {G.number_of_edges()} edges, max_core={max_core}, (k-1)-shell={k_minus_1_shell}")
        
        # Check if this matches your paper output
        if G.number_of_edges() == 9882 and max_core == 3 and k_minus_1_shell == 46:
            print(f"  🎯 EXACT MATCH FOUND with seed {seed}!")
            
            # Test FastCM+ on this exact network
            for budget in [3, 5, 10]:
                try:
                    G_result, edges = fastcm_plus_reinforce(G, budget)
                    print(f"    Budget {budget}: {len(edges)} edges")
                except Exception as e:
                    print(f"    Budget {budget}: FAILED - {e}")
            break
        else:
            print(f"    No match (edges={G.number_of_edges()}, max_core={max_core}, k-1={k_minus_1_shell})")

if __name__ == "__main__":
    compare_test_setups()
    test_exact_paper_network()