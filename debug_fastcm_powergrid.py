"""
Deep debugging of FastCM+ failure on power_grid network.
Investigate why it fails despite having (k-1)-shell of size 46.
"""

import networkx as nx
import numpy as np
from src.fastcm import fastcm_plus_reinforce

def analyze_power_grid_failure():
    """Detailed analysis of why FastCM+ fails on power_grid."""
    
    print("DEBUGGING FASTCM+ FAILURE ON POWER_GRID")
    print("="*60)
    
    # Recreate the exact power_grid network from your test
    G = nx.connected_watts_strogatz_graph(4941, 4, 0.1)
    
    print(f"Power Grid Network:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Average degree: {2*G.number_of_edges()/G.number_of_nodes():.2f}")
    
    # Analyze core structure in detail
    cores = nx.core_number(G)
    max_core = max(cores.values())
    
    print(f"\nCore Structure Analysis:")
    print(f"  Max k-core: {max_core}")
    
    # Detailed shell distribution
    shell_sizes = {}
    shell_nodes = {}
    for k in range(max_core + 1):
        shell_nodes[k] = [n for n, c in cores.items() if c == k]
        shell_sizes[k] = len(shell_nodes[k])
    
    print(f"  Shell distribution:")
    for k in range(max_core + 1):
        print(f"    {k}-shell: {shell_sizes[k]} nodes")
    
    k_minus_1_shell = shell_nodes.get(max_core - 1, [])
    print(f"\n(k-1)-shell Analysis:")
    print(f"  Size: {len(k_minus_1_shell)} nodes")
    print(f"  Sample nodes: {k_minus_1_shell[:10]}")
    
    if len(k_minus_1_shell) == 0:
        print("  ERROR: No (k-1)-shell found!")
        return
    
    # Analyze (k-1)-shell connectivity
    print(f"\n(k-1)-shell Connectivity Analysis:")
    
    # Check if (k-1)-shell nodes are connected to each other
    k_minus_1_set = set(k_minus_1_shell)
    internal_edges = 0
    possible_edges = 0
    
    for i, u in enumerate(k_minus_1_shell[:20]):  # Sample first 20 for speed
        for v in k_minus_1_shell[i+1:21]:  # Check against next 20
            possible_edges += 1
            if G.has_edge(u, v):
                internal_edges += 1
    
    print(f"  Internal connectivity (sample):")
    print(f"    Existing edges: {internal_edges}")
    print(f"    Possible edges: {possible_edges}")
    print(f"    Density: {internal_edges/possible_edges*100:.1f}%" if possible_edges > 0 else "    Density: N/A")
    
    # Analyze degree distribution within (k-1)-shell
    k_minus_1_degrees = []
    k_minus_1_k_core_neighbors = []
    
    for node in k_minus_1_shell[:50]:  # Sample for analysis
        degree = G.degree(node)
        k_core_neighbors = sum(1 for neighbor in G.neighbors(node) 
                              if cores[neighbor] >= max_core)
        k_minus_1_degrees.append(degree)
        k_minus_1_k_core_neighbors.append(k_core_neighbors)
    
    print(f"\n(k-1)-shell Node Analysis (sample of 50):")
    print(f"  Average degree: {np.mean(k_minus_1_degrees):.2f}")
    print(f"  Min degree: {min(k_minus_1_degrees)}")
    print(f"  Max degree: {max(k_minus_1_degrees)}")
    print(f"  Avg neighbors in k-core: {np.mean(k_minus_1_k_core_neighbors):.2f}")
    
    # Test FastCM+ with different budgets to see the pattern
    print(f"\nFastCM+ Budget Testing:")
    budgets_to_test = [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20]
    
    for budget in budgets_to_test:
        try:
            start_time = time.time()
            G_result, edges = fastcm_plus_reinforce(G, budget)
            runtime = time.time() - start_time
            print(f"  Budget {budget:2d}: {len(edges):2d} edges added ({runtime:.3f}s)")
            
            if len(edges) > 0:
                print(f"    Sample edges: {edges[:3]}")
                break  # Stop at first success
        except Exception as e:
            print(f"  Budget {budget:2d}: FAILED - {e}")
    
    # Manual edge analysis - what edges SHOULD be possible?
    print(f"\nManual Edge Possibility Analysis:")
    
    # Check if we can add edges between (k-1)-shell nodes
    sample_nodes = k_minus_1_shell[:10]
    possible_internal_edges = []
    
    for i, u in enumerate(sample_nodes):
        for v in sample_nodes[i+1:]:
            if not G.has_edge(u, v):
                possible_internal_edges.append((u, v))
    
    print(f"  Possible edges between first 10 (k-1)-shell nodes: {len(possible_internal_edges)}")
    if possible_internal_edges:
        print(f"    Examples: {possible_internal_edges[:5]}")
    
    # Check edges from (k-1)-shell to k-core
    k_core_nodes = shell_nodes.get(max_core, [])
    possible_to_k_core = []
    
    for u in sample_nodes[:5]:  # First 5 (k-1)-shell nodes
        for v in k_core_nodes[:10]:  # First 10 k-core nodes
            if not G.has_edge(u, v):
                possible_to_k_core.append((u, v))
    
    print(f"  Possible edges from (k-1)-shell to k-core (sample): {len(possible_to_k_core)}")
    if possible_to_k_core:
        print(f"    Examples: {possible_to_k_core[:5]}")

def compare_working_vs_failing_networks():
    """Compare network properties of working vs failing cases."""
    
    print(f"\nCOMPARING WORKING VS FAILING NETWORKS")
    print("="*60)
    
    networks = {
        'karate_club': (nx.karate_club_graph(), "WORKS"),
        'erdos_992': (nx.erdos_renyi_graph(992, 0.008), "WORKS"),
        'power_grid': (nx.connected_watts_strogatz_graph(100, 4, 0.1), "FAILS"),  # Smaller version
        'internet_as': (nx.barabasi_albert_graph(100, 3), "FAILS")
    }
    
    print(f"{'Network':<15} {'Status':<6} {'Nodes':<6} {'Edges':<6} {'Max Core':<8} {'(k-1) Shell':<10} {'Density':<8}")
    print("-" * 80)
    
    for name, (G, status) in networks.items():
        cores = nx.core_number(G)
        max_core = max(cores.values()) if cores else 0
        k_minus_1_shell = sum(1 for c in cores.values() if c == max_core - 1)
        density = nx.density(G)
        
        print(f"{name:<15} {status:<6} {G.number_of_nodes():<6} {G.number_of_edges():<6} {max_core:<8} {k_minus_1_shell:<10} {density:<8.3f}")
        
        # Quick FastCM+ test
        try:
            G_result, edges = fastcm_plus_reinforce(G, budget=3)
            fastcm_result = f"{len(edges)} edges"
        except:
            fastcm_result = "FAILED"
        
        print(f"                FastCM+ (budget=3): {fastcm_result}")

def investigate_fastcm_algorithm_requirements():
    """Try to understand what FastCM+ actually requires."""
    
    print(f"\nINVESTIGATING FASTCM+ ALGORITHM REQUIREMENTS")
    print("="*60)
    
    # Test on artificially constructed networks
    test_networks = []
    
    # Perfect case: Complete graph with one removed edge
    G1 = nx.complete_graph(10)
    G1.remove_edge(0, 1)
    test_networks.append(("Complete-1edge", G1))
    
    # Ring with additional edges
    G2 = nx.cycle_graph(20)
    for i in range(0, 20, 4):
        G2.add_edge(i, (i+2) % 20)
    test_networks.append(("Enhanced Ring", G2))
    
    # Small world with specific properties
    G3 = nx.watts_strogatz_graph(50, 6, 0.3)
    test_networks.append(("Small World 50", G3))
    
    for name, G in test_networks:
        print(f"\n{name}:")
        cores = nx.core_number(G)
        max_core = max(cores.values()) if cores else 0
        k_minus_1_shell = sum(1 for c in cores.values() if c == max_core - 1)
        
        print(f"  Max core: {max_core}, (k-1)-shell: {k_minus_1_shell}")
        
        if k_minus_1_shell > 0:
            # Test different budgets
            for budget in [1, 2, 3, 4, 5]:
                try:
                    G_result, edges = fastcm_plus_reinforce(G, budget)
                    print(f"    Budget {budget}: {len(edges)} edges")
                    if len(edges) > 0:
                        break
                except Exception as e:
                    print(f"    Budget {budget}: FAILED - {str(e)[:50]}")
        else:
            print(f"  No (k-1)-shell - FastCM+ should fail")

if __name__ == "__main__":
    import time
    
    analyze_power_grid_failure()
    compare_working_vs_failing_networks()
    investigate_fastcm_algorithm_requirements()