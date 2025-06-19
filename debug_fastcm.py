"""
Deep debugging of FastCM+ to understand why it's not adding edges.
"""

import networkx as nx
from src.fastcm import fastcm_plus_reinforce

def debug_fastcm_step_by_step():
    """Step-by-step debugging of FastCM+ algorithm."""
    print("DEEP FASTCM+ DEBUGGING")
    print("="*50)
    
    # Test on network that should work
    G = nx.erdos_renyi_graph(100, 0.05)  # Dense random
    cores = nx.core_number(G)
    max_core = max(cores.values())
    
    print(f"Test Network:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Max k-core: {max_core}")
    
    # Check shell distribution
    shell_dist = {}
    for k in range(max_core + 1):
        shell_dist[k] = [n for n, c in cores.items() if c == k]
    
    print(f"  Shell sizes: {[(k, len(nodes)) for k, nodes in shell_dist.items()]}")
    
    k_minus_1_shell = shell_dist.get(max_core - 1, [])
    print(f"  (k-1)-shell has {len(k_minus_1_shell)} nodes: {k_minus_1_shell[:10]}...")
    
    if len(k_minus_1_shell) == 0:
        print("  No (k-1)-shell - FastCM+ should fail")
        return
    
    print(f"\nCalling FastCM+...")
    try:
        G_result, edges = fastcm_plus_reinforce(G, budget=10)
        print(f"  FastCM+ returned: {len(edges)} edges")
        if len(edges) > 0:
            print(f"  Edges: {edges}")
        else:
            print("  No edges added - investigating why...")
            
            # Manual check: what edges should be possible?
            print(f"\nManual Analysis:")
            print(f"  Could add edges between (k-1)-shell nodes?")
            
            possible_edges = 0
            for i, u in enumerate(k_minus_1_shell[:5]):  # Check first 5
                for v in k_minus_1_shell[i+1:]:
                    if not G.has_edge(u, v):
                        possible_edges += 1
                        if possible_edges <= 3:  # Show first few
                            print(f"    Possible edge: ({u}, {v})")
            
            print(f"  Total possible edges in (k-1)-shell: {possible_edges}")
            
    except Exception as e:
        print(f"  FastCM+ failed with error: {e}")
        import traceback
        traceback.print_exc()

def test_fastcm_on_known_working_case():
    """Test FastCM+ on the one case we know worked."""
    print(f"\nTesting on Known Working Case:")
    print("-" * 40)
    
    # From your earlier debug output, this worked:
    G = nx.karate_club_graph()
    
    print(f"Karate Club (known to work with small budget):")
    
    for budget in [3, 5, 10, 15, 20]:
        G_result, edges = fastcm_plus_reinforce(G, budget)
        print(f"  Budget {budget:2d}: {len(edges)} edges added")
        
        if len(edges) == 0 and budget >= 10:
            print(f"    ⚠️  Failed with larger budget!")

if __name__ == "__main__":
    debug_fastcm_step_by_step()
    test_fastcm_on_known_working_case()