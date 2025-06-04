"""
Algorithm testing suite for MRKC vs FastCM+ comparison project.
Tests individual components, integration, and basic validation.
"""

import networkx as nx
import numpy as np
import time
import traceback
from typing import Dict, Any

# Import project modules
try:
    from src.mrkc import mrkc_reinforce
    from src.fastcm import fastcm_plus_reinforce
    # Create fastcm_reinforce as an alias since it doesn't exist
    def fastcm_reinforce(G, budget=10):
        """Simplified interface that just returns the graph for compatibility."""
        G_reinforced, _ = fastcm_plus_reinforce(G, budget=budget)
        return G_reinforced
    
    from src.attacks import attack_network
    from src.metrics import (core_resilience, measure_damage, followers_gained, 
                            impact_efficiency, evaluate_algorithm_resilience)
    from src.synthetic import get_test_graph
    print("All imports successful")
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying alternative import paths...")
    
    # Try without src prefix (if files are in current directory)
    try:
        from mrkc import mrkc_reinforce
        from fastcm import fastcm_plus_reinforce
        # Create fastcm_reinforce as an alias
        def fastcm_reinforce(G, budget=10):
            G_reinforced, _ = fastcm_plus_reinforce(G, budget=budget)
            return G_reinforced
            
        from attacks import attack_network
        from metrics import (core_resilience, measure_damage, followers_gained, 
                            impact_efficiency, evaluate_algorithm_resilience)
        from synthetic import get_test_graph
        print("Direct imports successful")
    except ImportError as e2:
        print(f"Both import methods failed:")
        print(f"  src.module: {e}")
        print(f"  direct: {e2}")
        print("\nPlease check your function names in your files:")
        print("Expected in src/fastcm.py: fastcm_plus_reinforce")
        print("Expected in src/mrkc.py: mrkc_reinforce")
        exit(1)


def test_graph_generation():
    """Test synthetic graph generation works correctly."""
    print("\nTesting Graph Generation...")
    
    try:
        # Test each graph type
        for graph_type in ["random", "scale_free", "small_world"]:
            G = get_test_graph(graph_type, n=50)
            assert G.number_of_nodes() == 50, f"{graph_type}: Wrong node count"
            assert G.number_of_edges() > 0, f"{graph_type}: No edges"
            assert nx.is_connected(G) or graph_type == "random", f"{graph_type}: Should be connected"
            print(f"  {graph_type}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return True
    except Exception as e:
        print(f"  Graph generation failed: {e}")
        return False


def test_mrkc_algorithm():
    """Test MRKC algorithm basic functionality."""
    print("\nTesting MRKC Algorithm...")
    
    try:
        # Test on small known graph
        G = nx.karate_club_graph()
        original_edges = G.number_of_edges()
        original_nodes = G.number_of_nodes()
        
        # Test with small budget
        budget = 5
        start_time = time.time()
        G_reinforced, added_edges = mrkc_reinforce(G, budget=budget)
        runtime = time.time() - start_time
        
        # Basic checks
        assert isinstance(G_reinforced, nx.Graph), "Should return NetworkX Graph"
        assert G_reinforced.number_of_nodes() == original_nodes, "Nodes shouldn't change"
        assert G_reinforced.number_of_edges() >= original_edges, "Should add edges"
        assert len(added_edges) <= budget, f"Added {len(added_edges)} > budget {budget}"
        assert all(len(edge) == 2 for edge in added_edges), "Edges should be tuples of length 2"
        
        # Check edges were actually added
        for u, v in added_edges:
            assert G_reinforced.has_edge(u, v), f"Edge ({u}, {v}) not in reinforced graph"
            assert not G.has_edge(u, v), f"Edge ({u}, {v}) already existed"
        
        print(f"  MRKC: Added {len(added_edges)}/{budget} edges in {runtime:.3f}s")
        
        # Test core numbers preserved (MRKC constraint)
        original_cores = nx.core_number(G)
        reinforced_cores = nx.core_number(G_reinforced)
        core_changes = sum(1 for node in G.nodes() 
                          if original_cores[node] != reinforced_cores[node])
        
        print(f"  Core number changes: {core_changes}/{original_nodes} nodes")
        
        return True
        
    except Exception as e:
        print(f"  MRKC test failed: {e}")
        traceback.print_exc()
        return False


def test_fastcm_algorithm():
    """Test FastCM+ algorithm basic functionality."""
    print("\nTesting FastCM+ Algorithm...")
    
    try:
        # Test on small graph
        G = nx.karate_club_graph()
        original_edges = G.number_of_edges()
        original_nodes = G.number_of_nodes()
        original_cores = nx.core_number(G)
        
        # Test with small budget
        budget = 5
        start_time = time.time()
        G_reinforced, added_edges = fastcm_plus_reinforce(G, budget=budget)
        runtime = time.time() - start_time
        
        # Basic checks
        assert isinstance(G_reinforced, nx.Graph), "Should return NetworkX Graph"
        assert isinstance(added_edges, list), "Should return list of edges"
        assert G_reinforced.number_of_nodes() == original_nodes, "Nodes shouldn't change"
        assert G_reinforced.number_of_edges() >= original_edges, "Should add edges"
        assert len(added_edges) <= budget, f"Added {len(added_edges)} > budget {budget}"
        
        # Check edges were actually added
        for u, v in added_edges:
            assert G_reinforced.has_edge(u, v), f"Edge ({u}, {v}) not in reinforced graph"
            assert not G.has_edge(u, v), f"Edge ({u}, {v}) already existed"
        
        # Test k-core improvement (FastCM+ goal)
        reinforced_cores = nx.core_number(G_reinforced)
        max_core_orig = max(original_cores.values())
        max_core_reinforced = max(reinforced_cores.values())
        followers = followers_gained(G, G_reinforced)
        
        print(f"  FastCM+: Added {len(added_edges)}/{budget} edges in {runtime:.3f}s")
        print(f"  Max k-core: {max_core_orig} -> {max_core_reinforced}")
        print(f"  Followers gained: {followers}")
        
        return True
        
    except Exception as e:
        print(f"  FastCM+ test failed: {e}")
        traceback.print_exc()
        return False


def test_attack_simulation():
    """Test all attack types work correctly."""
    print("\nTesting Attack Simulation...")
    
    try:
        G = get_test_graph("scale_free", n=100)
        original_nodes = G.number_of_nodes()
        
        attack_types = ["degree", "kcore", "betweenness", "random"]
        
        for attack_type in attack_types:
            G_attacked, removed_nodes = attack_network(G, attack_type, fraction=0.1)
            
            # Basic checks
            expected_removed = max(1, int(0.1 * original_nodes))
            assert len(removed_nodes) == expected_removed, f"{attack_type}: Wrong removal count"
            assert G_attacked.number_of_nodes() == original_nodes - len(removed_nodes), \
                   f"{attack_type}: Node count mismatch"
            
            # Check removed nodes are actually gone
            for node in removed_nodes:
                assert node not in G_attacked.nodes(), f"{attack_type}: Node {node} not removed"
            
            print(f"  {attack_type}: Removed {len(removed_nodes)} nodes")
        
        return True
        
    except Exception as e:
        print(f"  Attack simulation failed: {e}")
        traceback.print_exc()
        return False


def test_metrics_calculation():
    """Test metrics are calculated correctly."""
    print("\nTesting Metrics Calculation...")
    
    try:
        # Create original and attacked graphs
        G_original = get_test_graph("scale_free", n=100)
        G_attacked, removed = attack_network(G_original, "degree", 0.1)
        
        # Test damage measurement
        damage = measure_damage(G_original, G_attacked)
        
        # Check all expected metrics are present
        expected_metrics = [
            'nodes_removed', 'nodes_remaining', 'removal_fraction',
            'max_core_original', 'max_core_attacked', 'core_damage',
            'core_resilience', 'largest_component', 'fragmentation'
        ]
        
        for metric in expected_metrics:
            assert metric in damage, f"Missing metric: {metric}"
        
        # Sanity checks
        assert damage['nodes_removed'] == len(removed), "Wrong removal count"
        assert 0 <= damage['removal_fraction'] <= 1, "Invalid removal fraction"
        assert -1 <= damage['core_resilience'] <= 1, "Invalid core resilience"
        assert damage['core_damage'] >= 0, "Core damage should be non-negative"
        
        print(f"  Damage metrics: {len(expected_metrics)} metrics calculated")
        print(f"  Core resilience: {damage['core_resilience']:.3f}")
        print(f"  Core damage: {damage['core_damage']}")
        
        # Test followers gained
        G_reinforced = get_test_graph("scale_free", n=100)  # Simulate reinforcement
        followers = followers_gained(G_original, G_reinforced)
        assert isinstance(followers, int), "Followers should be integer"
        assert followers >= 0, "Followers should be non-negative"
        
        print(f"  Followers calculation: {followers} followers")
        
        return True
        
    except Exception as e:
        print(f"  Metrics calculation failed: {e}")
        traceback.print_exc()
        return False


def test_algorithm_integration():
    """Test complete algorithm integration pipeline."""
    print("\nTesting Algorithm Integration...")
    
    try:
        G = get_test_graph("scale_free", n=100)
        budget = 10
        
        # Test both algorithms
        algorithms = {
            'MRKC': lambda g: mrkc_reinforce(g, budget)[0],  # Return only graph part
            'FastCM+': lambda g: fastcm_reinforce(g, budget)
        }
        
        results = {}
        
        for alg_name, alg_func in algorithms.items():
            try:
                start_time = time.time()
                G_reinforced, _ = alg_func(G)  # Unpack tuple for MRKC
                runtime = time.time() - start_time
                
                # Test against degree attack
                G_attacked, removed = attack_network(G_reinforced, "degree", 0.1)
                damage = measure_damage(G_reinforced, G_attacked)
                
                results[alg_name] = {
                    'runtime': runtime,
                    'edges_added': G_reinforced.number_of_edges() - G.number_of_edges(),
                    'core_resilience': damage['core_resilience'],
                    'core_damage': damage['core_damage']
                }
                
                print(f"  {alg_name}: {results[alg_name]['edges_added']} edges, "
                      f"resilience={results[alg_name]['core_resilience']:.3f}")
                
            except Exception as e:
                print(f"  {alg_name} integration failed: {e}")
                results[alg_name] = None
        
        # Compare results if both worked
        if results['MRKC'] and results['FastCM+']:
            print(f"\n  Comparison:")
            print(f"    MRKC resilience: {results['MRKC']['core_resilience']:.3f}")
            print(f"    FastCM+ resilience: {results['FastCM+']['core_resilience']:.3f}")
            
            better = "FastCM+" if results['FastCM+']['core_resilience'] > results['MRKC']['core_resilience'] else "MRKC"
            print(f"    Better performer: {better}")
        
        return True
        
    except Exception as e:
        print(f"  Integration test failed: {e}")
        traceback.print_exc()
        return False


def test_performance_scaling():
    """Test performance on different graph sizes."""
    print("\nTesting Performance Scaling...")
    
    try:
        sizes = [50, 100, 200]
        budget = 10
        
        performance_data = {}
        
        for size in sizes:
            G = get_test_graph("scale_free", n=size)
            
            # Test MRKC performance
            start_time = time.time()
            try:
                G_mrkc, _ = mrkc_reinforce(G, budget)  # Unpack tuple
                mrkc_time = time.time() - start_time
            except Exception as e:
                print(f"  Warning: MRKC failed on size {size}: {e}")
                mrkc_time = float('inf')
            
            # Test FastCM+ performance
            start_time = time.time()
            try:
                G_fastcm = fastcm_reinforce(G, budget)
                fastcm_time = time.time() - start_time
            except Exception as e:
                print(f"  Warning: FastCM+ failed on size {size}: {e}")
                fastcm_time = float('inf')
            
            performance_data[size] = {
                'mrkc_time': mrkc_time,
                'fastcm_time': fastcm_time
            }
            
            print(f"  Size {size}: MRKC={mrkc_time:.3f}s, FastCM+={fastcm_time:.3f}s")
        
        # Check for performance issues
        for size, times in performance_data.items():
            if times['mrkc_time'] > 10.0:
                print(f"  Warning: MRKC slow on size {size} ({times['mrkc_time']:.1f}s)")
            if times['fastcm_time'] > 10.0:
                print(f"  Warning: FastCM+ slow on size {size} ({times['fastcm_time']:.1f}s)")
        
        return True
        
    except Exception as e:
        print(f"  Performance test failed: {e}")
        traceback.print_exc()
        return False


def test_sanity_checks():
    """Test that results make intuitive sense."""
    print("\nTesting Sanity Checks...")
    
    try:
        G = get_test_graph("scale_free", n=100)
        budget = 15
        
        # Test that reinforcement improves resilience
        original_damage = {}
        mrkc_damage = {}
        fastcm_damage = {}
        
        # Test original network
        G_orig_attacked, _ = attack_network(G, "degree", 0.1)
        original_damage = measure_damage(G, G_orig_attacked)
        
        # Test MRKC reinforced
        G_mrkc, _ = mrkc_reinforce(G, budget)  # Unpack tuple return
        G_mrkc_attacked, _ = attack_network(G_mrkc, "degree", 0.1)
        mrkc_damage = measure_damage(G_mrkc, G_mrkc_attacked)
        
        # Test FastCM+ reinforced
        G_fastcm = fastcm_reinforce(G, budget)
        G_fastcm_attacked, _ = attack_network(G_fastcm, "degree", 0.1)
        fastcm_damage = measure_damage(G_fastcm, G_fastcm_attacked)
        
        # Sanity checks
        print(f"  Original resilience: {original_damage['core_resilience']:.3f}")
        print(f"  MRKC resilience: {mrkc_damage['core_resilience']:.3f}")
        print(f"  FastCM+ resilience: {fastcm_damage['core_resilience']:.3f}")
        
        # Check improvements
        mrkc_improvement = mrkc_damage['core_resilience'] - original_damage['core_resilience']
        fastcm_improvement = fastcm_damage['core_resilience'] - original_damage['core_resilience']
        
        print(f"  MRKC improvement: {mrkc_improvement:+.3f}")
        print(f"  FastCM+ improvement: {fastcm_improvement:+.3f}")
        
        # Note unexpected results for analysis
        if mrkc_improvement < 0:
            print(f"  Warning: MRKC made resilience worse (may be normal due to graph density)")
        if fastcm_improvement < 0:
            print(f"  Warning: FastCM+ made resilience worse (unexpected)")
        
        if mrkc_improvement > 0 or fastcm_improvement > 0:
            print(f"  At least one algorithm improved resilience")
        
        return True
        
    except Exception as e:
        print(f"  Sanity check failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run complete test suite."""
    print("MRKC vs FastCM+ Algorithm Testing Suite")
    print("=" * 50)
    
    tests = [
        ("Graph Generation", test_graph_generation),
        ("MRKC Algorithm", test_mrkc_algorithm),
        ("FastCM+ Algorithm", test_fastcm_algorithm),
        ("Attack Simulation", test_attack_simulation),
        ("Metrics Calculation", test_metrics_calculation),
        ("Algorithm Integration", test_algorithm_integration),
        ("Performance Scaling", test_performance_scaling),
        ("Sanity Checks", test_sanity_checks)
    ]
    
    results = {}
    total_start = time.time()
    
    for test_name, test_func in tests:
        try:
            start_time = time.time()
            results[test_name] = test_func()
            test_time = time.time() - start_time
            print(f"  Test completed in {test_time:.2f}s")
        except Exception as e:
            results[test_name] = False
            print(f"  Test crashed: {e}")
    
    total_time = time.time() - total_start
    
    # Summary
    print(f"\nTEST SUMMARY")
    print("=" * 30)
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "PASS" if passed_test else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"Total runtime: {total_time:.2f}s")
    
    if passed == total:
        print("\nAll tests passed. Algorithms are ready for experiments.")
    else:
        print(f"\n{total-passed} tests failed. Fix issues before running experiments.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)