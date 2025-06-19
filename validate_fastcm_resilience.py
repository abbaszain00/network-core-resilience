"""
Specialized test to validate the finding that FastCM+ may hurt resilience.
This test isolates the resilience impact with controlled experiments.
"""

import networkx as nx
import numpy as np
from src.mrkc import mrkc_reinforce
from src.fastcm import fastcm_plus_reinforce
from src.attacks import attack_network
from src.metrics import measure_damage, followers_gained

def analyze_network_structure(G, name="Network"):
    """Analyze network structure comprehensively."""
    cores = nx.core_number(G)
    max_core = max(cores.values()) if cores else 0
    
    # Shell distribution
    shell_dist = {}
    for k in range(max_core + 1):
        shell_dist[k] = sum(1 for c in cores.values() if c == k)
    
    print(f"\n{name} Analysis:")
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"  Max k-core: {max_core}")
    print(f"  Shell distribution: {shell_dist}")
    print(f"  (k-1)-shell size: {shell_dist.get(max_core-1, 0)}")
    
    return {
        'max_core': max_core,
        'shell_dist': shell_dist,
        'k_minus_1_shell': shell_dist.get(max_core-1, 0)
    }

def detailed_resilience_test(G, name="Test Network"):
    """Detailed resilience test with comprehensive analysis."""
    print(f"\n{'='*80}")
    print(f"DETAILED RESILIENCE TEST: {name}")
    print(f"{'='*80}")
    
    # Analyze original network
    orig_analysis = analyze_network_structure(G, "Original")
    
    if orig_analysis['k_minus_1_shell'] == 0:
        print(f"⚠️  FastCM+ cannot work on this network (no (k-1)-shell)")
        return None
    
    budget = 20
    
    # Apply algorithms
    print(f"\nApplying Reinforcements (budget={budget}):")
    G_mrkc, mrkc_edges = mrkc_reinforce(G, budget)
    G_fastcm, fastcm_edges = fastcm_plus_reinforce(G, budget)
    
    print(f"  MRKC added: {len(mrkc_edges)} edges")
    print(f"  FastCM+ added: {len(fastcm_edges)} edges")
    
    if len(fastcm_edges) == 0:
        print(f"  FastCM+ failed to add edges - skipping resilience test")
        return None
    
    # Analyze reinforced networks
    mrkc_analysis = analyze_network_structure(G_mrkc, "MRKC Reinforced")
    fastcm_analysis = analyze_network_structure(G_fastcm, "FastCM+ Reinforced")
    
    # FastCM+ specific metrics
    followers = followers_gained(G, G_fastcm)
    print(f"  FastCM+ followers gained: {followers}")
    
    # Core number changes
    orig_cores = nx.core_number(G)
    mrkc_cores = nx.core_number(G_mrkc)
    fastcm_cores = nx.core_number(G_fastcm)
    
    mrkc_core_changes = sum(1 for n in G.nodes() if orig_cores[n] != mrkc_cores[n])
    fastcm_core_changes = sum(1 for n in G.nodes() if orig_cores[n] != fastcm_cores[n])
    
    print(f"  MRKC core number changes: {mrkc_core_changes}/{G.number_of_nodes()} nodes")
    print(f"  FastCM+ core number changes: {fastcm_core_changes}/{G.number_of_nodes()} nodes")
    
    # Test resilience under multiple attack scenarios
    attack_scenarios = [
        ('degree', 0.10),
        ('degree', 0.15), 
        ('degree', 0.20),
        ('betweenness', 0.10),
        ('betweenness', 0.15),
        ('kcore', 0.10),
        ('kcore', 0.15),
        ('random', 0.15),
    ]
    
    results = {
        'network_name': name,
        'original_max_core': orig_analysis['max_core'],
        'mrkc_edges_added': len(mrkc_edges),
        'fastcm_edges_added': len(fastcm_edges),
        'fastcm_followers': followers,
        'baseline': {},
        'mrkc': {},
        'fastcm': {},
        'improvements': {'mrkc': {}, 'fastcm': {}}
    }
    
    print(f"\nTesting Resilience Under Attacks:")
    print(f"{'Attack':<12} {'Intensity':<10} {'Baseline':<10} {'MRKC':<10} {'FastCM+':<10} {'MRKC Δ':<10} {'FastCM+ Δ':<10}")
    print("-" * 90)
    
    for attack_type, intensity in attack_scenarios:
        # Test baseline (original network)
        G_base_attacked, _ = attack_network(G, attack_type, intensity)
        base_damage = measure_damage(G, G_base_attacked)
        base_resilience = base_damage['core_resilience']
        
        # Test MRKC reinforced
        G_mrkc_attacked, _ = attack_network(G_mrkc, attack_type, intensity)
        mrkc_damage = measure_damage(G_mrkc, G_mrkc_attacked)
        mrkc_resilience = mrkc_damage['core_resilience']
        
        # Test FastCM+ reinforced
        G_fastcm_attacked, _ = attack_network(G_fastcm, attack_type, intensity)
        fastcm_damage = measure_damage(G_fastcm, G_fastcm_attacked)
        fastcm_resilience = fastcm_damage['core_resilience']
        
        # Calculate improvements
        mrkc_improvement = mrkc_resilience - base_resilience
        fastcm_improvement = fastcm_resilience - base_resilience
        
        print(f"{attack_type:<12} {intensity:<10.0%} {base_resilience:<10.3f} {mrkc_resilience:<10.3f} {fastcm_resilience:<10.3f} {mrkc_improvement:<10.3f} {fastcm_improvement:<10.3f}")
        
        # Store results
        key = f"{attack_type}_{intensity}"
        results['baseline'][key] = base_resilience
        results['mrkc'][key] = mrkc_resilience
        results['fastcm'][key] = fastcm_resilience
        results['improvements']['mrkc'][key] = mrkc_improvement
        results['improvements']['fastcm'][key] = fastcm_improvement
    
    # Summary statistics
    mrkc_improvements = list(results['improvements']['mrkc'].values())
    fastcm_improvements = list(results['improvements']['fastcm'].values())
    
    avg_mrkc = np.mean(mrkc_improvements)
    avg_fastcm = np.mean(fastcm_improvements)
    std_mrkc = np.std(mrkc_improvements)
    std_fastcm = np.std(fastcm_improvements)
    
    print(f"\nSummary Statistics:")
    print(f"  MRKC average improvement: {avg_mrkc:+.3f} ± {std_mrkc:.3f}")
    print(f"  FastCM+ average improvement: {avg_fastcm:+.3f} ± {std_fastcm:.3f}")
    
    # Statistical significance (simple t-test approximation)
    if len(mrkc_improvements) > 1:
        diff_mean = avg_mrkc - avg_fastcm
        pooled_std = np.sqrt((std_mrkc**2 + std_fastcm**2) / 2)
        if pooled_std > 0:
            t_stat = diff_mean / (pooled_std * np.sqrt(2/len(mrkc_improvements)))
            print(f"  Difference: {diff_mean:+.3f} (t-stat: {t_stat:.2f})")
        
    # Interpretation
    print(f"\nInterpretation:")
    if avg_fastcm < -0.05:
        print(f"  🚨 FastCM+ SIGNIFICANTLY HURTS resilience (avg: {avg_fastcm:+.3f})")
    elif avg_fastcm < 0:
        print(f"  ⚠️  FastCM+ slightly hurts resilience (avg: {avg_fastcm:+.3f})")
    elif avg_fastcm > 0.05:
        print(f"  ✅ FastCM+ improves resilience (avg: {avg_fastcm:+.3f})")
    else:
        print(f"  ➖ FastCM+ has minimal impact on resilience (avg: {avg_fastcm:+.3f})")
        
    if avg_mrkc > avg_fastcm:
        print(f"  🏆 MRKC is better for resilience")
    else:
        print(f"  🏆 FastCM+ is better for resilience")
    
    return results

def test_hypothesis_on_multiple_networks():
    """Test the FastCM+ resilience hypothesis on multiple suitable networks."""
    print(f"\n{'='*100}")
    print(f"HYPOTHESIS TEST: Does FastCM+ hurt resilience compared to MRKC?")
    print(f"{'='*100}")
    
    # Networks where FastCM+ should be able to work
    test_networks = [
        ("Karate Club", nx.karate_club_graph()),
        ("Dense Random 1", nx.erdos_renyi_graph(100, 0.06)),
        ("Dense Random 2", nx.erdos_renyi_graph(150, 0.04)),
        ("Small World 1", nx.watts_strogatz_graph(100, 8, 0.3)),
        ("Small World 2", nx.watts_strogatz_graph(120, 6, 0.4)),
        ("Community Network", nx.connected_caveman_graph(8, 12)),
    ]
    
    all_results = []
    valid_tests = 0
    
    for name, G in test_networks:
        result = detailed_resilience_test(G, name)
        if result is not None:
            all_results.append(result)
            valid_tests += 1
    
    if valid_tests == 0:
        print(f"\n❌ No networks suitable for FastCM+ testing!")
        return
    
    # Aggregate analysis across all networks
    print(f"\n{'='*100}")
    print(f"AGGREGATE ANALYSIS ACROSS {valid_tests} NETWORKS")
    print(f"{'='*100}")
    
    all_mrkc_improvements = []
    all_fastcm_improvements = []
    
    for result in all_results:
        all_mrkc_improvements.extend(result['improvements']['mrkc'].values())
        all_fastcm_improvements.extend(result['improvements']['fastcm'].values())
    
    # Overall statistics
    overall_mrkc = np.mean(all_mrkc_improvements)
    overall_fastcm = np.mean(all_fastcm_improvements)
    overall_mrkc_std = np.std(all_mrkc_improvements)
    overall_fastcm_std = np.std(all_fastcm_improvements)
    
    print(f"\nOverall Results Across All Networks and Attacks:")
    print(f"  MRKC: {overall_mrkc:+.3f} ± {overall_mrkc_std:.3f} (n={len(all_mrkc_improvements)})")
    print(f"  FastCM+: {overall_fastcm:+.3f} ± {overall_fastcm_std:.3f} (n={len(all_fastcm_improvements)})")
    print(f"  Difference: {overall_mrkc - overall_fastcm:+.3f}")
    
    # Count positive vs negative improvements
    mrkc_positive = sum(1 for x in all_mrkc_improvements if x > 0)
    mrkc_negative = sum(1 for x in all_mrkc_improvements if x < 0)
    fastcm_positive = sum(1 for x in all_fastcm_improvements if x > 0)
    fastcm_negative = sum(1 for x in all_fastcm_improvements if x < 0)
    
    print(f"\nImprovement Direction Analysis:")
    print(f"  MRKC: {mrkc_positive} positive, {mrkc_negative} negative ({mrkc_positive/(mrkc_positive+mrkc_negative)*100:.1f}% positive)")
    print(f"  FastCM+: {fastcm_positive} positive, {fastcm_negative} negative ({fastcm_positive/(fastcm_positive+fastcm_negative)*100:.1f}% positive)")
    
    # Final hypothesis test result
    print(f"\n🔬 HYPOTHESIS TEST RESULT:")
    if overall_fastcm < -0.02:  # More than 2% average decrease
        print(f"  📊 CONFIRMED: FastCM+ significantly hurts resilience (avg: {overall_fastcm:+.3f})")
        print(f"  📝 Conclusion: FastCM+ optimizes for k-core size at the expense of resilience")
    elif overall_fastcm < 0:
        print(f"  📊 PARTIALLY CONFIRMED: FastCM+ tends to hurt resilience (avg: {overall_fastcm:+.3f})")
        print(f"  📝 Conclusion: Trade-off exists between k-core growth and resilience")
    else:
        print(f"  📊 HYPOTHESIS REJECTED: FastCM+ does not consistently hurt resilience")
        print(f"  📝 Conclusion: No clear resilience penalty from FastCM+")
    
    if overall_mrkc > overall_fastcm + 0.01:
        print(f"  🏆 MRKC is consistently better for resilience")
    elif overall_fastcm > overall_mrkc + 0.01:
        print(f"  🏆 FastCM+ is actually better for resilience")
    else:
        print(f"  🤝 Both algorithms have similar resilience impact")
    
    return all_results

def quick_sanity_check():
    """Quick sanity check to verify our findings make sense."""
    print(f"\n{'='*60}")
    print(f"SANITY CHECK: Why might FastCM+ hurt resilience?")
    print(f"{'='*60}")
    
    # Create a simple test case
    G = nx.karate_club_graph()
    
    print(f"\nTest on Karate Club Network:")
    orig_cores = nx.core_number(G)
    orig_max = max(orig_cores.values())
    
    # Apply FastCM+
    G_fastcm, fastcm_edges = fastcm_plus_reinforce(G, budget=10)
    fastcm_cores = nx.core_number(G_fastcm)
    fastcm_max = max(fastcm_cores.values())
    followers = followers_gained(G, G_fastcm)
    
    print(f"  Original max k-core: {orig_max}")
    print(f"  FastCM+ max k-core: {fastcm_max}")
    print(f"  Followers gained: {followers}")
    print(f"  Edges added: {len(fastcm_edges)}")
    
    # Analyze what happened
    nodes_upgraded = sum(1 for n in G.nodes() if fastcm_cores[n] > orig_cores[n])
    
    print(f"\nStructural Changes:")
    print(f"  Nodes with increased core number: {nodes_upgraded}")
    print(f"  Core number distribution change:")
    
    for k in range(max(orig_max, fastcm_max) + 1):
        orig_count = sum(1 for c in orig_cores.values() if c == k)
        fastcm_count = sum(1 for c in fastcm_cores.values() if c == k)
        change = fastcm_count - orig_count
        if orig_count > 0 or fastcm_count > 0:
            print(f"    {k}-core: {orig_count} → {fastcm_count} ({change:+d})")
    
    print(f"\nPossible Explanation:")
    print(f"  FastCM+ creates larger, but potentially less stable k-cores")
    print(f"  Many nodes are 'just barely' in higher k-cores")
    print(f"  Under attack, these marginal nodes drop out quickly")
    print(f"  Result: Greater resilience loss despite larger initial k-core")

if __name__ == "__main__":
    # Run comprehensive validation
    quick_sanity_check()
    test_hypothesis_on_multiple_networks()