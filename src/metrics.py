import networkx as nx
from scipy import stats
import numpy as np

def core_resilience(G_original, G_attacked, top_percent=50):
    """
    Calculate core resilience as defined in the MRKC paper.
    Measures correlation between core number rankings before and after attack.
    
    Parameters:
        G_original: Original network
        G_attacked: Network after attack
        top_percent: Percentage of top nodes to consider (default 50%)
    
    Returns:
        float: Kendall's tau correlation coefficient
    """
    # Get core numbers for both networks
    cores_orig = nx.core_number(G_original)
    cores_attack = nx.core_number(G_attacked)
    
    # Get top nodes by core number in original network
    num_top = max(1, int(top_percent / 100.0 * G_original.number_of_nodes()))
    top_nodes = sorted(cores_orig.items(), key=lambda x: x[1], reverse=True)[:num_top]
    top_node_ids = [node for node, _ in top_nodes]
    
    # Get rankings for these nodes in both networks
    orig_rankings = []
    attack_rankings = []
    
    for node in top_node_ids:
        orig_rankings.append(cores_orig[node])
        # If node was removed in attack, core number = 0
        attack_rankings.append(cores_attack.get(node, 0))
    
    # Calculate Kendall's tau correlation
    if len(set(orig_rankings)) <= 1 or len(set(attack_rankings)) <= 1:
        return 0.0  # No variation to correlate
    
    try:
        correlation, _ = stats.kendalltau(orig_rankings, attack_rankings)
        return correlation if abs(correlation) < 1.0 else 0.0 # type: ignore
    except:
        return 0.0


def measure_damage(G_original, G_attacked):
    """
    Measure various types of network damage from attack.
    Returns comprehensive damage metrics used in resilience literature.
    """
    # Basic size metrics
    orig_nodes = G_original.number_of_nodes()
    attack_nodes = G_attacked.number_of_nodes()
    nodes_lost = orig_nodes - attack_nodes
    
    # Core structure damage
    orig_cores = nx.core_number(G_original)
    attack_cores = nx.core_number(G_attacked)
    
    orig_max_core = max(orig_cores.values()) if orig_cores else 0
    attack_max_core = max(attack_cores.values()) if attack_cores else 0
    core_damage = orig_max_core - attack_max_core
    
    # Connectivity damage
    orig_components = nx.number_connected_components(G_original)
    attack_components = nx.number_connected_components(G_attacked)
    
    # Largest component size (standard resilience metric)
    if G_attacked.number_of_nodes() == 0:
        largest_component = 0
    elif nx.is_connected(G_attacked):
        largest_component = attack_nodes
    else:
        largest_component = len(max(nx.connected_components(G_attacked), key=len))
    
    # Calculate core resilience
    resilience = core_resilience(G_original, G_attacked)
    
    return {
        'nodes_removed': nodes_lost,
        'nodes_remaining': attack_nodes,
        'removal_fraction': nodes_lost / orig_nodes if orig_nodes > 0 else 0,
        'max_core_original': orig_max_core,
        'max_core_attacked': attack_max_core,
        'core_damage': core_damage,
        'core_damage_fraction': core_damage / orig_max_core if orig_max_core > 0 else 0,
        'components_original': orig_components,
        'components_attacked': attack_components,
        'fragmentation': attack_components - orig_components,
        'largest_component': largest_component,
        'largest_component_fraction': largest_component / orig_nodes if orig_nodes > 0 else 0,
        'core_resilience': resilience
    }


def compare_resilience(results_dict, metric='core_damage'):
    """
    Compare resilience results across different algorithms/conditions.
    
    Parameters:
        results_dict: Dict with keys like 'mrkc_degree', 'fastcm_degree', etc.
        metric: Which damage metric to compare
    
    Returns:
        Comparison summary
    """
    comparison = {}
    
    for condition, damage_info in results_dict.items():
        if metric in damage_info:
            comparison[condition] = damage_info[metric]
    
    if not comparison:
        return {}
    
    # Find best and worst
    best_condition = min(comparison.items(), key=lambda x: x[1])
    worst_condition = max(comparison.items(), key=lambda x: x[1])
    
    return {
        'metric': metric,
        'results': comparison,
        'best': best_condition,
        'worst': worst_condition,
        'improvement': worst_condition[1] - best_condition[1]
    }


def resilience_summary(G_original, attack_results):
    """
    Create a summary of resilience across different attack types.
    
    Parameters:
        G_original: Original network
        attack_results: Dict of {attack_type: G_attacked}
    
    Returns:
        Summary of resilience metrics
    """
    summary = {
        'network_size': G_original.number_of_nodes(),
        'original_max_core': max(nx.core_number(G_original).values()) if G_original.nodes() else 0,
        'attacks': {}
    }
    
    for attack_type, G_attacked in attack_results.items():
        damage = measure_damage(G_original, G_attacked)
        summary['attacks'][attack_type] = damage
    
    # Overall resilience score (average core resilience across attacks)
    resilience_scores = [info['core_resilience'] for info in summary['attacks'].values()]
    summary['overall_resilience'] = np.mean(resilience_scores) if resilience_scores else 0.0
    
    return summary


def print_damage_report(damage_info, title="Network Damage Report"):
    """Print a formatted damage report."""
    print(f"\n{title}")
    print("=" * len(title))
    print(f"Nodes removed: {damage_info['nodes_removed']} ({damage_info['removal_fraction']:.1%})")
    print(f"Max k-core: {damage_info['max_core_original']} → {damage_info['max_core_attacked']} "
          f"(damage: {damage_info['core_damage']})")
    print(f"Components: {damage_info['components_original']} → {damage_info['components_attacked']} "
          f"(fragmentation: {damage_info['fragmentation']})")
    print(f"Largest component: {damage_info['largest_component']} nodes "
          f"({damage_info['largest_component_fraction']:.1%})")
    print(f"Core resilience: {damage_info['core_resilience']:.3f}")


def print_resilience_comparison(comparison_results):
    """Print comparison between different algorithms/conditions."""
    print(f"\nResilience Comparison ({comparison_results['metric']})")
    print("-" * 40)
    
    for condition, value in comparison_results['results'].items():
        marker = " ← BEST" if condition == comparison_results['best'][0] else ""
        marker = " ← WORST" if condition == comparison_results['worst'][0] else marker
        print(f"{condition:>15}: {value:.3f}{marker}")
    
    print(f"\nImprovement: {comparison_results['improvement']:.3f} "
          f"({comparison_results['best'][0]} vs {comparison_results['worst'][0]})")


def evaluate_algorithm_resilience(G, reinforce_func, attack_types=['degree', 'kcore'], 
                                 attack_intensity=0.1, budget=50):
    """
    Evaluate an algorithm's resilience against multiple attack types.
    
    Parameters:
        G: Original network
        reinforce_func: Function that takes (G, budget) and returns reinforced network
        attack_types: List of attack types to test
        attack_intensity: Attack intensity (fraction of nodes)
        budget: Reinforcement budget
    
    Returns:
        Dictionary with resilience results
    """
    from attacks import attack_network
    
    # Apply reinforcement
    G_reinforced = reinforce_func(G, budget)
    
    results = {
        'original_network': G,
        'reinforced_network': G_reinforced,
        'budget': budget,
        'attack_results': {}
    }
    
    # Test each attack type
    for attack_type in attack_types:
        G_attacked, removed = attack_network(G_reinforced, attack_type, attack_intensity)
        damage = measure_damage(G_reinforced, G_attacked)
        results['attack_results'][attack_type] = {
            'attacked_network': G_attacked,
            'removed_nodes': removed,
            'damage': damage
        }
    
    return results


def followers_gained(G_original, G_reinforced, k=None):
    """
    FastCM+ paper's key metric - count nodes that became k-core members.
    Essential for comparing k-core maximization vs resilience improvement.
    
    Parameters:
        G_original: Original network before reinforcement
        G_reinforced: Network after reinforcement
        k: Target k-core level (if None, uses max k-core in reinforced graph)
    
    Returns:
        int: Number of nodes that became k-core members
    """
    if k is None:
        reinforced_cores = nx.core_number(G_reinforced)
        k = max(reinforced_cores.values()) if reinforced_cores else 1
    
    orig_cores = nx.core_number(G_original)
    reinforced_cores = nx.core_number(G_reinforced)
    
    followers = 0
    for node in G_original.nodes():
        if orig_cores[node] < k and reinforced_cores[node] >= k:
            followers += 1
    
    return followers


def impact_efficiency(G_original, reinforce_func, attack_type="degree", attack_intensity=0.1, metric='core_damage', budget=10):
    """
    Measures resilience improvement per edge added after applying reinforcement and attack.
    """
    from attacks import attack_network

    # Reinforce the graph
    G_reinforced, _ = reinforce_func(G_original, budget)
    
    # Attack both graphs
    G_orig_attacked, _ = attack_network(G_original, attack_type, attack_intensity)
    G_reinf_attacked, _ = attack_network(G_reinforced, attack_type, attack_intensity)

    # Measure damage
    damage_orig = measure_damage(G_original, G_orig_attacked)
    damage_reinf = measure_damage(G_reinforced, G_reinf_attacked)
    
    # Improvement in damage
    improvement = damage_orig[metric] - damage_reinf[metric]

    # Cost: count edges added
    edges_added = G_reinforced.number_of_edges() - G_original.number_of_edges()
    
    return improvement / edges_added if edges_added > 0 else 0


