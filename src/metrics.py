import networkx as nx
from scipy import stats
import numpy as np

def core_resilience(G_original, G_attacked, method='max_core_ratio', top_percent=50):
    """
    Calculate core resilience using different methods.
    
    Parameters:
        G_original: Original network
        G_attacked: Network after attack
        method: 'max_core_ratio' (default), 'ranking_correlation', or 'avg_core_preservation'
        top_percent: For ranking correlation method only
    
    Returns:
        float: Resilience score (0.0 = total loss, 1.0 = no damage)
    """
    if method == 'max_core_ratio':
        orig_cores = nx.core_number(G_original)
        attack_cores = nx.core_number(G_attacked)
        
        orig_max = max(orig_cores.values()) if orig_cores else 0
        attack_max = max(attack_cores.values()) if attack_cores else 0
        
        return attack_max / orig_max if orig_max > 0 else 1.0
    
    elif method == 'avg_core_preservation':
        orig_cores = nx.core_number(G_original)
        attack_cores = nx.core_number(G_attacked)
        
        common_nodes = set(orig_cores.keys()) & set(attack_cores.keys())
        if not common_nodes:
            return 0.0
        
        total_preservation = 0
        for node in common_nodes:
            orig_core = orig_cores[node]
            attack_core = attack_cores[node]
            preservation = attack_core / orig_core if orig_core > 0 else 1.0
            total_preservation += preservation
        
        return total_preservation / len(common_nodes)
    
    elif method == 'ranking_correlation':
        cores_orig = nx.core_number(G_original)
        cores_attack = nx.core_number(G_attacked)
        
        num_top = max(1, int(top_percent / 100.0 * G_original.number_of_nodes()))
        top_nodes = sorted(cores_orig.items(), key=lambda x: x[1], reverse=True)[:num_top]
        top_node_ids = [node for node, _ in top_nodes]
        
        orig_rankings = []
        attack_rankings = []
        
        for node in top_node_ids:
            orig_rankings.append(cores_orig[node])
            attack_rankings.append(cores_attack.get(node, 0))
        
        if len(set(orig_rankings)) <= 1 or len(set(attack_rankings)) <= 1:
            return 0.0
        
        try:
            correlation, _ = stats.kendalltau(orig_rankings, attack_rankings)
            return correlation if abs(correlation) < 1.0 else 0.0
        except:
            return 0.0
    
    else:
        raise ValueError(f"Unknown method: {method}")


def measure_damage(G_original, G_attacked, resilience_method='max_core_ratio'):
    """
    Measure various types of network damage from attack.
    Returns comprehensive damage metrics used in resilience literature.
    
    Parameters:
        G_original: Original network before attack
        G_attacked: Network after attack
        resilience_method: Method for calculating core resilience
    
    Returns:
        dict: Dictionary containing all damage metrics
    """
    orig_nodes = G_original.number_of_nodes()
    attack_nodes = G_attacked.number_of_nodes()
    nodes_lost = orig_nodes - attack_nodes
    
    orig_cores = nx.core_number(G_original)
    attack_cores = nx.core_number(G_attacked)
    
    orig_max_core = max(orig_cores.values()) if orig_cores else 0
    attack_max_core = max(attack_cores.values()) if attack_cores else 0
    core_damage = orig_max_core - attack_max_core
    
    orig_components = nx.number_connected_components(G_original)
    attack_components = nx.number_connected_components(G_attacked)
    
    if G_attacked.number_of_nodes() == 0:
        largest_component = 0
    elif nx.is_connected(G_attacked):
        largest_component = attack_nodes
    else:
        largest_component = len(max(nx.connected_components(G_attacked), key=len))
    
    resilience = core_resilience(G_original, G_attacked, method=resilience_method)
    
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
        dict: Comparison summary with best/worst performers
    """
    comparison = {}
    
    for condition, damage_info in results_dict.items():
        if metric in damage_info:
            comparison[condition] = damage_info[metric]
    
    if not comparison:
        return {}
    
    best_condition = min(comparison.items(), key=lambda x: x[1])
    worst_condition = max(comparison.items(), key=lambda x: x[1])
    
    return {
        'metric': metric,
        'results': comparison,
        'best': best_condition,
        'worst': worst_condition,
        'improvement': worst_condition[1] - best_condition[1]
    }


def resilience_summary(G_original, attack_results, resilience_method='max_core_ratio'):
    """
    Create a summary of resilience across different attack types.
    
    Parameters:
        G_original: Original network
        attack_results: Dict of {attack_type: G_attacked}
        resilience_method: Method for calculating core resilience
    
    Returns:
        dict: Summary of resilience metrics across all attacks
    """
    summary = {
        'network_size': G_original.number_of_nodes(),
        'original_max_core': max(nx.core_number(G_original).values()) if G_original.nodes() else 0,
        'attacks': {}
    }
    
    for attack_type, G_attacked in attack_results.items():
        damage = measure_damage(G_original, G_attacked, resilience_method=resilience_method)
        summary['attacks'][attack_type] = damage
    
    resilience_scores = [info['core_resilience'] for info in summary['attacks'].values()]
    summary['overall_resilience'] = np.mean(resilience_scores) if resilience_scores else 0.0
    
    return summary


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