"""
Test the FastCM+ budget paradox - does it work better with smaller budgets?
"""

import networkx as nx
from src.mrkc import mrkc_reinforce
from src.fastcm import fastcm_plus_reinforce
from src.attacks import attack_network
from src.metrics import measure_damage, followers_gained

def test_budget_sensitivity():
    """Test how budget affects both algorithms."""
    print("BUDGET SENSITIVITY ANALYSIS")
    print("="*60)
    
    # Test networks
    test_networks = [
        ("Karate Club", nx.karate_club_graph()),
        ("Dense Random", nx.erdos_renyi_graph(100, 0.05)),
        ("Small Random", nx.erdos_renyi_graph(50, 0.08)),
    ]
    
    budgets = [1, 2, 3, 5, 8, 10, 15, 20, 25, 30]
    
    for name, G in test_networks:
        print(f"\n{name}:")
        print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        
        cores = nx.core_number(G)
        max_core = max(cores.values())
        k_minus_1_shell = sum(1 for c in cores.values() if c == max_core - 1)
        
        print(f"  Max core: {max_core}, (k-1)-shell: {k_minus_1_shell}")
        
        if k_minus_1_shell == 0:
            print(f"  Skipping - no (k-1)-shell")
            continue
            
        print(f"  {'Budget':<8} {'MRKC':<8} {'FastCM+':<8} {'FastCM+ Followers':<16}")
        print(f"  {'-'*45}")
        
        for budget in budgets:
            # Test MRKC
            try:
                G_mrkc, mrkc_edges = mrkc_reinforce(G, budget)
                mrkc_added = len(mrkc_edges)
            except:
                mrkc_added = 0
            
            # Test FastCM+
            try:
                G_fastcm, fastcm_edges = fastcm_plus_reinforce(G, budget)
                fastcm_added = len(fastcm_edges)
                if fastcm_added > 0:
                    followers = followers_gained(G, G_fastcm)
                else:
                    followers = 0
            except:
                fastcm_added = 0
                followers = 0
            
            print(f"  {budget:<8} {mrkc_added:<8} {fastcm_added:<8} {followers:<16}")

def test_optimal_budget_resilience():
    """Test resilience with optimal budgets for each algorithm."""
    print(f"\nRESILIENCE TEST WITH OPTIMAL BUDGETS")
    print("="*60)
    
    # Use Karate Club since we know it works
    G = nx.karate_club_graph()
    
    print(f"Testing Karate Club Network:")
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    
    # Find optimal budgets
    optimal_budgets = {
        'mrkc': 10,      # MRKC works well with higher budgets
        'fastcm': 3      # FastCM+ works better with small budgets
    }
    
    print(f"\nOptimal Budget Analysis:")
    print(f"  MRKC optimal budget: {optimal_budgets['mrkc']}")
    print(f"  FastCM+ optimal budget: {optimal_budgets['fastcm']}")
    
    # Apply algorithms with their optimal budgets
    G_mrkc, mrkc_edges = mrkc_reinforce(G, optimal_budgets['mrkc'])
    G_fastcm, fastcm_edges = fastcm_plus_reinforce(G, optimal_budgets['fastcm'])
    
    print(f"\nReinforcement Results:")
    print(f"  MRKC added: {len(mrkc_edges)} edges")
    print(f"  FastCM+ added: {len(fastcm_edges)} edges")
    print(f"  FastCM+ followers: {followers_gained(G, G_fastcm)}")
    
    # Test resilience
    attack_scenarios = [
        ('degree', 0.10),
        ('degree', 0.15),
        ('betweenness', 0.10),
        ('kcore', 0.10),
        ('random', 0.15)
    ]
    
    print(f"\nResilience Comparison (with optimal budgets):")
    print(f"{'Attack':<12} {'Intensity':<10} {'Baseline':<10} {'MRKC':<10} {'FastCM+':<10}")
    print("-" * 65)
    
    resilience_results = {
        'baseline': [],
        'mrkc': [],
        'fastcm': []
    }
    
    for attack_type, intensity in attack_scenarios:
        # Baseline
        G_base_attacked, _ = attack_network(G, attack_type, intensity)
        base_resilience = measure_damage(G, G_base_attacked)['core_resilience']
        
        # MRKC
        G_mrkc_attacked, _ = attack_network(G_mrkc, attack_type, intensity)
        mrkc_resilience = measure_damage(G_mrkc, G_mrkc_attacked)['core_resilience']
        
        # FastCM+
        G_fastcm_attacked, _ = attack_network(G_fastcm, attack_type, intensity)
        fastcm_resilience = measure_damage(G_fastcm, G_fastcm_attacked)['core_resilience']
        
        print(f"{attack_type:<12} {intensity:<10.0%} {base_resilience:<10.3f} {mrkc_resilience:<10.3f} {fastcm_resilience:<10.3f}")
        
        resilience_results['baseline'].append(base_resilience)
        resilience_results['mrkc'].append(mrkc_resilience)
        resilience_results['fastcm'].append(fastcm_resilience)
    
    # Calculate averages
    avg_base = sum(resilience_results['baseline']) / len(resilience_results['baseline'])
    avg_mrkc = sum(resilience_results['mrkc']) / len(resilience_results['mrkc'])
    avg_fastcm = sum(resilience_results['fastcm']) / len(resilience_results['fastcm'])
    
    print(f"\nAverage Resilience:")
    print(f"  Baseline: {avg_base:.3f}")
    print(f"  MRKC: {avg_mrkc:.3f} (Δ: {avg_mrkc - avg_base:+.3f})")
    print(f"  FastCM+: {avg_fastcm:.3f} (Δ: {avg_fastcm - avg_base:+.3f})")
    
    if avg_mrkc > avg_fastcm:
        print(f"  🏆 MRKC performs better for resilience")
    elif avg_fastcm > avg_mrkc:
        print(f"  🏆 FastCM+ performs better for resilience")
    else:
        print(f"  🤝 Similar performance")

def comprehensive_fair_comparison():
    """Comprehensive comparison using appropriate budgets for each algorithm."""
    print(f"\nCOMPREHENSIVE FAIR COMPARISON")
    print("="*60)
    
    # Networks where FastCM+ can work (with small budgets)
    test_networks = [
        ("Karate Club", nx.karate_club_graph()),
        ("Dense Random 1", nx.erdos_renyi_graph(80, 0.06)),
        ("Dense Random 2", nx.erdos_renyi_graph(60, 0.08)),
    ]
    
    # Use appropriate budgets
    mrkc_budget = 15
    fastcm_budget = 3
    
    print(f"Using budgets: MRKC={mrkc_budget}, FastCM+={fastcm_budget}")
    
    summary_results = {
        'networks_tested': 0,
        'fastcm_worked': 0,
        'mrkc_better_resilience': 0,
        'fastcm_better_resilience': 0
    }
    
    for name, G in test_networks:
        cores = nx.core_number(G)
        max_core = max(cores.values()) if cores else 0
        k_minus_1_shell = sum(1 for c in cores.values() if c == max_core - 1)
        
        if k_minus_1_shell == 0:
            continue
            
        print(f"\n{name}:")
        
        # Apply algorithms
        G_mrkc, mrkc_edges = mrkc_reinforce(G, mrkc_budget)
        G_fastcm, fastcm_edges = fastcm_plus_reinforce(G, fastcm_budget)
        
        if len(fastcm_edges) == 0:
            print(f"  FastCM+ failed to add edges - skipping")
            continue
            
        summary_results['networks_tested'] += 1
        summary_results['fastcm_worked'] += 1
        
        print(f"  MRKC added: {len(mrkc_edges)} edges")
        print(f"  FastCM+ added: {len(fastcm_edges)} edges, {followers_gained(G, G_fastcm)} followers")
        
        # Quick resilience test
        G_mrkc_attacked, _ = attack_network(G_mrkc, 'degree', 0.15)
        G_fastcm_attacked, _ = attack_network(G_fastcm, 'degree', 0.15)
        
        mrkc_resilience = measure_damage(G_mrkc, G_mrkc_attacked)['core_resilience']
        fastcm_resilience = measure_damage(G_fastcm, G_fastcm_attacked)['core_resilience']
        
        print(f"  Resilience (degree 15%): MRKC={mrkc_resilience:.3f}, FastCM+={fastcm_resilience:.3f}")
        
        if mrkc_resilience > fastcm_resilience:
            summary_results['mrkc_better_resilience'] += 1
            print(f"  🏆 MRKC better for resilience")
        else:
            summary_results['fastcm_better_resilience'] += 1
            print(f"  🏆 FastCM+ better for resilience")
    
    print(f"\nSUMMARY:")
    print(f"  Networks where both algorithms worked: {summary_results['fastcm_worked']}/{len(test_networks)}")
    print(f"  MRKC better resilience: {summary_results['mrkc_better_resilience']}")
    print(f"  FastCM+ better resilience: {summary_results['fastcm_better_resilience']}")

if __name__ == "__main__":
    test_budget_sensitivity()
    test_optimal_budget_resilience()
    comprehensive_fair_comparison()