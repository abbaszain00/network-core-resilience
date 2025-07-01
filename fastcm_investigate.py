#!/usr/bin/env python3
"""
Analysis of FastCM+ performance patterns across different networks.

Abbas Zain-Ul-Abidin (K21067382)
"""

import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path

import sys
sys.path.append('src')
from shared_networks import get_consistent_networks

def analyze_fastcm_failure_patterns():
    """Analyze FastCM+ performance across network types."""
    
    print("FASTCM+ PERFORMANCE ANALYSIS")
    print("=" * 30)
    
    # Load results
    try:
        df = pd.read_csv('final_evaluation/evaluation_results.csv')
        print("Loaded evaluation results")
    except:
        print("Could not load results file")
        return
    
    # Load networks
    try:
        networks = get_consistent_networks(max_network_size=5000)
        print(f"Loaded {len(networks)} networks")
    except:
        print("Could not load networks")
        return
    
    # Focus on FastCM+ data
    fastcm_data = df[df['algorithm'] == 'FastCM+']
    
    print(f"\nNetwork Analysis")
    print("-" * 15)
    
    network_analysis = []
    
    for network_name in fastcm_data['network'].unique():
        if network_name not in networks:
            continue
            
        G = networks[network_name]
        net_data = fastcm_data[fastcm_data['network'] == network_name]
        
        # Basic properties
        cores = nx.core_number(G)
        max_core = max(cores.values()) if cores else 0
        
        # FastCM+ performance
        success_rate = (net_data['edges_added'] > 0).mean()
        avg_edges = net_data['edges_added'].mean()
        avg_followers = net_data['followers_gained'].mean()
        
        # Network structure
        density = nx.density(G)
        clustering = nx.average_clustering(G)
        
        # Shell sizes
        shell_sizes = {}
        for k in range(max_core + 1):
            shell_nodes = [n for n, c in cores.items() if c == k]
            shell_sizes[k] = len(shell_nodes)
        
        # Degree stats
        degrees = dict(G.degree())
        degree_std = np.std(list(degrees.values()))
        degree_mean = np.mean(list(degrees.values()))
        degree_cv = degree_std / degree_mean if degree_mean > 0 else 0
        
        network_analysis.append({
            'network': network_name,
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': density,
            'clustering': clustering,
            'max_core': max_core,
            'degree_cv': degree_cv,
            'success_rate': success_rate,
            'avg_edges': avg_edges,
            'followers': avg_followers,
            'shell_1_size': shell_sizes.get(1, 0),
            'shell_max_size': shell_sizes.get(max_core, 0) if max_core > 0 else 0
        })
    
    analysis_df = pd.DataFrame(network_analysis)
    
    print(f"Network characteristics vs FastCM+ performance:")
    print(analysis_df[['network', 'nodes', 'density', 'max_core', 'success_rate']].round(3))
    
    print(f"\nPerformance Categories")
    print("-" * 20)
    
    # Group by performance
    high_success = analysis_df[analysis_df['success_rate'] > 0.8]
    low_success = analysis_df[analysis_df['success_rate'] < 0.2]
    
    print(f"High success networks (>80%): {len(high_success)}")
    if len(high_success) > 0:
        print(f"  Average density: {high_success['density'].mean():.3f}")
        print(f"  Average max k-core: {high_success['max_core'].mean():.1f}")
        print(f"  Networks: {list(high_success['network'])}")
    
    print(f"\nLow success networks (<20%): {len(low_success)}")
    if len(low_success) > 0:
        print(f"  Average density: {low_success['density'].mean():.3f}")
        print(f"  Average max k-core: {low_success['max_core'].mean():.1f}")
        print(f"  Networks: {list(low_success['network'])}")
    
    print(f"\nShell Structure Analysis")
    print("-" * 22)
    
    # Check shell structures
    for _, row in analysis_df.iterrows():
        network_name = row['network']
        success_rate = row['success_rate']
        
        if network_name not in networks:
            continue
            
        G = networks[network_name]
        cores = nx.core_number(G)
        max_core = max(cores.values()) if cores else 0
        
        # Check shell sizes for different k values
        shell_info = []
        for k in range(2, min(max_core + 2, 6)):
            k_minus_1_shell = [n for n, c in cores.items() if c == k-1]
            shell_info.append(f"k={k}: {len(k_minus_1_shell)} nodes")
        
        status = "FAIL" if success_rate < 0.2 else "PASS" if success_rate > 0.8 else "OK"
        print(f"{network_name:15} [{status}]: {', '.join(shell_info)}")
    
    print(f"\nAlgorithm Requirements")
    print("-" * 20)
    
    print("FastCM+ needs:")
    print("- Non-empty (k-1)-shell")
    print("- Connected shell components")
    print("- Nodes with specific degree properties")
    print("- Sufficient budget for conversions")
    
    # Check requirements for failed networks
    for network_name in low_success['network'] if len(low_success) > 0 else []:
        if network_name not in networks:
            continue
            
        G = networks[network_name]
        cores = nx.core_number(G)
        max_core = max(cores.values()) if cores else 0
        
        print(f"\n{network_name}:")
        
        for k in range(2, min(max_core + 2, 4)):
            shell_nodes = [n for n, c in cores.items() if c == k-1]
            if len(shell_nodes) == 0:
                print(f"  Empty ({k-1})-shell for k={k}")
                continue
            
            # Check shell connectivity
            shell_subgraph = G.subgraph(shell_nodes)
            components = list(nx.connected_components(shell_subgraph))
            
            print(f"  k={k}: {len(shell_nodes)} nodes in {len(components)} components")
            
            # Check collapse nodes
            k_minus_1_core = [n for n, c in cores.items() if c >= k-1]
            collapse_nodes = []
            for node in shell_nodes:
                neighbors_in_core = [n for n in G.neighbors(node) if n in k_minus_1_core]
                if len(neighbors_in_core) == k-1:
                    collapse_nodes.append(node)
            
            print(f"    Collapse nodes: {len(collapse_nodes)}")
            
            if len(collapse_nodes) == 0:
                print(f"    No collapse nodes available")
    
    print(f"\nCorrelation Analysis")
    print("-" * 18)
    
    # Property correlations
    properties = ['density', 'clustering', 'max_core', 'degree_cv', 'nodes']
    
    print("Correlations with FastCM+ success:")
    for prop in properties:
        corr = analysis_df['success_rate'].corr(analysis_df[prop])
        print(f"  {prop}: {corr:.3f}")
    
    print(f"\nKey Findings")
    print("-" * 11)
    
    if len(low_success) > 0 and len(high_success) > 0:
        density_diff = high_success['density'].mean() - low_success['density'].mean()
        core_diff = high_success['max_core'].mean() - low_success['max_core'].mean()
        
        print(f"Successful vs failed networks:")
        print(f"  Density difference: {density_diff:+.3f}")
        print(f"  Max k-core difference: {core_diff:+.1f}")
    
    # Check what we actually found
    if len(low_success) > 0:
        print(f"\nObserved issues in failed networks:")
        # Only print what we actually detected
        empty_shells = 0
        few_collapse = 0
        for network_name in low_success['network']:
            if network_name not in networks:
                continue
            G = networks[network_name]
            cores = nx.core_number(G)
            max_core = max(cores.values()) if cores else 0
            
            for k in range(2, min(max_core + 2, 4)):
                shell_nodes = [n for n, c in cores.items() if c == k-1]
                if len(shell_nodes) == 0:
                    empty_shells += 1
                else:
                    # Count collapse nodes
                    k_minus_1_core = [n for n, c in cores.items() if c >= k-1]
                    collapse_count = 0
                    for node in shell_nodes:
                        neighbors_in_core = [n for n in G.neighbors(node) if n in k_minus_1_core]
                        if len(neighbors_in_core) == k-1:
                            collapse_count += 1
                    if collapse_count < 2:
                        few_collapse += 1
        
        if empty_shells > 0:
            print(f"- {empty_shells} cases of empty shells")
        if few_collapse > 0:
            print(f"- {few_collapse} cases of insufficient collapse nodes")

if __name__ == "__main__":
    analyze_fastcm_failure_patterns()