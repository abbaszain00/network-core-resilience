#!/usr/bin/env python3
"""
Final evaluation of MRKC vs FastCM+ algorithms for network resilience.

Abbas Zain-Ul-Abidin (K21067382)
Supervisor: Dr. Grigorios Loukides
"""

import sys
sys.path.append('src')

import pandas as pd
import networkx as nx
import time
import numpy as np
from pathlib import Path
from datetime import datetime

from synthetic import get_all_synthetic
from real_world import get_all_real_networks
from mrkc import mrkc_reinforce
from fastcm import fastcm_plus_reinforce
from attacks import attack_network
from metrics import measure_damage, followers_gained

class FinalEvaluationSuite:
    """Evaluation framework for comparing network reinforcement algorithms."""
    
    def __init__(self, output_dir="final_evaluation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        self.start_time = None
        
    def get_network_properties(self, G):
        """Extract basic network structural properties."""
        
        core_numbers = nx.core_number(G)
        max_core = max(core_numbers.values()) if core_numbers else 0
        
        return {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(), 
            'density': nx.density(G),
            'max_core': max_core,
            'clustering': nx.average_clustering(G)
        }
        
    def run_comprehensive_evaluation(self, max_network_size=5000, runs_per_config=3):
        """Execute full experimental evaluation."""
        
        self.start_time = time.time()
        print("ALGORITHM EVALUATION STARTING")
        print("=" * 40)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load test networks
        print("\nLoading networks...")
        from shared_networks import get_consistent_networks
        networks = get_consistent_networks(max_network_size=max_network_size)
        print(f"Loaded {len(networks)} networks")
        
        # Define experimental parameters
        algorithms = {
            'MRKC': mrkc_reinforce,
            'FastCM+': fastcm_plus_reinforce
        }
        
        attack_types = ['degree', 'kcore', 'betweenness', 'random']
        budgets = [5, 10, 15, 20, 30]
        attack_intensities = [0.05, 0.1, 0.15, 0.2]
        
        total_configs = (len(networks) * len(algorithms) * len(attack_types) * 
                        len(budgets) * len(attack_intensities))
        total_runs = total_configs * runs_per_config
        
        print(f"Experimental setup:")
        print(f"   Networks: {len(networks)}")
        print(f"   Algorithms: {list(algorithms.keys())}")
        print(f"   Attack types: {attack_types}")
        print(f"   Budget levels: {budgets}")
        print(f"   Attack intensities: {attack_intensities}")
        print(f"   Runs per config: {runs_per_config}")
        print(f"   Total runs: {total_runs}")
        
        # Execute experiments
        successful_runs = 0
        config_count = 0
        
        for net_name, G in networks.items():
            if G.number_of_nodes() < 10 or G.number_of_nodes() > max_network_size:
                continue
                
            print(f"\nTesting: {net_name} ({G.number_of_nodes()} nodes)")
            network_props = self.get_network_properties(G)
            
            for budget in budgets:
                for algo_name, algo_func in algorithms.items():
                    
                    # Apply reinforcement
                    try:
                        G_reinforced, edges_added = algo_func(G, budget=budget)
                        followers = followers_gained(G, G_reinforced)
                    except Exception as e:
                        print(f"    {algo_name} failed: {e}")
                        continue
                    
                    # Test against attacks
                    for attack_type in attack_types:
                        for attack_intensity in attack_intensities:
                            config_count += 1
                            print(f"  [{config_count}/{total_configs}] {algo_name} vs {attack_type} "
                                  f"(budget={budget}, intensity={attack_intensity:.0%})")
                            
                            for run in range(runs_per_config):
                                try:
                                    result = self.run_experiment(
                                        net_name, G, G_reinforced, network_props,
                                        algo_name, budget, attack_type, attack_intensity,
                                        edges_added, followers, run
                                    )
                                    
                                    if result:
                                        self.results.append(result)
                                        successful_runs += 1
                                
                                except Exception as e:
                                    print(f"      Run {run} failed: {e}")
                                    continue
        
        # Report completion
        total_time = time.time() - self.start_time
        print(f"\nEVALUATION COMPLETE")
        print(f"Time: {total_time/60:.1f} minutes")
        print(f"Success rate: {successful_runs}/{total_runs} ({successful_runs/total_runs:.1%})")
        
        return len(self.results) > 0
    
    def run_experiment(self, net_name, G_orig, G_reinforced, network_props,
                      algorithm, budget, attack_type, attack_intensity,
                      edges_added, followers, run_id):
        """Execute single experimental run."""
        
        # Attack original network
        G_orig_attacked, removed_orig = attack_network(G_orig, attack_type, attack_intensity)
        damage_orig = measure_damage(G_orig, G_orig_attacked, resilience_method='max_core_ratio')
        
        # Attack reinforced network
        G_reinf_attacked, removed_reinf = attack_network(G_reinforced, attack_type, attack_intensity)
        damage_reinf = measure_damage(G_reinforced, G_reinf_attacked, resilience_method='max_core_ratio')
        
        # Calculate metrics
        core_improvement = damage_orig['core_damage'] - damage_reinf['core_damage']
        resilience_improvement = damage_reinf['core_resilience'] - damage_orig['core_resilience']
        efficiency = core_improvement / len(edges_added) if len(edges_added) > 0 else 0
        
        # Return experimental result
        return {
            'network': net_name,
            'algorithm': algorithm,
            'budget': budget,
            'attack_type': attack_type,
            'attack_intensity': attack_intensity,
            'run_id': run_id,
            
            'net_nodes': network_props['nodes'],
            'net_edges': network_props['edges'],
            'net_density': network_props['density'],
            'net_max_core': network_props['max_core'],
            'net_clustering': network_props['clustering'],
            
            'edges_added': len(edges_added),
            'followers_gained': followers,
            
            'nodes_removed_orig': len(removed_orig),
            'nodes_removed_reinf': len(removed_reinf),
            
            'orig_core_resilience': damage_orig['core_resilience'],
            'orig_core_damage': damage_orig['core_damage'],
            'reinf_core_resilience': damage_reinf['core_resilience'],
            'reinf_core_damage': damage_reinf['core_damage'],
            
            'core_improvement': core_improvement,
            'resilience_improvement': resilience_improvement,
            'efficiency': efficiency
        }
    
    def save_results(self):
        """Save results to CSV file."""
        
        if not self.results:
            print("No results to save")
            return None
        
        df = pd.DataFrame(self.results)
        csv_path = self.output_dir / "evaluation_results.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"Results saved to {csv_path}")
        return df
    
    def analyze_results(self, df):
        """Analyze experimental results."""
        
        print("\nRESULTS ANALYSIS")
        print("=" * 40)
        
        print("\nAlgorithm Performance Comparison")
        print("-" * 30)
        
        # Overall performance
        algo_perf = df.groupby('algorithm').agg({
            'reinf_core_resilience': ['mean', 'std'],
            'followers_gained': ['mean', 'std'],
            'efficiency': ['mean', 'std']
        }).round(4)
        
        print("Overall Performance:")
        print(algo_perf)
        
        # Resilience comparison
        mrkc_data = df[df['algorithm'] == 'MRKC']['reinf_core_resilience']
        fastcm_data = df[df['algorithm'] == 'FastCM+']['reinf_core_resilience']
        
        print(f"\nResilience Summary:")
        print(f"MRKC: {mrkc_data.mean():.4f} ± {mrkc_data.std():.4f}")
        print(f"FastCM+: {fastcm_data.mean():.4f} ± {fastcm_data.std():.4f}")
        print(f"Difference: {mrkc_data.mean() - fastcm_data.mean():.4f}")
        
        print("\nAttack-Specific Performance")
        print("-" * 25)
        
        attack_analysis = df.groupby(['algorithm', 'attack_type'])['reinf_core_resilience'].mean().unstack().round(4)
        print("Resilience by Attack Type:")
        print(attack_analysis)
        
        # Best algorithm per attack
        print("\nBest Algorithm by Attack:")
        for attack in df['attack_type'].unique():
            attack_data = df[df['attack_type'] == attack]
            best = attack_data.groupby('algorithm')['reinf_core_resilience'].mean().idxmax()
            score = attack_data.groupby('algorithm')['reinf_core_resilience'].mean()[best]
            print(f"- {attack}: {best} ({score:.4f})")
        
        print("\nNetwork Size Analysis")
        print("-" * 20)
        
        df['size_category'] = pd.cut(df['net_nodes'], bins=[0, 50, 200, float('inf')], 
                                   labels=['Small', 'Medium', 'Large'])
        size_analysis = df.groupby(['size_category', 'algorithm'])['reinf_core_resilience'].mean().unstack().round(4)
        print("Performance by Network Size:")
        print(size_analysis)
        
        print("\nCost-Effectiveness Analysis")
        print("-" * 25)
        
        budget_analysis = df.groupby(['algorithm', 'budget'])['efficiency'].mean().unstack().round(4)
        print("Efficiency by Budget:")
        print(budget_analysis)
        
        roi_analysis = df.groupby('algorithm').apply(
            lambda x: (x['core_improvement'] / x['budget']).mean()
        ).round(4)
        print(f"\nReturn on Investment:")
        for algo, roi in roi_analysis.items():
            print(f"- {algo}: {roi:.4f}")
        
        print("\nKey Findings")
        print("-" * 12)
        
        best_overall = df.groupby('algorithm')['reinf_core_resilience'].mean().idxmax()
        best_expansion = df.groupby('algorithm')['followers_gained'].mean().idxmax()
        best_efficiency = df.groupby('algorithm')['efficiency'].mean().idxmax()
        
        print(f"- Best resilience: {best_overall}")
        print(f"- Best expansion: {best_expansion}")
        print(f"- Best efficiency: {best_efficiency}")

def main():
    """Execute evaluation suite."""
    
    print("NETWORK RESILIENCE EVALUATION")
    print("MSc Project - Abbas Zain-Ul-Abidin")
    print("Supervisor: Dr. Grigorios Loukides")
    
    evaluator = FinalEvaluationSuite()
    
    success = evaluator.run_comprehensive_evaluation(
        max_network_size=5000,
        runs_per_config=3
    )
    
    if success:
        df = evaluator.save_results()
        evaluator.analyze_results(df)
        print(f"\nEvaluation complete - results saved")
        return True
    else:
        print(f"\nEvaluation failed")
        return False

if __name__ == "__main__":
    main()