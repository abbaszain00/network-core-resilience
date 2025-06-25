#!/usr/bin/env python3
"""
Final comprehensive evaluation of MRKC vs FastCM+ across all dimensions.
This completes your main research questions before moving to shell analysis.
"""

import sys
sys.path.append('src')

import pandas as pd
import networkx as nx
import time
import json
import numpy as np
from pathlib import Path

from synthetic import get_all_synthetic
from real_world import get_all_real_networks
from mrkc import mrkc_reinforce
from fastcm import fastcm_plus_reinforce
from attacks import attack_network
from metrics import measure_damage, followers_gained

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class FinalEvaluationSuite:
    """Comprehensive evaluation answering all main research questions."""
    
    def __init__(self, output_dir="final_evaluation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        
    def run_comprehensive_evaluation(self, max_network_size=500, runs_per_config=5):
        """Run the final comprehensive evaluation."""
        
        print("FINAL COMPREHENSIVE ALGORITHM EVALUATION")
        print("=" * 60)
        print("This will answer all your main research questions definitively.")
        
        # Load all available networks
        print("\n📚 Loading networks...")
        networks = {}
        
        # Real networks
        from shared_networks import get_consistent_networks
        networks = get_consistent_networks(max_nodes=max_network_size)
        
        # Synthetic networks - comprehensive coverage
        synthetic_nets = get_all_synthetic(max_nodes=max_network_size, include_variants=True)
        networks.update(synthetic_nets)
        print(f"Loaded {len(synthetic_nets)} synthetic networks")
        
        print(f"Total networks: {len(networks)}")
        
        # Experimental parameters - comprehensive coverage
        algorithms = {
            'MRKC': mrkc_reinforce,
            'FastCM+': fastcm_plus_reinforce
        }
        
        attack_types = ['degree', 'kcore', 'betweenness', 'random']
        budgets = [5, 10, 15, 20, 30]  # More budget levels
        attack_intensities = [0.05, 0.1, 0.15, 0.2]  # Multiple attack intensities
        
        total_configs = (len(networks) * len(algorithms) * len(attack_types) * 
                        len(budgets) * len(attack_intensities))
        total_runs = total_configs * runs_per_config
        
        print(f"\n🔬 Experimental Design:")
        print(f"• Networks: {len(networks)}")
        print(f"• Algorithms: {list(algorithms.keys())}")
        print(f"• Attack types: {attack_types}")
        print(f"• Budgets: {budgets}")
        print(f"• Attack intensities: {attack_intensities}")
        print(f"• Runs per config: {runs_per_config}")
        print(f"• Total configurations: {total_configs}")
        print(f"• Total runs: {total_runs}")
        
        # Run experiments
        config_count = 0
        successful_runs = 0
        
        for net_name, G in networks.items():
            if G.number_of_nodes() < 10 or G.number_of_nodes() > max_network_size:
                continue
                
            print(f"\n📊 Network: {net_name} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
            
            # Basic network properties
            network_props = self.analyze_network_properties(G)
            
            for budget in budgets:
                for algo_name, algo_func in algorithms.items():
                    
                    # Apply reinforcement once per budget/algorithm combo
                    try:
                        start_time = time.time()
                        if algo_name == 'FastCM+':
                            G_reinforced, edges_added = algo_func(G, budget=budget)
                        else:
                            G_reinforced, edges_added = algo_func(G, budget=budget)
                        
                        reinforce_time = time.time() - start_time
                        followers = followers_gained(G, G_reinforced)
                        
                    except Exception as e:
                        print(f"    ❌ {algo_name} failed: {e}")
                        continue
                    
                    for attack_type in attack_types:
                        for attack_intensity in attack_intensities:
                            config_count += 1
                            
                            print(f"  [{config_count}/{total_configs}] {algo_name} vs {attack_type} "
                                  f"(budget={budget}, intensity={attack_intensity:.0%})")
                            
                            # Run multiple times for this configuration
                            for run in range(runs_per_config):
                                try:
                                    result = self.run_single_evaluation(
                                        net_name, G, G_reinforced, network_props,
                                        algo_name, budget, attack_type, attack_intensity,
                                        edges_added, followers, reinforce_time, run
                                    )
                                    
                                    if result:
                                        self.results.append(result)
                                        successful_runs += 1
                                
                                except Exception as e:
                                    print(f"      Run {run} failed: {e}")
                                    continue
        
        print(f"\n✅ Completed {successful_runs} successful runs out of {total_runs} attempted")
        print(f"Success rate: {successful_runs/total_runs:.1%}")
        
        return len(self.results) > 0
    
    def analyze_network_properties(self, G):
        """Extract comprehensive network properties for analysis."""
        
        cores = nx.core_number(G)
        degrees = dict(G.degree())
        
        # Basic properties
        props = {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G),
            'is_connected': nx.is_connected(G),
            'components': nx.number_connected_components(G)
        }
        
        # Degree properties
        degree_vals = list(degrees.values())
        if degree_vals:
            props.update({
                'avg_degree': sum(degree_vals) / len(degree_vals),
                'max_degree': max(degree_vals),
                'min_degree': min(degree_vals),
                'degree_variance': pd.Series(degree_vals).var()
            })
        
        # Core properties
        core_vals = list(cores.values())
        if core_vals:
            props.update({
                'max_core': max(core_vals),
                'avg_core': sum(core_vals) / len(core_vals),
                'core_diversity': len(set(core_vals))
            })
        
        # Clustering
        try:
            props['avg_clustering'] = nx.average_clustering(G)
        except:
            props['avg_clustering'] = 0
        
        # Network type detection
        network_type = 'unknown'
        if 'scale' in str(G).lower() or 'barabasi' in str(G).lower():
            network_type = 'scale_free'
        elif 'random' in str(G).lower() or 'erdos' in str(G).lower():
            network_type = 'random'
        elif 'small' in str(G).lower() or 'watts' in str(G).lower():
            network_type = 'small_world'
        elif any(real_name in str(G).lower() for real_name in ['karate', 'florentine', 'dolphins']):
            network_type = 'real_world'
        
        props['network_type'] = network_type
        
        return props
    
    def run_single_evaluation(self, net_name, G_orig, G_reinforced, network_props,
                            algorithm, budget, attack_type, attack_intensity,
                            edges_added, followers, reinforce_time, run_id):
        """Run a single evaluation configuration."""
        
        # Attack original network (baseline)
        G_orig_attacked, removed_orig = attack_network(G_orig, attack_type, attack_intensity)
        damage_orig = measure_damage(G_orig, G_orig_attacked, resilience_method='max_core_ratio')
        
        # Attack reinforced network
        G_reinf_attacked, removed_reinf = attack_network(G_reinforced, attack_type, attack_intensity)
        damage_reinf = measure_damage(G_reinforced, G_reinf_attacked, resilience_method='max_core_ratio')
        
        # Calculate improvements and efficiency
        core_improvement = damage_orig['core_damage'] - damage_reinf['core_damage']
        resilience_improvement = damage_reinf['core_resilience'] - damage_orig['core_resilience']
        efficiency = core_improvement / len(edges_added) if len(edges_added) > 0 else 0
        
        # Comprehensive result record - convert numpy types to Python types
        result = {
            # Experiment metadata
            'network': net_name,
            'algorithm': algorithm,
            'budget': int(budget),
            'attack_type': attack_type,
            'attack_intensity': float(attack_intensity),
            'run_id': int(run_id),
            
            # Network properties - convert numpy types
            **{f'net_{k}': self._convert_numpy(v) for k, v in network_props.items()},
            
            # Reinforcement results
            'edges_added': int(len(edges_added)),
            'followers_gained': int(followers),
            'reinforce_time': float(reinforce_time),
            
            # Attack results
            'nodes_removed_orig': int(len(removed_orig)),
            'nodes_removed_reinf': int(len(removed_reinf)),
            
            # Original network damage
            'orig_core_resilience': float(damage_orig['core_resilience']),
            'orig_core_damage': float(damage_orig['core_damage']),
            'orig_largest_component_frac': float(damage_orig['largest_component_fraction']),
            'orig_fragmentation': float(damage_orig['fragmentation']),
            
            # Reinforced network damage
            'reinf_core_resilience': float(damage_reinf['core_resilience']),
            'reinf_core_damage': float(damage_reinf['core_damage']),
            'reinf_largest_component_frac': float(damage_reinf['largest_component_fraction']),
            'reinf_fragmentation': float(damage_reinf['fragmentation']),
            
            # Performance metrics
            'core_improvement': float(core_improvement),
            'resilience_improvement': float(resilience_improvement),
            'efficiency': float(efficiency),
            'relative_improvement': float(resilience_improvement / damage_orig['core_resilience'] if damage_orig['core_resilience'] > 0 else 0)
        }
        
        return result
    
    def _convert_numpy(self, obj):
        """Convert numpy types to Python native types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict()
        else:
            return obj
    
    def save_results(self):
        """Save comprehensive results."""
        
        if not self.results:
            print("No results to save")
            return None
        
        # Save detailed results
        df = pd.DataFrame(self.results)
        csv_path = self.output_dir / "comprehensive_evaluation.csv"
        df.to_csv(csv_path, index=False)
        
        # Save summary statistics with proper JSON encoding
        summary = self.generate_summary_statistics(df)
        summary_path = self.output_dir / "evaluation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)
        
        print(f"💾 Results saved to {csv_path}")
        print(f"📋 Summary saved to {summary_path}")
        
        return df
    
    def generate_summary_statistics(self, df):
        """Generate comprehensive summary statistics."""
        
        summary = {
            'experiment_overview': {
                'total_runs': int(len(df)),
                'networks_tested': int(df['network'].nunique()),
                'algorithms': list(df['algorithm'].unique()),
                'attack_types': list(df['attack_type'].unique()),
                'budget_levels': [int(x) for x in sorted(df['budget'].unique())],
                'attack_intensities': [float(x) for x in sorted(df['attack_intensity'].unique())]
            }
        }
        
        # Algorithm performance comparison - convert to regular Python types
        algo_stats = df.groupby('algorithm').agg({
            'reinf_core_resilience': ['mean', 'std', 'min', 'max'],
            'followers_gained': ['mean', 'std', 'sum'],
            'efficiency': ['mean', 'std'],
            'core_improvement': ['mean', 'std'],
            'reinforce_time': ['mean', 'std']
        }).round(4)
        
        # Convert to regular dict with proper types
        algo_stats_dict = {}
        for algo in algo_stats.index:
            algo_stats_dict[algo] = {}
            for col in algo_stats.columns:
                metric_name = f"{col[0]}_{col[1]}"
                value = algo_stats.loc[algo, col]
                algo_stats_dict[algo][metric_name] = float(value) if not pd.isna(value) else None
        
        summary['algorithm_performance'] = algo_stats_dict
        
        # Attack-specific performance
        attack_stats = df.groupby(['algorithm', 'attack_type'])['reinf_core_resilience'].mean().unstack().round(4)
        attack_stats_dict = {}
        for algo in attack_stats.index:
            attack_stats_dict[algo] = {}
            for attack in attack_stats.columns:
                value = attack_stats.loc[algo, attack]
                attack_stats_dict[algo][attack] = float(value) if not pd.isna(value) else None
        
        summary['attack_specific_performance'] = attack_stats_dict
        
        # Network type analysis
        if 'net_network_type' in df.columns:
            network_stats = df.groupby(['net_network_type', 'algorithm'])['reinf_core_resilience'].mean().unstack().round(4)
            network_stats_dict = {}
            for net_type in network_stats.index:
                network_stats_dict[net_type] = {}
                for algo in network_stats.columns:
                    value = network_stats.loc[net_type, algo]
                    network_stats_dict[net_type][algo] = float(value) if not pd.isna(value) else None
            
            summary['network_type_performance'] = network_stats_dict
        
        # Budget effectiveness
        budget_stats = df.groupby(['algorithm', 'budget'])['efficiency'].mean().unstack().round(4)
        budget_stats_dict = {}
        for algo in budget_stats.index:
            budget_stats_dict[algo] = {}
            for budget in budget_stats.columns:
                value = budget_stats.loc[algo, budget]
                budget_stats_dict[algo][str(budget)] = float(value) if not pd.isna(value) else None
        
        summary['budget_effectiveness'] = budget_stats_dict
        
        return summary
    
    def analyze_results(self, df):
        """Analyze results to answer research questions."""
        
        print("\n" + "=" * 60)
        print("COMPREHENSIVE RESULTS ANALYSIS")
        print("=" * 60)
        
        print("\n🔬 RESEARCH QUESTION 1: Algorithm Performance Comparison")
        print("-" * 55)
        
        # Overall performance
        algo_perf = df.groupby('algorithm').agg({
            'reinf_core_resilience': ['mean', 'std'],
            'followers_gained': ['mean', 'std'],
            'efficiency': ['mean', 'std'],
            'core_improvement': ['mean', 'std']
        }).round(4)
        
        print("Overall Algorithm Performance:")
        print(algo_perf)
        
        # Statistical significance testing
        mrkc_data = df[df['algorithm'] == 'MRKC']['reinf_core_resilience']
        fastcm_data = df[df['algorithm'] == 'FastCM+']['reinf_core_resilience']
        
        print(f"\nResilient Performance:")
        print(f"MRKC: {mrkc_data.mean():.4f} ± {mrkc_data.std():.4f}")
        print(f"FastCM+: {fastcm_data.mean():.4f} ± {fastcm_data.std():.4f}")
        print(f"Difference: {mrkc_data.mean() - fastcm_data.mean():.4f}")
        
        print("\n🎯 RESEARCH QUESTION 2: Attack-Specific Performance")
        print("-" * 50)
        
        attack_analysis = df.groupby(['algorithm', 'attack_type'])['reinf_core_resilience'].mean().unstack().round(4)
        print("Resilience by Attack Type:")
        print(attack_analysis)
        
        # Find best algorithm for each attack
        print("\nBest Algorithm for Each Attack:")
        for attack in df['attack_type'].unique():
            attack_data = df[df['attack_type'] == attack]
            best = attack_data.groupby('algorithm')['reinf_core_resilience'].mean().idxmax()
            score = attack_data.groupby('algorithm')['reinf_core_resilience'].mean()[best]
            print(f"• {attack}: {best} ({score:.4f})")
        
        print("\n🏗️ RESEARCH QUESTION 3: Network Structure Dependencies")
        print("-" * 55)
        
        if 'net_network_type' in df.columns:
            network_analysis = df.groupby(['net_network_type', 'algorithm'])['reinf_core_resilience'].mean().unstack().round(4)
            print("Performance by Network Type:")
            print(network_analysis)
        
        # Size dependencies
        df['size_category'] = pd.cut(df['net_nodes'], bins=[0, 50, 100, 200, float('inf')], 
                                   labels=['Small', 'Medium', 'Large', 'XLarge'])
        size_analysis = df.groupby(['size_category', 'algorithm'])['reinf_core_resilience'].mean().unstack().round(4)
        print("\nPerformance by Network Size:")
        print(size_analysis)
        
        print("\n💰 RESEARCH QUESTION 4: Cost-Effectiveness Analysis")
        print("-" * 50)
        
        # Budget efficiency
        budget_analysis = df.groupby(['algorithm', 'budget'])['efficiency'].mean().unstack().round(4)
        print("Efficiency by Budget Level:")
        print(budget_analysis)
        
        # ROI analysis
        roi_analysis = df.groupby('algorithm').apply(
            lambda x: (x['core_improvement'] / x['budget']).mean()
        ).round(4)
        print(f"\nReturn on Investment (improvement per budget unit):")
        for algo, roi in roi_analysis.items():
            print(f"• {algo}: {roi:.4f}")
        
        print("\n📊 KEY FINDINGS SUMMARY")
        print("-" * 25)
        
        # Generate key insights
        best_overall = df.groupby('algorithm')['reinf_core_resilience'].mean().idxmax()
        best_expansion = df.groupby('algorithm')['followers_gained'].mean().idxmax()
        best_efficiency = df.groupby('algorithm')['efficiency'].mean().idxmax()
        
        print(f"• Best overall resilience: {best_overall}")
        print(f"• Best k-core expansion: {best_expansion}")
        print(f"• Best efficiency: {best_efficiency}")
        
        # Attack intensity effects
        intensity_effects = df.groupby(['algorithm', 'attack_intensity'])['reinf_core_resilience'].mean().unstack().round(4)
        print(f"\nResilience vs Attack Intensity:")
        print(intensity_effects)

def main():
    """Run the final comprehensive evaluation."""
    
    print("FINAL ALGORITHM EVALUATION SUITE")
    print("=" * 50)
    print("This will definitively answer your main research questions.")
    print("After this completes, we'll move to the FastCM+ shell analysis.")
    
    # Create evaluation suite
    evaluator = FinalEvaluationSuite()
    
    # Run comprehensive evaluation
    success = evaluator.run_comprehensive_evaluation(
        max_network_size=300,  # Reasonable size for completion
        runs_per_config=3      # Multiple runs for statistics
    )
    
    if success:
        # Save results
        df = evaluator.save_results()
        
        # Analyze results
        evaluator.analyze_results(df)
        
        print(f"\n🎉 FINAL EVALUATION COMPLETE!")
        print(f"📊 You now have comprehensive data answering all research questions")
        print(f"📁 Check final_evaluation/ for complete results")
        print(f"🚀 Ready to proceed to FastCM+ shell structure analysis!")
        
        return True
    else:
        print(f"\n❌ Evaluation failed - check for errors")
        return False

if __name__ == "__main__":
    main()