#!/usr/bin/env python3
"""
Final comprehensive evaluation of MRKC vs FastCM+ algorithms.

This is the main evaluation script for my project. Started as a simple comparison
but grew into a comprehensive evaluation framework as I needed to test more scenarios.

The goal is to answer my main research questions:
1. Which algorithm performs better overall?
2. How do they perform against different attack types?
3. Does network structure affect their performance?
4. Which is more cost-effective?

Development process was pretty messy - kept adding more experimental conditions
as I realized I needed to test more scenarios. Probably should have planned this
better from the start but it evolved as I learned what was important to test.

Changes over time:
- v1: Basic algorithm comparison on a few networks
- v2: Added different attack types after realizing random wasn't enough  
- v3: Added multiple budget levels and attack intensities
- v4: Added network property analysis for better understanding
- v5: Current version with comprehensive statistical analysis

TODO: Could probably optimize the experimental loops
FIXME: Some of the network property calculations are slow on large networks
NOTE: Running this takes a while - be patient!
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
    """Custom JSON encoder to handle numpy types - learned this the hard way"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class FinalEvaluationSuite:
    """
    Comprehensive evaluation of network reinforcement algorithms.
    
    This class grew over time as I needed to test more and more scenarios.
    Started simple but became quite complex as I realized how many factors
    could affect algorithm performance.
    """
    
    def __init__(self, output_dir="final_evaluation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        # Track some stats during experiments
        self.experiment_stats = {
            'total_attempted': 0,
            'successful': 0,
            'failed_reinforcement': 0,
            'failed_attacks': 0,
            'failed_metrics': 0
        }
        
    def run_comprehensive_evaluation(self, max_network_size=500, runs_per_config=5):
        """
        Run the main evaluation experiments.
        
        This method handles the entire experimental pipeline. Had to design this
        carefully to make sure I test all the scenarios I need for my research questions.
        """
        
        print("FINAL ALGORITHM EVALUATION")
        print("=" * 50)
        print("This will test both algorithms across multiple scenarios to answer")
        print("the main research questions for my project.")
        
        # Load all networks using consistent loading
        print("\nLoading networks for evaluation...")
        networks = {}
        
        # Use shared network loading to ensure consistency with diagnostics
        from shared_networks import get_consistent_networks
        networks = get_consistent_networks(max_network_size=max_network_size)
        
        print(f"Loaded {len(networks)} networks for testing")
        
        # Define experimental parameters
        # These parameters evolved as I figured out what I needed to test
        algorithms = {
            'MRKC': mrkc_reinforce,
            'FastCM+': fastcm_plus_reinforce
        }
        
        # Test different attack types - learned these are standard in the literature
        attack_types = ['degree', 'kcore', 'betweenness', 'random']
        
        # Test different budget levels - need to see how performance scales
        budgets = [5, 10, 15, 20, 30]
        
        # Test different attack intensities - networks behave differently under different stress
        attack_intensities = [0.05, 0.1, 0.15, 0.2]
        
        # Calculate total experimental load
        total_configs = (len(networks) * len(algorithms) * len(attack_types) * 
                        len(budgets) * len(attack_intensities))
        total_runs = total_configs * runs_per_config
        
        print(f"\nExperimental design:")
        print(f"- Networks: {len(networks)}")
        print(f"- Algorithms: {list(algorithms.keys())}")
        print(f"- Attack types: {attack_types}")
        print(f"- Budget levels: {budgets}")
        print(f"- Attack intensities: {attack_intensities}")
        print(f"- Runs per configuration: {runs_per_config}")
        print(f"- Total configurations: {total_configs}")
        print(f"- Total experimental runs: {total_runs}")
        print(f"\nThis might take a while...")
        
        # Main experimental loop
        config_count = 0
        successful_runs = 0
        
        for net_name, G in networks.items():
            # Skip networks that are too small or too large
            if G.number_of_nodes() < 10 or G.number_of_nodes() > max_network_size:
                continue
                
            print(f"\nTesting network: {net_name} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
            
            # Analyze network properties once per network
            # This helps answer research question 3 about network dependencies
            network_props = self.analyze_network_properties(G)
            
            for budget in budgets:
                for algo_name, algo_func in algorithms.items():
                    
                    # Apply reinforcement once per budget/algorithm combination
                    # This is more efficient than doing it for every attack
                    try:
                        start_time = time.time()
                        if algo_name == 'FastCM+':
                            G_reinforced, edges_added = algo_func(G, budget=budget)
                        else:
                            G_reinforced, edges_added = algo_func(G, budget=budget)
                        
                        reinforce_time = time.time() - start_time
                        followers = followers_gained(G, G_reinforced)
                        
                    except Exception as e:
                        print(f"    {algo_name} reinforcement failed: {e}")
                        self.experiment_stats['failed_reinforcement'] += 1
                        continue
                    
                    # Test this reinforced network against all attacks
                    for attack_type in attack_types:
                        for attack_intensity in attack_intensities:
                            config_count += 1
                            self.experiment_stats['total_attempted'] += runs_per_config
                            
                            print(f"  [{config_count}/{total_configs}] {algo_name} vs {attack_type} "
                                  f"(budget={budget}, intensity={attack_intensity:.0%})")
                            
                            # Run multiple trials for statistical reliability
                            for run in range(runs_per_config):
                                try:
                                    result = self.run_single_experiment(
                                        net_name, G, G_reinforced, network_props,
                                        algo_name, budget, attack_type, attack_intensity,
                                        edges_added, followers, reinforce_time, run
                                    )
                                    
                                    if result:
                                        self.results.append(result)
                                        successful_runs += 1
                                        self.experiment_stats['successful'] += 1
                                
                                except Exception as e:
                                    print(f"      Run {run} failed: {e}")
                                    # Track different types of failures for debugging
                                    if 'attack' in str(e).lower():
                                        self.experiment_stats['failed_attacks'] += 1
                                    else:
                                        self.experiment_stats['failed_metrics'] += 1
                                    continue
        
        print(f"\nExperiment completed!")
        print(f"Successful runs: {successful_runs} out of {total_runs} attempted")
        print(f"Success rate: {successful_runs/total_runs:.1%}")
        
        # Print some basic failure statistics for debugging
        print(f"\nFailure breakdown:")
        print(f"- Reinforcement failures: {self.experiment_stats['failed_reinforcement']}")
        print(f"- Attack simulation failures: {self.experiment_stats['failed_attacks']}")
        print(f"- Metrics calculation failures: {self.experiment_stats['failed_metrics']}")
        
        return len(self.results) > 0
    
    def analyze_network_properties(self, G):
        """
        Extract network properties for analysis.
        
        This function grew as I realized I needed more network characteristics
        to understand why algorithms perform differently on different networks.
        """
        
        cores = nx.core_number(G)
        degrees = dict(G.degree())
        
        # Basic structural properties
        props = {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G),
            'is_connected': nx.is_connected(G),
            'components': nx.number_connected_components(G)
        }
        
        # Degree-based properties
        degree_vals = list(degrees.values())
        if degree_vals:
            props.update({
                'avg_degree': sum(degree_vals) / len(degree_vals),
                'max_degree': max(degree_vals),
                'min_degree': min(degree_vals),
                'degree_variance': pd.Series(degree_vals).var()
            })
        
        # Core-based properties
        core_vals = list(cores.values())
        if core_vals:
            props.update({
                'max_core': max(core_vals),
                'avg_core': sum(core_vals) / len(core_vals),
                'core_diversity': len(set(core_vals))
            })
        
        # Clustering coefficient - skip for very large networks as it's slow
        try:
            if G.number_of_nodes() <= 1000:
                props['avg_clustering'] = nx.average_clustering(G)
            else:
                props['avg_clustering'] = 0  # Placeholder for large networks
        except:
            props['avg_clustering'] = 0
        
        # Try to classify network type based on name patterns
        # This is imperfect but helps with analysis
        network_type = 'unknown'
        net_name = str(G).lower()
        if 'scale' in net_name or 'barabasi' in net_name:
            network_type = 'scale_free'
        elif 'random' in net_name or 'erdos' in net_name:
            network_type = 'random'
        elif 'small' in net_name or 'watts' in net_name:
            network_type = 'small_world'
        elif any(real_name in net_name for real_name in ['karate', 'florentine', 'davis']):
            network_type = 'real_world'
        
        props['network_type'] = network_type
        
        return props
    
    def run_single_experiment(self, net_name, G_orig, G_reinforced, network_props,
                            algorithm, budget, attack_type, attack_intensity,
                            edges_added, followers, reinforce_time, run_id):
        """
        Run a single experimental trial.
        
        This function handles one complete experiment: attack both networks,
        measure damage, calculate improvements. Had to be careful about
        randomness here to get reliable results.
        """
        
        # Attack the original network to establish baseline
        G_orig_attacked, removed_orig = attack_network(G_orig, attack_type, attack_intensity)
        damage_orig = measure_damage(G_orig, G_orig_attacked, resilience_method='max_core_ratio')
        
        # Attack the reinforced network
        G_reinf_attacked, removed_reinf = attack_network(G_reinforced, attack_type, attack_intensity)
        damage_reinf = measure_damage(G_reinforced, G_reinf_attacked, resilience_method='max_core_ratio')
        
        # Calculate improvement metrics
        core_improvement = damage_orig['core_damage'] - damage_reinf['core_damage']
        resilience_improvement = damage_reinf['core_resilience'] - damage_orig['core_resilience']
        efficiency = core_improvement / len(edges_added) if len(edges_added) > 0 else 0
        
        # Create comprehensive result record
        # Converting numpy types to avoid JSON serialization issues
        result = {
            # Experiment metadata
            'network': net_name,
            'algorithm': algorithm,
            'budget': int(budget),
            'attack_type': attack_type,
            'attack_intensity': float(attack_intensity),
            'run_id': int(run_id),
            
            # Network properties (prefixed to avoid confusion with results)
            **{f'net_{k}': self._convert_numpy(v) for k, v in network_props.items()},
            
            # Reinforcement results
            'edges_added': int(len(edges_added)),
            'followers_gained': int(followers),
            'reinforce_time': float(reinforce_time),
            
            # Attack results
            'nodes_removed_orig': int(len(removed_orig)),
            'nodes_removed_reinf': int(len(removed_reinf)),
            
            # Original network damage metrics
            'orig_core_resilience': float(damage_orig['core_resilience']),
            'orig_core_damage': float(damage_orig['core_damage']),
            'orig_largest_component_frac': float(damage_orig['largest_component_fraction']),
            'orig_fragmentation': float(damage_orig['fragmentation']),
            
            # Reinforced network damage metrics
            'reinf_core_resilience': float(damage_reinf['core_resilience']),
            'reinf_core_damage': float(damage_reinf['core_damage']),
            'reinf_largest_component_frac': float(damage_reinf['largest_component_fraction']),
            'reinf_fragmentation': float(damage_reinf['fragmentation']),
            
            # Performance metrics - these answer the main research questions
            'core_improvement': float(core_improvement),
            'resilience_improvement': float(resilience_improvement),
            'efficiency': float(efficiency),
            'relative_improvement': float(resilience_improvement / damage_orig['core_resilience'] if damage_orig['core_resilience'] > 0 else 0)
        }
        
        return result
    
    def _convert_numpy(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
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
        """
        Save all experimental results to files.
        
        Saves both the raw data and summary statistics for analysis.
        The summary helps me quickly understand the main findings.
        """
        
        if not self.results:
            print("No results to save - something went wrong")
            return None
        
        # Save the main experimental data
        df = pd.DataFrame(self.results)
        csv_path = self.output_dir / "comprehensive_evaluation.csv"
        df.to_csv(csv_path, index=False)
        
        # Generate and save summary statistics
        summary = self.generate_summary_statistics(df)
        summary_path = self.output_dir / "evaluation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)
        
        print(f"Results saved to {csv_path}")
        print(f"Summary saved to {summary_path}")
        
        return df
    
    def generate_summary_statistics(self, df):
        """
        Generate summary statistics for quick analysis.
        
        This creates aggregated statistics that help answer the research questions
        without having to dig through all the raw data every time.
        """
        
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
        
        # Algorithm performance comparison - answers research question 1
        algo_stats = df.groupby('algorithm').agg({
            'reinf_core_resilience': ['mean', 'std', 'min', 'max'],
            'followers_gained': ['mean', 'std', 'sum'],
            'efficiency': ['mean', 'std'],
            'core_improvement': ['mean', 'std'],
            'reinforce_time': ['mean', 'std']
        }).round(4)
        
        # Convert to regular dict with proper types (pandas can be annoying)
        algo_stats_dict = {}
        for algo in algo_stats.index:
            algo_stats_dict[algo] = {}
            for col in algo_stats.columns:
                metric_name = f"{col[0]}_{col[1]}"
                value = algo_stats.loc[algo, col]
                algo_stats_dict[algo][metric_name] = float(value) if not pd.isna(value) else None
        
        summary['algorithm_performance'] = algo_stats_dict
        
        # Attack-specific performance - answers research question 2
        attack_stats = df.groupby(['algorithm', 'attack_type'])['reinf_core_resilience'].mean().unstack().round(4)
        attack_stats_dict = {}
        for algo in attack_stats.index:
            attack_stats_dict[algo] = {}
            for attack in attack_stats.columns:
                value = attack_stats.loc[algo, attack]
                attack_stats_dict[algo][attack] = float(value) if not pd.isna(value) else None
        
        summary['attack_specific_performance'] = attack_stats_dict
        
        # Network type analysis - answers research question 3
        if 'net_network_type' in df.columns:
            network_stats = df.groupby(['net_network_type', 'algorithm'])['reinf_core_resilience'].mean().unstack().round(4)
            network_stats_dict = {}
            for net_type in network_stats.index:
                network_stats_dict[net_type] = {}
                for algo in network_stats.columns:
                    value = network_stats.loc[net_type, algo]
                    network_stats_dict[net_type][algo] = float(value) if not pd.isna(value) else None
            
            summary['network_type_performance'] = network_stats_dict
        
        # Budget effectiveness - answers research question 4
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
        """
        Analyze results and print key findings.
        
        This provides a quick overview of the main findings to answer
        my research questions. Helps me understand what the data is telling me.
        """
        
        print("\n" + "=" * 60)
        print("EXPERIMENTAL RESULTS ANALYSIS")
        print("=" * 60)
        
        print("\nRESEARCH QUESTION 1: Algorithm Performance Comparison")
        print("-" * 55)
        
        # Overall performance comparison
        algo_perf = df.groupby('algorithm').agg({
            'reinf_core_resilience': ['mean', 'std'],
            'followers_gained': ['mean', 'std'],
            'efficiency': ['mean', 'std'],
            'core_improvement': ['mean', 'std']
        }).round(4)
        
        print("Overall Algorithm Performance:")
        print(algo_perf)
        
        # Statistical comparison
        mrkc_data = df[df['algorithm'] == 'MRKC']['reinf_core_resilience']
        fastcm_data = df[df['algorithm'] == 'FastCM+']['reinf_core_resilience']
        
        print(f"\nResilience Performance Summary:")
        print(f"MRKC: {mrkc_data.mean():.4f} ± {mrkc_data.std():.4f}")
        print(f"FastCM+: {fastcm_data.mean():.4f} ± {fastcm_data.std():.4f}")
        print(f"Difference: {mrkc_data.mean() - fastcm_data.mean():.4f}")
        
        print("\nRESEARCH QUESTION 2: Attack-Specific Performance")
        print("-" * 50)
        
        attack_analysis = df.groupby(['algorithm', 'attack_type'])['reinf_core_resilience'].mean().unstack().round(4)
        print("Resilience by Attack Type:")
        print(attack_analysis)
        
        # Find best algorithm for each attack type
        print("\nBest Algorithm for Each Attack:")
        for attack in df['attack_type'].unique():
            attack_data = df[df['attack_type'] == attack]
            best = attack_data.groupby('algorithm')['reinf_core_resilience'].mean().idxmax()
            score = attack_data.groupby('algorithm')['reinf_core_resilience'].mean()[best]
            print(f"- {attack}: {best} ({score:.4f})")
        
        print("\nRESEARCH QUESTION 3: Network Structure Dependencies")
        print("-" * 55)
        
        if 'net_network_type' in df.columns:
            network_analysis = df.groupby(['net_network_type', 'algorithm'])['reinf_core_resilience'].mean().unstack().round(4)
            print("Performance by Network Type:")
            print(network_analysis)
        
        # Network size dependencies
        df['size_category'] = pd.cut(df['net_nodes'], bins=[0, 50, 100, 200, float('inf')], 
                                   labels=['Small', 'Medium', 'Large', 'XLarge'])
        size_analysis = df.groupby(['size_category', 'algorithm'])['reinf_core_resilience'].mean().unstack().round(4)
        print("\nPerformance by Network Size:")
        print(size_analysis)
        
        print("\nRESEARCH QUESTION 4: Cost-Effectiveness Analysis")
        print("-" * 50)
        
        # Budget efficiency analysis
        budget_analysis = df.groupby(['algorithm', 'budget'])['efficiency'].mean().unstack().round(4)
        print("Efficiency by Budget Level:")
        print(budget_analysis)
        
        # Return on investment calculation
        roi_analysis = df.groupby('algorithm').apply(
            lambda x: (x['core_improvement'] / x['budget']).mean()
        ).round(4)
        print(f"\nReturn on Investment (improvement per budget unit):")
        for algo, roi in roi_analysis.items():
            print(f"- {algo}: {roi:.4f}")
        
        print("\nKEY FINDINGS SUMMARY")
        print("-" * 25)
        
        # Generate key insights for the report
        best_overall = df.groupby('algorithm')['reinf_core_resilience'].mean().idxmax()
        best_expansion = df.groupby('algorithm')['followers_gained'].mean().idxmax()
        best_efficiency = df.groupby('algorithm')['efficiency'].mean().idxmax()
        
        print(f"- Best overall resilience: {best_overall}")
        print(f"- Best k-core expansion: {best_expansion}")
        print(f"- Best efficiency: {best_efficiency}")
        
        # Attack intensity effects
        intensity_effects = df.groupby(['algorithm', 'attack_intensity'])['reinf_core_resilience'].mean().unstack().round(4)
        print(f"\nResilience vs Attack Intensity:")
        print(intensity_effects)

def main():
    """Run the final comprehensive evaluation for my project."""
    
    print("FINAL ALGORITHM EVALUATION")
    print("=" * 40)
    print("This evaluation will provide the main results for my project.")
    print("Testing both algorithms comprehensively to answer research questions.")
    
    # Create evaluation suite
    evaluator = FinalEvaluationSuite()
    
    # Run comprehensive evaluation
    success = evaluator.run_comprehensive_evaluation(
        max_network_size=300,  # Reasonable size to complete in reasonable time
        runs_per_config=3      # Multiple runs for statistical reliability
    )
    
    if success:
        # Save all results
        df = evaluator.save_results()
        
        # Analyze and display key findings
        evaluator.analyze_results(df)
        
        print(f"\nEVALUATION COMPLETE!")
        print(f"Comprehensive data collected to answer all research questions")
        print(f"Check final_evaluation/ directory for complete results")
        print(f"Ready to write up findings for the project report!")
        
        return True
    else:
        print(f"\nEvaluation failed - check for errors")
        return False

if __name__ == "__main__":
    main()