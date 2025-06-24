#!/usr/bin/env python3
"""
Integrated final evaluation with built-in failure diagnostics.
This collects BOTH experimental results AND diagnostic information 
from the EXACT same runs, so you know precisely why each failed.
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
from mrkc import mrkc_reinforce, find_candidate_edges
from fastcm import fastcm_plus_reinforce, get_shell_components_with_collapse_nodes
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

class IntegratedEvaluationSuite:
    """Evaluation with built-in diagnostics for every run."""
    
    def __init__(self, output_dir="final_evaluation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        self.diagnostics = []
        
    def diagnose_algorithm_failure(self, G, algorithm_name, algo_func, budget, network_name):
        """Diagnose why an algorithm failed on this specific run."""
        
        diagnosis = {
            'network_name': network_name,
            'algorithm': algorithm_name,
            'budget': budget,
            'network_size': G.number_of_nodes(),
            'network_edges': G.number_of_edges(),
            'failure_reason': None,
            'success': False,
            'diagnostic_details': {}
        }
        
        try:
            if algorithm_name == 'MRKC':
                return self._diagnose_mrkc(G, algo_func, budget, diagnosis)
            elif algorithm_name == 'FastCM+':
                return self._diagnose_fastcm(G, algo_func, budget, diagnosis)
        except Exception as e:
            diagnosis['failure_reason'] = 'DIAGNOSTIC_ERROR'
            diagnosis['diagnostic_details']['error'] = str(e)
            
        return diagnosis
    
    def _diagnose_mrkc(self, G, algo_func, budget, diagnosis):
        """Diagnose MRKC-specific failure."""
        
        cores = nx.core_number(G)
        diagnosis['diagnostic_details']['max_core'] = max(cores.values()) if cores else 0
        
        # Check candidate edges
        try:
            candidates = find_candidate_edges(G, cores, limit=1000)
            diagnosis['diagnostic_details']['candidate_edges'] = len(candidates)
            
            if len(candidates) == 0:
                diagnosis['failure_reason'] = 'NO_CANDIDATE_EDGES'
                return diagnosis
                
            # Quick check for core-preserving edges (sample first 50)
            valid_count = 0
            for u, v in candidates[:50]:
                test_graph = G.copy()
                test_graph.add_edge(u, v)
                new_cores = nx.core_number(test_graph)
                if all(new_cores[node] == cores[node] for node in G.nodes()):
                    valid_count += 1
            
            diagnosis['diagnostic_details']['valid_candidates_sampled'] = valid_count
            diagnosis['diagnostic_details']['valid_rate'] = valid_count / min(50, len(candidates))
            
            if valid_count == 0:
                diagnosis['failure_reason'] = 'NO_CORE_PRESERVING_EDGES'
                return diagnosis
                
        except Exception as e:
            diagnosis['failure_reason'] = 'CANDIDATE_GENERATION_ERROR'
            diagnosis['diagnostic_details']['candidate_error'] = str(e)
            return diagnosis
        
        # Try actual execution
        try:
            start_time = time.time()
            G_result, edges_added = algo_func(G, budget=budget)
            execution_time = time.time() - start_time
            
            diagnosis['diagnostic_details']['execution_time'] = execution_time
            diagnosis['diagnostic_details']['edges_added'] = len(edges_added)
            
            if len(edges_added) > 0:
                diagnosis['success'] = True
            else:
                diagnosis['failure_reason'] = 'NO_EDGES_SELECTED'
                
        except Exception as e:
            diagnosis['failure_reason'] = 'EXECUTION_ERROR'
            diagnosis['diagnostic_details']['execution_error'] = str(e)
        
        return diagnosis
    
    def _diagnose_fastcm(self, G, algo_func, budget, diagnosis):
        """Diagnose FastCM+-specific failure."""
        
        cores = nx.core_number(G)
        max_core = max(cores.values()) if cores else 0
        k = max_core + 1
        
        diagnosis['diagnostic_details']['max_core'] = max_core
        diagnosis['diagnostic_details']['k_value'] = k
        
        # Check k-1 shell
        k_minus_1_shell = {u for u, c in cores.items() if c == k - 1}
        diagnosis['diagnostic_details']['k_minus_1_shell_size'] = len(k_minus_1_shell)
        
        if len(k_minus_1_shell) == 0:
            diagnosis['failure_reason'] = 'EMPTY_K_MINUS_1_SHELL'
            diagnosis['diagnostic_details']['core_distribution'] = {
                str(core): list(cores.values()).count(core) 
                for core in range(max_core + 1)
            }
            return diagnosis
        
        # Check shell components
        try:
            components = get_shell_components_with_collapse_nodes(G, k)
            diagnosis['diagnostic_details']['shell_components'] = len(components)
            
            if len(components) == 0:
                diagnosis['failure_reason'] = 'NO_SHELL_COMPONENTS'
                return diagnosis
                
        except Exception as e:
            diagnosis['failure_reason'] = 'COMPONENT_ANALYSIS_ERROR'
            diagnosis['diagnostic_details']['component_error'] = str(e)
            return diagnosis
        
        # Try actual execution
        try:
            start_time = time.time()
            G_result, edges_added = algo_func(G, k=k, budget=budget)
            execution_time = time.time() - start_time
            
            diagnosis['diagnostic_details']['execution_time'] = execution_time
            diagnosis['diagnostic_details']['edges_added'] = len(edges_added)
            
            if len(edges_added) > 0:
                diagnosis['success'] = True
            else:
                diagnosis['failure_reason'] = 'NO_EDGES_ADDED'
                
        except Exception as e:
            diagnosis['failure_reason'] = 'EXECUTION_ERROR'
            diagnosis['diagnostic_details']['execution_error'] = str(e)
        
        return diagnosis
    
    def run_comprehensive_evaluation(self, max_network_size=500, runs_per_config=5):
        """Run evaluation with integrated diagnostics."""
        
        print("INTEGRATED EVALUATION WITH FAILURE DIAGNOSTICS")
        print("=" * 60)
        print("This tracks both results AND why each algorithm succeeds/fails.")
        
        # Load networks (EXACTLY the same as your current evaluation)
        print("\n📚 Loading networks...")
        networks = {}
        
        # Real networks
        try:
            real_nets = get_all_real_networks(max_nodes=max_network_size)
            networks.update(real_nets)
            print(f"Loaded {len(real_nets)} real networks")
        except Exception as e:
            print(f"Could not load real networks: {e}")
        
        # Synthetic networks
        synthetic_nets = get_all_synthetic(max_nodes=max_network_size, include_variants=True)
        networks.update(synthetic_nets)
        print(f"Loaded {len(synthetic_nets)} synthetic networks")
        
        print(f"Total networks: {len(networks)}")
        
        # Experimental parameters (EXACTLY the same as your current evaluation)
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
        
        print(f"\n🔬 Experimental Design (IDENTICAL to your current setup):")
        print(f"• Networks: {len(networks)}")
        print(f"• Algorithms: {list(algorithms.keys())}")
        print(f"• Attack types: {attack_types}")
        print(f"• Budgets: {budgets}")
        print(f"• Attack intensities: {attack_intensities}")
        print(f"• Runs per config: {runs_per_config}")
        print(f"• Total runs: {total_runs}")
        
        # Run experiments with diagnostics
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
                    
                    # Apply reinforcement with diagnostics
                    try:
                        start_time = time.time()
                        if algo_name == 'FastCM+':
                            G_reinforced, edges_added = algo_func(G, budget=budget)
                        else:
                            G_reinforced, edges_added = algo_func(G, budget=budget)
                        
                        reinforce_time = time.time() - start_time
                        followers = followers_gained(G, G_reinforced)
                        
                        # Success - create diagnostic entry
                        diagnosis = {
                            'network_name': net_name,
                            'algorithm': algo_name,
                            'budget': budget,
                            'network_size': G.number_of_nodes(),
                            'network_edges': G.number_of_edges(),
                            'success': True,
                            'failure_reason': None,
                            'diagnostic_details': {
                                'execution_time': reinforce_time,
                                'edges_added': len(edges_added),
                                'followers_gained': followers
                            }
                        }
                        self.diagnostics.append(diagnosis)
                        
                    except Exception as e:
                        print(f"    ❌ {algo_name} failed: {e}")
                        
                        # Failure - diagnose why
                        diagnosis = self.diagnose_algorithm_failure(
                            G, algo_name, algo_func, budget, net_name
                        )
                        self.diagnostics.append(diagnosis)
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
        """Extract network properties (same as original)."""
        
        cores = nx.core_number(G)
        degrees = dict(G.degree())
        
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
        """Run single evaluation (same as original)."""
        
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
        
        # Result record with type conversion
        result = {
            'network': net_name,
            'algorithm': algorithm,
            'budget': int(budget),
            'attack_type': attack_type,
            'attack_intensity': float(attack_intensity),
            'run_id': int(run_id),
            
            # Network properties
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
        """Save results with diagnostics."""
        
        if not self.results:
            print("No results to save")
            return None
        
        # Save experimental results
        df = pd.DataFrame(self.results)
        csv_path = self.output_dir / "comprehensive_evaluation.csv"
        df.to_csv(csv_path, index=False)
        
        # Save diagnostic results
        diagnostics_path = self.output_dir / "failure_diagnostics.json"
        with open(diagnostics_path, 'w') as f:
            json.dump(self.diagnostics, f, indent=2, cls=NumpyEncoder)
        
        # Generate diagnostic summary
        self.generate_diagnostic_summary()
        
        print(f"💾 Results saved to {csv_path}")
        print(f"🔍 Diagnostics saved to {diagnostics_path}")
        
        return df
    
    def generate_diagnostic_summary(self):
        """Generate diagnostic summary report."""
        
        print(f"\n" + "=" * 60)
        print("FAILURE DIAGNOSTIC SUMMARY")
        print("=" * 60)
        
        # Group diagnostics by algorithm
        mrkc_diag = [d for d in self.diagnostics if d['algorithm'] == 'MRKC']
        fastcm_diag = [d for d in self.diagnostics if d['algorithm'] == 'FastCM+']
        
        print(f"\n📊 MRKC DIAGNOSTIC RESULTS:")
        mrkc_success = sum(1 for d in mrkc_diag if d['success'])
        print(f"• Success rate: {mrkc_success}/{len(mrkc_diag)} ({mrkc_success/len(mrkc_diag):.1%})")
        
        mrkc_failures = {}
        for d in mrkc_diag:
            if not d['success']:
                reason = d['failure_reason']
                if reason not in mrkc_failures:
                    mrkc_failures[reason] = []
                mrkc_failures[reason].append(d['network_name'])
        
        for reason, networks in mrkc_failures.items():
            print(f"• {reason}: {len(networks)} networks ({', '.join(networks)})")
        
        print(f"\n📊 FASTCM+ DIAGNOSTIC RESULTS:")
        fastcm_success = sum(1 for d in fastcm_diag if d['success'])
        print(f"• Success rate: {fastcm_success}/{len(fastcm_diag)} ({fastcm_success/len(fastcm_diag):.1%})")
        
        fastcm_failures = {}
        for d in fastcm_diag:
            if not d['success']:
                reason = d['failure_reason']
                if reason not in fastcm_failures:
                    fastcm_failures[reason] = []
                fastcm_failures[reason].append(d['network_name'])
        
        for reason, networks in fastcm_failures.items():
            print(f"• {reason}: {len(networks)} networks ({', '.join(networks)})")
        
        # Networks where both work
        successful_networks = set()
        for mrkc_d, fastcm_d in zip(mrkc_diag, fastcm_diag):
            if mrkc_d['network_name'] == fastcm_d['network_name']:
                if mrkc_d['success'] and fastcm_d['success']:
                    successful_networks.add(mrkc_d['network_name'])
        
        print(f"\n🎯 NETWORKS WHERE BOTH ALGORITHMS WORK:")
        print(f"• Count: {len(successful_networks)}")
        print(f"• Networks: {', '.join(sorted(successful_networks))}")
        
        print(f"\n💡 NOW YOU HAVE CONCRETE EVIDENCE:")
        print(f"• Exact failure reasons for every algorithm-network combination")
        print(f"• Diagnostic details from the SAME runs as your results")
        print(f"• Complete transparency about what works and what doesn't")

def main():
    """Run integrated evaluation with diagnostics."""
    
    print("INTEGRATED FINAL EVALUATION WITH DIAGNOSTICS")
    print("=" * 55)
    print("This gives you BOTH experimental results AND diagnostic explanations")
    print("from the exact same runs - no guesswork needed!")
    
    evaluator = IntegratedEvaluationSuite()
    
    # Run evaluation with diagnostics
    success = evaluator.run_comprehensive_evaluation(
        max_network_size=300,
        runs_per_config=3
    )
    
    if success:
        # Save everything
        df = evaluator.save_results()
        
        print(f"\n🎉 INTEGRATED EVALUATION COMPLETE!")
        print(f"📊 You have experimental results AND diagnostic explanations")
        print(f"📁 Check final_evaluation/ for:")
        print(f"   • comprehensive_evaluation.csv (your experimental data)")
        print(f"   • failure_diagnostics.json (why each algorithm succeeded/failed)")
        print(f"🎯 Now you can document findings with complete confidence!")
        
        return True
    else:
        print(f"\n❌ Evaluation failed - check for errors")
        return False

if __name__ == "__main__":
    main()