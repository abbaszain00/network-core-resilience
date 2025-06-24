#!/usr/bin/env python3
"""
Diagnostic tool to identify WHY algorithms fail on specific networks.
This gives you concrete evidence for your report instead of assumptions.
"""

import sys
sys.path.append('src')

import networkx as nx
import time
from pathlib import Path
import json

from real_world import get_all_real_networks
from synthetic import get_all_synthetic
from mrkc import mrkc_reinforce, find_candidate_edges
from fastcm import fastcm_plus_reinforce, get_shell_components_with_collapse_nodes

class AlgorithmDiagnostics:
    """Diagnose exactly why algorithms succeed or fail on each network."""
    
    def __init__(self):
        self.results = []
    
    def diagnose_mrkc_failure(self, G, budget=10, timeout=30):
        """Diagnose exactly why MRKC fails or succeeds."""
        
        diagnosis = {
            'algorithm': 'MRKC',
            'network_size': G.number_of_nodes(),
            'network_edges': G.number_of_edges(),
            'budget': budget,
            'success': False,
            'failure_reason': None,
            'details': {}
        }
        
        try:
            # Step 1: Check basic network properties
            cores = nx.core_number(G)
            max_core = max(cores.values()) if cores else 0
            
            diagnosis['details']['max_core'] = max_core
            diagnosis['details']['core_distribution'] = {
                str(k): list(cores.values()).count(k) 
                for k in range(max_core + 1)
            }
            
            # Step 2: Check candidate edge generation
            print(f"  Checking candidate edges...")
            start_time = time.time()
            
            try:
                candidates = find_candidate_edges(G, cores, limit=1000)
                candidate_time = time.time() - start_time
                
                diagnosis['details']['candidate_edges_found'] = len(candidates)
                diagnosis['details']['candidate_generation_time'] = candidate_time
                
                if len(candidates) == 0:
                    diagnosis['failure_reason'] = 'NO_CANDIDATE_EDGES'
                    diagnosis['details']['explanation'] = 'No potential edges found that could be added'
                    return diagnosis
                    
            except Exception as e:
                diagnosis['failure_reason'] = 'CANDIDATE_GENERATION_ERROR'
                diagnosis['details']['error'] = str(e)
                return diagnosis
            
            # Step 3: Check core preservation
            print(f"  Checking core preservation...")
            valid_candidates = 0
            preservation_time = time.time()
            
            for i, (u, v) in enumerate(candidates[:100]):  # Check first 100 for speed
                if time.time() - preservation_time > timeout:
                    diagnosis['failure_reason'] = 'CORE_PRESERVATION_TIMEOUT'
                    diagnosis['details']['explanation'] = f'Core preservation check timed out after {timeout}s'
                    return diagnosis
                
                # Test if this edge preserves core numbers
                test_graph = G.copy()
                test_graph.add_edge(u, v)
                new_cores = nx.core_number(test_graph)
                
                if all(new_cores[node] == cores[node] for node in G.nodes()):
                    valid_candidates += 1
            
            diagnosis['details']['valid_candidates_sampled'] = valid_candidates
            diagnosis['details']['valid_candidates_rate'] = valid_candidates / min(100, len(candidates))
            
            if valid_candidates == 0:
                diagnosis['failure_reason'] = 'NO_CORE_PRESERVING_EDGES'
                diagnosis['details']['explanation'] = 'No candidate edges preserve original core numbers'
                return diagnosis
            
            # Step 4: Try actual MRKC execution
            print(f"  Running MRKC...")
            start_time = time.time()
            
            try:
                G_result, edges_added = mrkc_reinforce(G, budget=budget, max_candidates=1000)
                execution_time = time.time() - start_time
                
                if len(edges_added) > 0:
                    diagnosis['success'] = True
                    diagnosis['details']['edges_added'] = len(edges_added)
                    diagnosis['details']['execution_time'] = execution_time
                else:
                    diagnosis['failure_reason'] = 'NO_EDGES_SELECTED'
                    diagnosis['details']['explanation'] = 'Algorithm found candidates but selected no edges'
                    diagnosis['details']['execution_time'] = execution_time
                    
            except Exception as e:
                diagnosis['failure_reason'] = 'EXECUTION_ERROR'
                diagnosis['details']['error'] = str(e)
                diagnosis['details']['execution_time'] = time.time() - start_time
        
        except Exception as e:
            diagnosis['failure_reason'] = 'UNKNOWN_ERROR'
            diagnosis['details']['error'] = str(e)
        
        return diagnosis
    
    def diagnose_fastcm_failure(self, G, budget=10, k=None):
        """Diagnose exactly why FastCM+ fails or succeeds."""
        
        diagnosis = {
            'algorithm': 'FastCM+',
            'network_size': G.number_of_nodes(),
            'network_edges': G.number_of_edges(),
            'budget': budget,
            'k_value': k,
            'success': False,
            'failure_reason': None,
            'details': {}
        }
        
        try:
            # Step 1: Determine k-value
            cores = nx.core_number(G)
            if k is None:
                k = max(cores.values()) + 1 if cores else 1
            
            diagnosis['k_value'] = k
            diagnosis['details']['original_max_core'] = max(cores.values()) if cores else 0
            
            # Step 2: Check k-1 shell existence
            k_minus_1_shell = {u for u, c in cores.items() if c == k - 1}
            diagnosis['details']['k_minus_1_shell_size'] = len(k_minus_1_shell)
            
            if len(k_minus_1_shell) == 0:
                diagnosis['failure_reason'] = 'EMPTY_K_MINUS_1_SHELL'
                diagnosis['details']['explanation'] = f'No nodes in (k-1)-shell for k={k}'
                diagnosis['details']['core_distribution'] = {
                    str(core): list(cores.values()).count(core) 
                    for core in range(max(cores.values()) + 1)
                }
                return diagnosis
            
            # Step 3: Check shell components
            print(f"  Checking shell components...")
            try:
                components = get_shell_components_with_collapse_nodes(G, k)
                diagnosis['details']['shell_components'] = len(components)
                
                if len(components) == 0:
                    diagnosis['failure_reason'] = 'NO_SHELL_COMPONENTS'
                    diagnosis['details']['explanation'] = 'No valid shell components found'
                    return diagnosis
                
                # Analyze components
                component_details = []
                for i, (comp, collapse) in enumerate(components):
                    component_details.append({
                        'component_size': len(comp),
                        'collapse_nodes': len(collapse)
                    })
                
                diagnosis['details']['component_analysis'] = component_details
                
            except Exception as e:
                diagnosis['failure_reason'] = 'COMPONENT_ANALYSIS_ERROR'
                diagnosis['details']['error'] = str(e)
                return diagnosis
            
            # Step 4: Try actual FastCM+ execution
            print(f"  Running FastCM+...")
            start_time = time.time()
            
            try:
                G_result, edges_added = fastcm_plus_reinforce(G, k=k, budget=budget)
                execution_time = time.time() - start_time
                
                if len(edges_added) > 0:
                    diagnosis['success'] = True
                    diagnosis['details']['edges_added'] = len(edges_added)
                    diagnosis['details']['execution_time'] = execution_time
                else:
                    diagnosis['failure_reason'] = 'NO_EDGES_ADDED'
                    diagnosis['details']['explanation'] = 'Algorithm completed but added no edges'
                    diagnosis['details']['execution_time'] = execution_time
                    
            except Exception as e:
                diagnosis['failure_reason'] = 'EXECUTION_ERROR'
                diagnosis['details']['error'] = str(e)
                diagnosis['details']['execution_time'] = time.time() - start_time
        
        except Exception as e:
            diagnosis['failure_reason'] = 'UNKNOWN_ERROR'
            diagnosis['details']['error'] = str(e)
        
        return diagnosis
    
    def diagnose_all_networks(self, max_network_size=5000):
        """Run diagnostics on all networks to identify failure patterns."""
        
        print("ALGORITHM FAILURE DIAGNOSTICS")
        print("=" * 50)
        print("This will identify exactly why algorithms fail on each network.")
        
        # Load networks
        print("\nLoading networks...")
        networks = {}
        
        # Real networks
        try:
            real_nets = get_all_real_networks(max_nodes=max_network_size)
            networks.update(real_nets)
            print(f"Loaded {len(real_nets)} real networks")
        except Exception as e:
            print(f"Could not load real networks: {e}")
        
        # Synthetic networks
        try:
            synthetic_nets = get_all_synthetic(max_nodes=max_network_size)
            networks.update(synthetic_nets)
            print(f"Loaded {len(synthetic_nets)} synthetic networks")
        except Exception as e:
            print(f"Could not load synthetic networks: {e}")
        
        print(f"Total networks to diagnose: {len(networks)}")
        
        # Run diagnostics
        for net_name, G in networks.items():
            if G.number_of_nodes() < 5:  # Skip tiny networks
                continue
                
            print(f"\n🔍 Diagnosing: {net_name} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
            
            # Diagnose MRKC
            mrkc_result = self.diagnose_mrkc_failure(G, budget=10)
            mrkc_result['network_name'] = net_name
            self.results.append(mrkc_result)
            
            status = "✅ SUCCESS" if mrkc_result['success'] else f"❌ {mrkc_result['failure_reason']}"
            print(f"  MRKC: {status}")
            
            # Diagnose FastCM+
            fastcm_result = self.diagnose_fastcm_failure(G, budget=10)
            fastcm_result['network_name'] = net_name
            self.results.append(fastcm_result)
            
            status = "✅ SUCCESS" if fastcm_result['success'] else f"❌ {fastcm_result['failure_reason']}"
            print(f"  FastCM+: {status}")
    
    def generate_failure_report(self, output_file="diagnostics_report.json"):
        """Generate a comprehensive failure analysis report."""
        
        # Save detailed results
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate summary
        print("\n" + "=" * 60)
        print("FAILURE ANALYSIS SUMMARY")
        print("=" * 60)
        
        # Group by algorithm
        mrkc_results = [r for r in self.results if r['algorithm'] == 'MRKC']
        fastcm_results = [r for r in self.results if r['algorithm'] == 'FastCM+']
        
        print(f"\n📊 MRKC ANALYSIS ({len(mrkc_results)} networks):")
        mrkc_failures = {}
        mrkc_successes = 0
        
        for result in mrkc_results:
            if result['success']:
                mrkc_successes += 1
            else:
                reason = result['failure_reason']
                if reason not in mrkc_failures:
                    mrkc_failures[reason] = []
                mrkc_failures[reason].append(result['network_name'])
        
        print(f"• Successes: {mrkc_successes}/{len(mrkc_results)} ({mrkc_successes/len(mrkc_results):.1%})")
        print("• Failure reasons:")
        for reason, networks in mrkc_failures.items():
            print(f"  - {reason}: {len(networks)} networks ({', '.join(networks[:3])}{'...' if len(networks) > 3 else ''})")
        
        print(f"\n📊 FASTCM+ ANALYSIS ({len(fastcm_results)} networks):")
        fastcm_failures = {}
        fastcm_successes = 0
        
        for result in fastcm_results:
            if result['success']:
                fastcm_successes += 1
            else:
                reason = result['failure_reason']
                if reason not in fastcm_failures:
                    fastcm_failures[reason] = []
                fastcm_failures[reason].append(result['network_name'])
        
        print(f"• Successes: {fastcm_successes}/{len(fastcm_results)} ({fastcm_successes/len(fastcm_results):.1%})")
        print("• Failure reasons:")
        for reason, networks in fastcm_failures.items():
            print(f"  - {reason}: {len(networks)} networks ({', '.join(networks[:3])}{'...' if len(networks) > 3 else ''})")
        
        # Networks where both work
        both_work = []
        for mrkc_res, fastcm_res in zip(mrkc_results, fastcm_results):
            if mrkc_res['network_name'] == fastcm_res['network_name']:
                if mrkc_res['success'] and fastcm_res['success']:
                    both_work.append(mrkc_res['network_name'])
        
        print(f"\n🎯 NETWORKS WHERE BOTH ALGORITHMS WORK:")
        print(f"• Count: {len(both_work)}/{len(mrkc_results)}")
        print(f"• Networks: {', '.join(both_work)}")
        
        print(f"\n💾 Detailed results saved to: {output_file}")
        
        return {
            'mrkc_success_rate': mrkc_successes / len(mrkc_results),
            'fastcm_success_rate': fastcm_successes / len(fastcm_results),
            'both_work_count': len(both_work),
            'mrkc_failures': mrkc_failures,
            'fastcm_failures': fastcm_failures,
            'working_networks': both_work
        }

def main():
    """Run comprehensive algorithm diagnostics."""
    
    diagnostics = AlgorithmDiagnostics()
    
    # Run diagnostics on all networks
    diagnostics.diagnose_all_networks(max_network_size=5000)
    
    # Generate report
    summary = diagnostics.generate_failure_report()
    
    print(f"\n🎉 DIAGNOSTICS COMPLETE!")
    print(f"You now have concrete evidence for why algorithms succeed/fail.")
    print(f"Use this data to document your findings with certainty!")

if __name__ == "__main__":
    main()