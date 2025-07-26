#!/usr/bin/env python3
"""
Analysis script for evaluation results.
Network Resilience Algorithm Comparison: MRKC vs FastCM+

Abbas Zain-Ul-Abidin (K21067382)
MSc Computer Science, King's College London
Supervisor: Dr. Grigorios Loukides
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def load_experimental_data():
    """Load and validate experimental results from CSV file."""
    try:
        df = pd.read_csv('final_evaluation/evaluation_results.csv')
        print(f"Loaded {len(df)} experimental results")
        return df
    except FileNotFoundError:
        print("Error: Could not load evaluation_results.csv")
        print("Run final_evaluation.py first to generate results")
        return None

def analyse_overall_performance(df):
    """Analyse overall algorithm performance across all experimental conditions."""
    print("\nRESULTS SUMMARY")
    print("=" * 40)
    
    # Experimental setup validation
    print(f"Algorithms: {list(df['algorithm'].unique())}")
    print(f"Networks: {df['network'].nunique()}")
    print(f"Attack types: {list(df['attack_type'].unique())}")
    print(f"Budget levels: {sorted(df['budget'].unique().tolist())}")
    print(f"Attack intensities: {sorted(df['attack_intensity'].unique().tolist())}")
    
    # Core resilience analysis
    print(f"\nALGORITHM PERFORMANCE")
    print("-" * 20)
    
    resilience_stats = df.groupby('algorithm')['reinf_core_resilience'].agg(['mean', 'std']).round(4)
    print("Core Resilience:")
    for algo in resilience_stats.index:
        mean_val = resilience_stats.loc[algo, 'mean']
        std_val = resilience_stats.loc[algo, 'std']
        print(f"  {algo}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Statistical significance testing
    if 'MRKC' in df['algorithm'].values and 'FastCM+' in df['algorithm'].values:
        mrkc_scores = df[df['algorithm'] == 'MRKC']['reinf_core_resilience']
        fastcm_scores = df[df['algorithm'] == 'FastCM+']['reinf_core_resilience']
        
        difference = mrkc_scores.mean() - fastcm_scores.mean()
        percentage_diff = (difference / fastcm_scores.mean()) * 100
        
        print(f"\nMRKC vs FastCM+ difference: {difference:.4f} ({percentage_diff:+.1f}%)")
        
        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(mrkc_scores, fastcm_scores)
        print(f"Statistical test: t={t_stat:.3f}, p={p_value:.4f}")
        print(f"Sample sizes: MRKC={len(mrkc_scores)}, FastCM+={len(fastcm_scores)}")
        
        if p_value < 0.05:
            print("Result: Statistically significant difference")
            better = "MRKC" if difference > 0 else "FastCM+"
            print(f"{better} performs significantly better")
        else:
            print("Result: No statistically significant difference")
    
    # K-core expansion analysis
    print(f"\nK-core Expansion Analysis:")
    followers_stats = df.groupby('algorithm')['followers_gained'].agg(['mean', 'std']).round(2)
    for algo in followers_stats.index:
        mean_val = followers_stats.loc[algo, 'mean']
        std_val = followers_stats.loc[algo, 'std']
        print(f"  {algo}: {mean_val:.1f} ± {std_val:.1f} followers")
    
    # Efficiency analysis
    print(f"\nEfficiency Analysis (improvement per edge):")
    efficiency_stats = df.groupby('algorithm')['efficiency'].agg(['mean', 'std']).round(4)
    for algo in efficiency_stats.index:
        mean_val = efficiency_stats.loc[algo, 'mean']
        std_val = efficiency_stats.loc[algo, 'std']
        print(f"  {algo}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Core improvement analysis
    print(f"\nCore Improvement Analysis:")
    improvement_stats = df.groupby('algorithm')['core_improvement'].agg(['mean', 'std']).round(3)
    for algo in improvement_stats.index:
        mean_val = improvement_stats.loc[algo, 'mean']
        std_val = improvement_stats.loc[algo, 'std']
        print(f"  {algo}: {mean_val:.3f} ± {std_val:.3f}")

def analyse_attack_strategies(df):
    """Analyse algorithm performance by attack strategy."""
    print(f"\nPERFORMANCE BY ATTACK STRATEGY")
    print("-" * 33)
    
    attack_results = {}
    
    # Performance breakdown by attack type
    for attack in df['attack_type'].unique():
        print(f"\n{attack.upper()} attacks:")
        attack_data = df[df['attack_type'] == attack]
        attack_summary = attack_data.groupby('algorithm')['reinf_core_resilience'].agg(['mean', 'std']).round(4)
        
        for algo in attack_summary.index:
            mean_val = attack_summary.loc[algo, 'mean']
            std_val = attack_summary.loc[algo, 'std']
            print(f"  {algo}: {mean_val:.4f} ± {std_val:.4f}")
        
        # Determine better performing algorithm
        if len(attack_summary) == 2:
            best_algo = attack_summary['mean'].idxmax()
            difference = attack_summary.loc[best_algo, 'mean'] - attack_summary.loc[attack_summary.index != best_algo, 'mean'].iloc[0]
            print(f"  Best: {best_algo} (advantage: {difference:.4f})")
            
            attack_results[attack] = {
                'best_algorithm': best_algo,
                'performance_difference': difference
            }
    
    # Summary of attack strategy results
    print(f"\nAttack Strategy Summary:")
    for attack, results in attack_results.items():
        print(f"  {attack:12s}: {results['best_algorithm']} wins by {results['performance_difference']:+.4f}")
    
    return attack_results

def analyse_network_effects(df):
    """Analyse network-specific performance patterns."""
    print(f"\nNETWORK SIZE EFFECTS")
    print("-" * 19)
    
    # Network size categorisation
    small_nets = df[df['net_nodes'] <= 100]
    medium_nets = df[(df['net_nodes'] > 100) & (df['net_nodes'] <= 500)]
    large_nets = df[df['net_nodes'] > 500]
    
    for data, label in [(small_nets, "Small networks (≤100 nodes)"), 
                        (medium_nets, "Medium networks (100-500 nodes)"),
                        (large_nets, "Large networks (>500 nodes)")]:
        if len(data) > 0:
            print(f"\n{label}:")
            size_perf = data.groupby('algorithm')['reinf_core_resilience'].agg(['mean', 'std']).round(4)
            for algo in size_perf.index:
                mean_val = size_perf.loc[algo, 'mean']
                std_val = size_perf.loc[algo, 'std']
                print(f"  {algo}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Individual network performance analysis
    print(f"\nNETWORK-SPECIFIC PERFORMANCE")
    print("-" * 27)
    
    network_perf = df.groupby(['network', 'algorithm'])['reinf_core_resilience'].mean().unstack()
    if 'MRKC' in network_perf.columns and 'FastCM+' in network_perf.columns:
        network_perf['MRKC_Advantage'] = network_perf['MRKC'] - network_perf['FastCM+']
        network_perf = network_perf.sort_values('MRKC_Advantage', ascending=False)
        
        print("Networks where MRKC has largest advantage:")
        top_networks = network_perf.head(3)[['MRKC', 'FastCM+', 'MRKC_Advantage']].round(4)
        for network, row in top_networks.iterrows():
            print(f"  {network:20s}: MRKC={row['MRKC']:.4f}, FastCM+={row['FastCM+']:.4f}, Diff={row['MRKC_Advantage']:+.4f}")
        
        if any(network_perf['MRKC_Advantage'] < 0):
            print("\nNetworks where FastCM+ performs better:")
            negative_networks = network_perf[network_perf['MRKC_Advantage'] < 0]
            for network, row in negative_networks.iterrows():
                print(f"  {network:20s}: MRKC={row['MRKC']:.4f}, FastCM+={row['FastCM+']:.4f}, Diff={row['MRKC_Advantage']:+.4f}")
        else:
            print("\nMRKC performs better on all networks tested")

def analyse_budget_efficiency(df):
    """Analyse algorithm performance across different budget constraints."""
    print(f"\nBUDGET EFFICIENCY ANALYSIS")
    print("-" * 25)
    
    print("Resilience performance by budget level:")
    for budget in sorted(df['budget'].unique()):
        print(f"\nBudget {budget} edges:")
        budget_data = df[df['budget'] == budget]
        budget_resilience = budget_data.groupby('algorithm')['reinf_core_resilience'].agg(['mean', 'std']).round(4)
        
        for algo in budget_resilience.index:
            mean_val = budget_resilience.loc[algo, 'mean']
            std_val = budget_resilience.loc[algo, 'std']
            print(f"  {algo}: {mean_val:.4f} ± {std_val:.4f}")
    
    print(f"\nEfficiency by budget level:")
    for budget in sorted(df['budget'].unique()):
        print(f"\nBudget {budget} edges:")
        budget_data = df[df['budget'] == budget]
        budget_efficiency = budget_data.groupby('algorithm')['efficiency'].agg(['mean', 'std']).round(4)
        
        for algo in budget_efficiency.index:
            mean_val = budget_efficiency.loc[algo, 'mean']
            std_val = budget_efficiency.loc[algo, 'std']
            print(f"  {algo}: {mean_val:.4f} ± {std_val:.4f}")

def generate_summary_findings(df):
    """Generate key findings and recommendations."""
    print(f"\nKEY FINDINGS AND RECOMMENDATIONS")
    print("-" * 32)
    
    # Identify best performing algorithms for different metrics
    best_resilience = df.groupby('algorithm')['reinf_core_resilience'].mean().idxmax()
    best_expansion = df.groupby('algorithm')['followers_gained'].mean().idxmax()
    best_efficiency = df.groupby('algorithm')['efficiency'].mean().idxmax()
    
    print(f"Best overall resilience: {best_resilience}")
    print(f"Best k-core expansion: {best_expansion}")
    print(f"Best efficiency: {best_efficiency}")
    
    # Algorithm comparison summary
    resilience_scores = df.groupby('algorithm')['reinf_core_resilience'].mean()
    expansion_scores = df.groupby('algorithm')['followers_gained'].mean()
    efficiency_scores = df.groupby('algorithm')['efficiency'].mean()
    
    print(f"\nDirect algorithm comparison:")
    for algo in df['algorithm'].unique():
        res_score = resilience_scores[algo]
        exp_score = expansion_scores[algo]
        eff_score = efficiency_scores[algo]
        print(f"  {algo}: resilience={res_score:.4f}, expansion={exp_score:.1f}, efficiency={eff_score:.4f}")
    
    # Practical recommendations
    print(f"\nPractical recommendations:")
    if best_resilience == best_efficiency:
        print(f"- {best_resilience} optimal for both resilience and efficiency")
    else:
        print(f"- Trade-off exists: {best_resilience} for resilience, {best_efficiency} for efficiency")
    
    print(f"- Use {best_expansion} for maximum k-core growth")
    print(f"- Use {best_resilience} for network protection against attacks")
    
    # Attack intensity analysis
    print(f"\nAttack intensity effects:")
    intensity_analysis = df.groupby(['algorithm', 'attack_intensity'])['reinf_core_resilience'].mean().unstack()
    if not intensity_analysis.empty:
        print("Performance degradation under increasing attack intensity:")
        for algo in intensity_analysis.index:
            low_intensity = intensity_analysis.loc[algo, 0.05]
            high_intensity = intensity_analysis.loc[algo, 0.2]
            degradation = ((low_intensity - high_intensity) / low_intensity) * 100
            print(f"  {algo}: {degradation:.1f}% performance drop (5% to 20% attack intensity)")

def generate_summary_table(df):
    """Generate summary statistics table for thesis."""
    print(f"\nGenerating summary statistics table...")
    
    summary = df.groupby('algorithm').agg({
        'reinf_core_resilience': ['mean', 'std'],
        'followers_gained': ['mean', 'std'], 
        'efficiency': ['mean', 'std'],
        'core_improvement': ['mean', 'std']
    }).round(4)
    
    # Flatten column names for CSV export
    summary.columns = ['_'.join(col) for col in summary.columns]
    summary.to_csv('algorithm_performance_summary.csv')
    print("Summary table saved: algorithm_performance_summary.csv")
    
    # Generate attack-specific summary
    attack_summary = df.groupby(['attack_type', 'algorithm'])['reinf_core_resilience'].agg(['mean', 'std']).round(4)
    attack_summary.to_csv('attack_performance_summary.csv')
    print("Attack strategy table saved: attack_performance_summary.csv")
    
    return summary

def create_visualisations(df):
    """Create publication-quality visualisations for thesis."""
    print(f"\nCreating visualisations...")
    
    # Set professional plotting style
    plt.style.use('default')
    plt.rcParams.update({'font.size': 12})
    
    # Overall algorithm performance comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['reinf_core_resilience', 'followers_gained', 'efficiency']
    titles = ['Core Resilience', 'Followers Gained', 'Efficiency (per edge)']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        
        # Calculate means and standard deviations for error bars
        stats_data = df.groupby('algorithm')[metric].agg(['mean', 'std']).reset_index()
        
        bars = ax.bar(stats_data['algorithm'], stats_data['mean'], 
                     yerr=stats_data['std'], capsize=5, 
                     color=['lightblue', 'lightcoral'], alpha=0.8,
                     edgecolor='black', linewidth=1)
        
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, mean_val in zip(bars, stats_data['mean']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + stats_data['std'].iloc[0]/2,
                   f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('overall_algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Attack strategy performance comparison
    plt.figure(figsize=(10, 6))
    attack_perf = df.groupby(['algorithm', 'attack_type'])['reinf_core_resilience'].mean().unstack()
    
    if not attack_perf.empty:
        ax = attack_perf.plot(kind='bar', width=0.8, 
                             color=['lightblue', 'lightcoral', 'lightgreen', 'lightyellow'])
        plt.title('Algorithm Performance by Attack Strategy', fontweight='bold')
        plt.ylabel('Core Resilience', fontweight='bold')
        plt.xlabel('Algorithm', fontweight='bold')
        plt.legend(title='Attack Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=0)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('attack_strategy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Budget vs performance analysis
    plt.figure(figsize=(10, 6))
    
    for algo in df['algorithm'].unique():
        algo_data = df[df['algorithm'] == algo]
        budget_means = algo_data.groupby('budget')['reinf_core_resilience'].mean()
        budget_stds = algo_data.groupby('budget')['reinf_core_resilience'].std()
        
        plt.errorbar(budget_means.index, budget_means.values, yerr=budget_stds.values, 
                    marker='o', linewidth=2, markersize=6, capsize=5, label=algo)
    
    plt.xlabel('Budget (Number of Edges)', fontweight='bold')
    plt.ylabel('Core Resilience', fontweight='bold')
    plt.title('Algorithm Performance vs Budget Constraint', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('budget_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Plots saved:")
    print("- overall_algorithm_comparison.png")
    print("- attack_strategy_comparison.png") 
    print("- budget_performance_comparison.png")

def main_analysis():
    """Execute complete analysis pipeline."""
    print("NETWORK RESILIENCE ALGORITHM EVALUATION")
    print("MRKC vs FastCM+ Comparative Analysis")
    print("Abbas Zain-Ul-Abidin (K21067382)")
    print("=" * 50)
    
    # Load experimental data
    df = load_experimental_data()
    if df is None:
        return
    
    # Execute analysis components
    analyse_overall_performance(df)
    attack_results = analyse_attack_strategies(df)
    analyse_network_effects(df)
    analyse_budget_efficiency(df)
    generate_summary_findings(df)
    
    # Generate outputs for thesis
    summary_stats = generate_summary_table(df)
    create_visualisations(df)
    
    print(f"\nAnalysis complete. Generated files for thesis:")
    print("- CSV tables for LaTeX import")
    print("- High-resolution plots for figures")
    print("- Statistical summaries for results section")

if __name__ == "__main__":
    main_analysis()