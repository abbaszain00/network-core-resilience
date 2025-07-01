#!/usr/bin/env python3
"""
Analysis script for evaluation results.

Abbas Zain-Ul-Abidin (K21067382)
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def quick_analysis():
    """Analyze evaluation results from CSV file."""
    
    try:
        df = pd.read_csv('final_evaluation/evaluation_results.csv')
        print(f"Loaded {len(df)} experimental results")
    except:
        print("Error: Could not load evaluation_results.csv")
        print("Run final_evaluation.py first to generate results")
        return
    
    print("\nRESULTS SUMMARY")
    print("=" * 40)
    
    # Experimental setup
    print(f"Algorithms: {list(df['algorithm'].unique())}")
    print(f"Networks: {df['network'].nunique()}")
    print(f"Attack types: {list(df['attack_type'].unique())}")
    print(f"Budget levels: {sorted(df['budget'].unique().tolist())}")
    print(f"Attack intensities: {sorted(df['attack_intensity'].unique().tolist())}")
    
    # Algorithm performance
    print(f"\nALGORITHM PERFORMANCE")
    print("-" * 20)
    
    resilience_stats = df.groupby('algorithm')['reinf_core_resilience'].agg(['mean', 'std']).round(3)
    print("Core Resilience:")
    for algo in resilience_stats.index:
        mean_val = resilience_stats.loc[algo, 'mean']
        std_val = resilience_stats.loc[algo, 'std']
        print(f"  {algo}: {mean_val:.3f} ± {std_val:.3f}")
    
    # Simple comparison with statistical test
    if 'MRKC' in df['algorithm'].values and 'FastCM+' in df['algorithm'].values:
        mrkc_scores = df[df['algorithm'] == 'MRKC']['reinf_core_resilience']
        fastcm_scores = df[df['algorithm'] == 'FastCM+']['reinf_core_resilience']
        
        difference = mrkc_scores.mean() - fastcm_scores.mean()
        print(f"\nMRKC vs FastCM+ difference: {difference:.3f}")
        
        # Statistical significance test
        t_stat, p_value = stats.ttest_ind(mrkc_scores, fastcm_scores)
        print(f"Statistical test: t={t_stat:.3f}, p={p_value:.4f}")
        
        if p_value < 0.05:
            print("Result: Statistically significant difference")
        else:
            print("Result: No significant difference")
        
        if abs(difference) > 0.01:
            better = "MRKC" if difference > 0 else "FastCM+"
            print(f"{better} performs better")
    
    # K-core expansion
    print(f"\nK-core Expansion:")
    followers_stats = df.groupby('algorithm')['followers_gained'].mean().round(1)
    for algo, followers in followers_stats.items():
        print(f"  {algo}: {followers:.1f} followers")
    
    # Efficiency
    print(f"\nEfficiency (improvement per edge):")
    efficiency_stats = df.groupby('algorithm')['efficiency'].mean().round(4)
    for algo, eff in efficiency_stats.items():
        print(f"  {algo}: {eff:.4f}")
    
    # Attack performance breakdown
    print(f"\nPERFORMANCE BY ATTACK TYPE")
    print("-" * 25)
    
    # Calculate performance for each attack type
    for attack in df['attack_type'].unique():
        print(f"\n{attack.upper()} attacks:")
        attack_data = df[df['attack_type'] == attack]
        attack_results = attack_data.groupby('algorithm')['reinf_core_resilience'].mean()
        
        for algo in attack_results.index:
            score = attack_results[algo]
            print(f"  {algo}: {score:.3f}")
        
        # Find best for this attack
        best = attack_results.idxmax()
        print(f"  Best: {best}")
    
    print(f"\nBest algorithm per attack:")
    for attack in df['attack_type'].unique():
        attack_data = df[df['attack_type'] == attack]
        best = attack_data.groupby('algorithm')['reinf_core_resilience'].mean().idxmax()
        score = attack_data.groupby('algorithm')['reinf_core_resilience'].mean().max()
        print(f"  {attack}: {best} ({score:.3f})")
    
    # Network size effects
    print(f"\nNETWORK SIZE EFFECTS")
    print("-" * 19)
    
    small_nets = df[df['net_nodes'] <= 100]
    large_nets = df[df['net_nodes'] > 500]
    
    if len(small_nets) > 0:
        print("Small networks (≤100 nodes):")
        small_perf = small_nets.groupby('algorithm')['reinf_core_resilience'].mean().round(4)
        for algo, score in small_perf.items():
            print(f"  {algo}: {score:.4f}")
    
    if len(large_nets) > 0:
        print("Large networks (>500 nodes):")
        large_perf = large_nets.groupby('algorithm')['reinf_core_resilience'].mean().round(4)
        for algo, score in large_perf.items():
            print(f"  {algo}: {score:.4f}")
    
    # Budget efficiency
    print(f"\nBUDGET EFFICIENCY")
    print("-" * 15)
    
    print("Efficiency by budget level:")
    for budget in sorted(df['budget'].unique()):
        print(f"\nBudget {budget}:")
        budget_data = df[df['budget'] == budget]
        budget_results = budget_data.groupby('algorithm')['efficiency'].mean()
        
        for algo in budget_results.index:
            eff = budget_results[algo]
            print(f"  {algo}: {eff:.3f}")
    
    # Summary findings
    print(f"\nKEY FINDINGS")
    print("-" * 11)
    
    best_resilience = df.groupby('algorithm')['reinf_core_resilience'].mean().idxmax()
    best_expansion = df.groupby('algorithm')['followers_gained'].mean().idxmax()
    best_efficiency = df.groupby('algorithm')['efficiency'].mean().idxmax()
    
    print(f"Best resilience: {best_resilience}")
    print(f"Best expansion: {best_expansion}")
    print(f"Best efficiency: {best_efficiency}")
    
    # Compare algorithms directly
    resilience_scores = df.groupby('algorithm')['reinf_core_resilience'].mean()
    expansion_scores = df.groupby('algorithm')['followers_gained'].mean()
    
    print(f"\nDirect comparison:")
    for algo in df['algorithm'].unique():
        res_score = resilience_scores[algo]
        exp_score = expansion_scores[algo]
        print(f"{algo}: resilience {res_score:.3f}, expansion {exp_score:.1f}")
    
    print(f"\nRecommendations:")
    if best_resilience == best_efficiency:
        print(f"- {best_resilience} optimal for both resilience and efficiency")
    else:
        print(f"- Trade-off: {best_resilience} for resilience, {best_efficiency} for efficiency")
    
    print(f"- Use {best_expansion} for maximum k-core growth")
    print(f"- Use {best_resilience} for network protection")
    
    # Create simple visualization
    create_basic_plots(df)

def create_basic_plots(df):
    """Create basic comparison plots."""
    
    print(f"\nCreating basic plots...")
    
    # Simple boxplot comparison
    plt.figure(figsize=(8, 6))
    algorithms = df['algorithm'].unique()
    data_to_plot = [df[df['algorithm'] == algo]['reinf_core_resilience'] for algo in algorithms]
    
    plt.boxplot(data_to_plot, labels=algorithms)
    plt.title('Algorithm Performance Comparison')
    plt.ylabel('Core Resilience')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Attack type performance
    plt.figure(figsize=(10, 6))
    attack_perf = df.groupby(['algorithm', 'attack_type'])['reinf_core_resilience'].mean().unstack()
    attack_perf.plot(kind='bar', width=0.8)
    plt.title('Performance by Attack Type')
    plt.ylabel('Core Resilience')
    plt.xlabel('Algorithm')
    plt.legend(title='Attack Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('attack_performance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Plots saved: algorithm_comparison.png, attack_performance.png")

if __name__ == "__main__":
    quick_analysis()