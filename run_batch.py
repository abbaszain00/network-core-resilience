import networkx as nx
import json
from src import generation, mrkc, fastcm, attacks, metrics

results = []

for seed in range(5):  # You can increase this later
    print(f"\n==== Seed {seed} ====")
    
    # Generate graph
    G = generation.get_test_graph(type="scale_free", n=30, seed=seed)

    # === MRKC Reinforcement ===
    G_mrkc, _ = mrkc.mrkc_reinforce(G, budget=5)
    G_mrkc_attacked = attacks.degree_based_attack(G_mrkc, num_nodes=3)
    count_mrkc, pct_mrkc = metrics.retained_top_kcore_members(G, G_mrkc_attacked)

    # === FastCM+ Reinforcement ===
    G_fastcm, _ = fastcm.fastcm_reinforce(G, budget=5)
    G_fastcm_attacked = attacks.degree_based_attack(G_fastcm, num_nodes=3)
    count_fastcm, pct_fastcm = metrics.retained_top_kcore_members(G, G_fastcm_attacked)

    # === Original Graph ===
    G_attacked = attacks.degree_based_attack(G, num_nodes=3)
    count_orig, pct_orig = metrics.retained_top_kcore_members(G, G_attacked)

    # === Store Results ===
    results.append({
        "seed": seed,
        "strategy": "original",
        "retained_top_core": count_orig,
        "retention_pct": pct_orig
    })
    results.append({
        "seed": seed,
        "strategy": "mrkc (scaffold)",
        "retained_top_core": count_mrkc,
        "retention_pct": pct_mrkc
    })
    results.append({
        "seed": seed,
        "strategy": "fastcm+ (scaffold)",
        "retained_top_core": count_fastcm,
        "retention_pct": pct_fastcm
    })

# === Print Summary ===
print("\n==== Summary ====")
for r in results:
    print(r)

with open("mrkc_fastcm_results.json", "w") as f:
    json.dump(results, f, indent=2)

