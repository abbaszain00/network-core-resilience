from src import generation, metrics, visualise, mrkc, attacks, fastcm
import networkx as nx

# === 1. Generate the graph ===
G = generation.get_test_graph(type="scale_free", n=30)
# visualise.draw_graph(G, title="Original Graph")

print("Original:")
print("Nodes:", G.number_of_nodes(), "Edges:", G.number_of_edges())
print("Avg Core Number:", metrics.average_core_number(G))
print("Max Core Number:", metrics.max_core_number(G))
print("Core Distribution:", metrics.core_distribution(G))

# === 2. Apply MRKC Reinforcement ===
G_mrkc, added_edges = mrkc.mrkc_reinforce(G, budget=5)
# visualise.draw_graph(G_mrkc, title="After MRKC Reinforcement")

print("\nAfter MRKC Reinforcement:")
print("Nodes:", G_mrkc.number_of_nodes(), "Edges:", G_mrkc.number_of_edges())
print("Edges added by MRKC:", added_edges)
print("Avg Core Number After MRKC:", metrics.average_core_number(G_mrkc))
print("Max Core Number After MRKC:", metrics.max_core_number(G_mrkc))
print("Core Distribution After MRKC:", metrics.core_distribution(G_mrkc))

# === 3. Apply FastCM+ Reinforcement ===
G_fastcm, fastcm_edges = fastcm.fastcm_reinforce(G, budget=5)
# visualise.draw_graph(G_fastcm, title="After FastCM+ Reinforcement")

print("\nAfter FastCM+ Reinforcement:")
print("Nodes:", G_fastcm.number_of_nodes(), "Edges:", G_fastcm.number_of_edges())
print("Edges added by FastCM+:", fastcm_edges)
print("Avg Core Number After FastCM+:", metrics.average_core_number(G_fastcm))
print("Max Core Number After FastCM+:", metrics.max_core_number(G_fastcm))
print("Core Distribution After FastCM+:", metrics.core_distribution(G_fastcm))

print("\nΔ Core (MRKC vs FastCM+):")
print("Avg Core MRKC:", metrics.average_core_number(G_mrkc))
print("Avg Core FastCM+:", metrics.average_core_number(G_fastcm))
print("Max Core MRKC:", metrics.max_core_number(G_mrkc))
print("Max Core FastCM+:", metrics.max_core_number(G_fastcm))

# === 4. Apply Degree-Based Attack ===

# Original graph
G_attacked = attacks.degree_based_attack(G, num_nodes=3)
# visualise.draw_graph(G_attacked, title="Original Graph → After Attack")
print("\nAfter Attack on Original Graph:")
print("Avg Core Number:", metrics.average_core_number(G_attacked))
print("Max Core Number:", metrics.max_core_number(G_attacked))
print("Core Distribution:", metrics.core_distribution(G_attacked))
count, percent = metrics.retained_top_kcore_members(G, G_attacked)
print(f"Original: Retained in top-k-core after attack: {count} / 30 ({percent:.2%})")


# MRKC-reinforced graph
G_mrkc_attacked = attacks.degree_based_attack(G_mrkc, num_nodes=3)
# visualise.draw_graph(G_mrkc_attacked, title="MRKC → After Attack")

print("\nAfter Attack on MRKC-Reinforced Graph:")
print("Avg Core Number:", metrics.average_core_number(G_mrkc_attacked))
print("Max Core Number:", metrics.max_core_number(G_mrkc_attacked))
print("Core Distribution:", metrics.core_distribution(G_mrkc_attacked))
count, percent = metrics.retained_top_kcore_members(G, G_mrkc_attacked)
print(f"MRKC+: Retained in top-k-core after attack: {count} / 30 ({percent:.2%})")

# FastCM+-reinforced graph
G_fastcm_attacked = attacks.degree_based_attack(G_fastcm, num_nodes=3)
# visualise.draw_graph(G_fastcm_attacked, title="FastCM+ → After Attack")

print("\nAfter Attack on FastCM+ Reinforced Graph:")
print("Avg Core Number:", metrics.average_core_number(G_fastcm_attacked))
print("Max Core Number:", metrics.max_core_number(G_fastcm_attacked))
print("Core Distribution:", metrics.core_distribution(G_fastcm_attacked))
count, percent = metrics.retained_top_kcore_members(G, G_fastcm_attacked)
print(f"FastCM+: Retained in top-k-core after attack: {count} / 30 ({percent:.2%})")

# === 5. Apply K-Core Attack ===

# Original Graph
G_kcore_attacked = attacks.kcore_based_attack(G, num_nodes=3)
# visualise.draw_graph(G_kcore_attacked, title="Original Graph → K-Core Attack")
print("\n[K-CORE ATTACK] Original:")
print("Avg Core:", metrics.average_core_number(G_kcore_attacked))
print("Max Core:", metrics.max_core_number(G_kcore_attacked))
print("Core Distribution:", metrics.core_distribution(G_kcore_attacked))
count, percent = metrics.retained_top_kcore_members(G, G_kcore_attacked)
print(f"Retained in top-k-core: {count} / 30 ({percent:.2%})")

# MRKC-reinforced graph
G_mrkc_kcore_attacked = attacks.kcore_based_attack(G_mrkc, num_nodes=3)
# visualise.draw_graph(G_mrkc_kcore_attacked, title="MRKC → K-Core Attack")
print("\n[K-CORE ATTACK] MRKC:")
print("Avg Core:", metrics.average_core_number(G_mrkc_kcore_attacked))
print("Max Core:", metrics.max_core_number(G_mrkc_kcore_attacked))
print("Core Distribution:", metrics.core_distribution(G_mrkc_kcore_attacked))
count, percent = metrics.retained_top_kcore_members(G, G_mrkc_kcore_attacked)
print(f"Retained in top-k-core: {count} / 30 ({percent:.2%})")


# FastCM+-reinforced graph
G_fastcm_kcore_attacked = attacks.kcore_based_attack(G_fastcm, num_nodes=3)
# visualise.draw_graph(G_fastcm_kcore_attacked, title="FastCM+ → K-Core Attack")
print("\n[K-CORE ATTACK] FastCM+:")
print("Avg Core:", metrics.average_core_number(G_fastcm_kcore_attacked))
print("Max Core:", metrics.max_core_number(G_fastcm_kcore_attacked))
print("Core Distribution:", metrics.core_distribution(G_fastcm_kcore_attacked))
count, percent = metrics.retained_top_kcore_members(G, G_fastcm_kcore_attacked)
print(f"Retained in top-k-core: {count} / 30 ({percent:.2%})")


from src.mrkc import core_strength


print("\nNode\tCore\tCore Strength")
core_dict = nx.core_number(G)
for node in sorted(G.nodes())[:5]:
    cs = core_strength(G, node)
    print(f"{node}\t{core_dict[node]}\t{cs}")

