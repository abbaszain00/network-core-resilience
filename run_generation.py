from src import generation, metrics, visualise, mrkc, attacks, fastcm
import networkx as nx

# === 1. Generate the graph ===
G = generation.get_test_graph(type="scale_free", n=30)
visualise.draw_graph(G, title="Original Graph")

print("Original:")
print("Nodes:", G.number_of_nodes(), "Edges:", G.number_of_edges())
print("Avg Core Number:", metrics.average_core_number(G))
print("Max Core Number:", metrics.max_core_number(G))
print("Core Distribution:", metrics.core_distribution(G))


# === 2. Apply MRKC Reinforcement ===
G_mrkc, added_edges = mrkc.mrkc_reinforce(G, budget=5)
visualise.draw_graph(G_mrkc, title="After MRKC Reinforcement")

print("\nAfter MRKC Reinforcement:")
print("Nodes:", G_mrkc.number_of_nodes(), "Edges:", G_mrkc.number_of_edges())
print("Edges added by MRKC:", added_edges)
print("Avg Core Number After MRKC:", metrics.average_core_number(G_mrkc))
print("Max Core Number After MRKC:", metrics.max_core_number(G_mrkc))
print("Core Distribution After MRKC:", metrics.core_distribution(G_mrkc))

# === 3. Apply FastCM+ Reinforcement ===
G_fastcm, fastcm_edges = fastcm.fastcm_reinforce(G, budget=5)
visualise.draw_graph(G_fastcm, title="After FastCM+ Reinforcement")

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
