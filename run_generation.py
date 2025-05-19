from src import generation, metrics, visualise, mrkc, attacks
import networkx as nx

# === 1. Generate the graph ===
G = generation.get_test_graph(type="scale_free", n=30)
visualise.draw_graph(G, title="Original Graph")

print("Original:")
print("Nodes:", G.number_of_nodes(), "Edges:", G.number_of_edges())
print("Avg Core Number:", metrics.average_core_number(G))

# === 2. Apply MRKC Reinforcement ===
G_mrkc, added_edges = mrkc.mrkc_reinforce(G, budget=5)
visualise.draw_graph(G_mrkc, title="After MRKC Reinforcement")

print("\nAfter MRKC Reinforcement:")
print("Nodes:", G_mrkc.number_of_nodes(), "Edges:", G_mrkc.number_of_edges())
print("Edges added by MRKC:", added_edges)
print("Avg Core Number After MRKC:", metrics.average_core_number(G_mrkc))


