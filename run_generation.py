from src import generation, attacks, metrics

# Generate original graph
G = generation.generate_random_graph(n=15, p=0.2, seed=42)
print("Original:")
print("Nodes:", G.number_of_nodes(), "Edges:", G.number_of_edges())
print("Avg Core Number:", metrics.average_core_number(G))

# Degree-based attack
G_deg = attacks.degree_based_attack(G, num_nodes=2)
print("\nAfter Degree-Based Attack:")
print("Nodes:", G_deg.number_of_nodes(), "Edges:", G_deg.number_of_edges())
print("Avg Core Number:", metrics.average_core_number(G_deg))

# K-core-based attack
G_core = attacks.kcore_based_attack(G, num_nodes=2)
print("\nAfter K-Core-Based Attack:")
print("Nodes:", G_core.number_of_nodes(), "Edges:", G_core.number_of_edges())
print("Avg Core Number:", metrics.average_core_number(G_core))
