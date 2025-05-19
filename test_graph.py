import networkx as nx
import numpy as np

# Generate a small random graph
G = nx.erdos_renyi_graph(n=10, p=0.3, seed=42)

# Print basic graph info
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

# Compute and display k-core values
core_numbers = nx.core_number(G)
print("\nCore numbers:")
for node, core in core_numbers.items():
    print(f"Node {node}: k={core}")

# Convert to adjacency matrix using NumPy
adj_matrix = nx.to_numpy_array(G)
print("\nAdjacency Matrix:\n", adj_matrix)

# Compute degrees using NumPy
degrees = np.sum(adj_matrix, axis=1)
print("\nDegrees (via NumPy):", degrees)
