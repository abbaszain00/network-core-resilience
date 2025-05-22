from src import generation, fastcm
import networkx as nx

# Generate graph
G = generation.get_test_graph(type="random", n=100, seed=42)
core = nx.core_number(G)
k = max(core.values())

# Step 1: Extract components
components = fastcm.get_shell_components_with_collapse_nodes(G, k)

# Step 2: Estimate cost
component_info = []
for comp, collapse_nodes in components:
    cost = fastcm.estimate_complete_conversion_cost(G, comp, collapse_nodes, k)
    component_info.append((comp, collapse_nodes, cost))

# Step 3: Select components within budget
selected = fastcm.select_components_dp_under_budget(component_info, budget=10)

# Step 4: Apply conversions
G_reinforced, added_edges = fastcm.apply_component_conversions(G, selected, k)

# Summary
print(f"Original k-core size: {sum(1 for v in core if core[v] >= k)}")
core_reinforced = nx.core_number(G_reinforced)
print(f"New k-core size: {sum(1 for v in core_reinforced if core_reinforced[v] >= k)}")
print(f"Edges added: {len(added_edges)}")
