"""
FastCM+ Algorithm Implementation

This implements the FastCM+ algorithm from Sun et al. (2022) PVLDB paper.
This was way more complex than MRKC - took me weeks to understand all the components.

The paper has a lot of mathematical notation that was hard to translate into code.
Had to implement several different parts:
1. Shell component analysis (k-1 shell partitioning)
2. Complete conversion strategy  
3. Partial conversion with onion layers
4. Dynamic programming for component selection

Main challenges:
- Understanding the onion layer concept (section 6.1)
- Getting the anchor gain calculation right (definition 8)
- Implementing the dynamic programming part correctly
- Making sure the algorithm doesn't break on edge cases

Development notes:
- v1: Basic shell finding, no conversion strategies
- v2: Added complete conversion, but kept getting errors
- v3: Added partial conversion, still buggy
- v4: Fixed most bugs, added better error handling
- v5: Current version with DP optimization

TODO: Could probably optimize the onion layer calculation
FIXME: Might not handle disconnected components perfectly

References:
- Sun et al. "Fast Algorithms for Core Maximization on Large Graphs" PVLDB 2022
"""

import networkx as nx
from typing import List, Tuple, Set, Dict

def get_shell_components_with_collapse_nodes(G: nx.Graph, k: int) -> List[Tuple[Set[int], Set[int]]]:
    """Find (k-1)-shell components and their collapse nodes"""
    # This function finds the shell components as described in the paper
    # Had to read section 5.2 multiple times to get this right
    core_numbers = nx.core_number(G)
    shell_nodes = {u for u in G.nodes if core_numbers[u] == k - 1}

    if not shell_nodes:
        return []  # No k-1 shell exists

    # Get all nodes in (k-1)-core for neighbor checking
    k_minus_1_core = {u for u in G.nodes if core_numbers[u] >= k - 1}
    
    # Find connected components within the shell
    components = list(nx.connected_components(G.subgraph(shell_nodes)))
    results = []

    for comp in components:
        # Find collapse nodes - nodes with exactly k-1 neighbors in (k-1)-core
        # This definition was confusing at first
        collapse_nodes = set()
        for u in comp:
            neighbors_in_core = [v for v in G.neighbors(u) if v in k_minus_1_core]
            if len(neighbors_in_core) == k - 1:
                collapse_nodes.add(u)
        results.append((comp, collapse_nodes))

    return results

def estimate_complete_conversion_cost(G: nx.Graph, component: Set[int], collapse_nodes: Set[int], k: int) -> int:
    """Estimate how many edges needed for complete conversion"""
    # Algorithm 1 from the paper - had to work through the math carefully
    core_numbers = nx.core_number(G)
    k_core = {u for u in G.nodes if core_numbers[u] >= k}
    n = len(collapse_nodes)

    if n == 0:
        return 0  # Nothing to convert

    # Basic pairing strategy - each edge handles 2 collapse nodes
    cost = n // 2
    if n % 2 == 1:
        # Need one more edge for the leftover node
        if component | k_core:  # Make sure we have valid targets
            cost += 1
        else:
            # This shouldn't happen in practice but better safe than sorry
            raise RuntimeError("No valid target for the remaining collapse node")

    return cost

def apply_complete_conversion(G: nx.Graph, component: Set[int], collapse_nodes: Set[int], k: int) -> List[Tuple[int, int]]:
    """Apply complete conversion strategy from Algorithm 1"""
    # Implementation of Algorithm 1 - this took forever to debug
    added_edges = []
    core_numbers = nx.core_number(G)
    k_core = {u for u in G.nodes if core_numbers[u] >= k}
    used_nodes = set()
    nodes_list = list(collapse_nodes)

    # Strategy 1: Pair up collapse nodes where possible
    for i in range(len(nodes_list)):
        if nodes_list[i] in used_nodes:
            continue
        u = nodes_list[i]
        
        # Try to find another unused collapse node to pair with
        for j in range(i + 1, len(nodes_list)):
            v = nodes_list[j]
            if v not in used_nodes and not G.has_edge(u, v):
                added_edges.append((u, v))
                used_nodes.update([u, v])
                break

    # Strategy 2: Handle leftover nodes
    leftover_nodes = [u for u in nodes_list if u not in used_nodes]
    for u in leftover_nodes:
        # Find a valid target in component or k-core
        possible_targets = sorted(v for v in (component | k_core) - {u} if not G.has_edge(u, v))
        if possible_targets:
            added_edges.append((u, possible_targets[0]))
        else:
            # This is bad - no valid targets available
            raise RuntimeError(f"No valid target for node {u}")

    return added_edges

def compute_onion_layers(G: nx.Graph, k: int) -> Dict[int, int]:
    """Compute onion layers for partial conversion (Section 6.1)"""
    # This was probably the hardest part to understand from the paper
    # The onion metaphor helps but the implementation details were tricky
    core_numbers = nx.core_number(G)
    k_minus_1_core_nodes = {u for u in G.nodes if core_numbers[u] >= k - 1}

    if not k_minus_1_core_nodes:
        return {}

    # Work on subgraph of (k-1)-core
    subgraph = G.subgraph(k_minus_1_core_nodes).copy()
    layer_assignments = {}
    current_layer = 0

    # Peel layers like an onion - remove nodes with degree < k-1
    while subgraph.number_of_nodes() > 0:
        # Find nodes that have insufficient degree
        nodes_to_remove = [u for u in subgraph.nodes if subgraph.degree(u) < k - 1]

        if not nodes_to_remove:
            # All remaining nodes go to current layer
            for u in subgraph.nodes:
                layer_assignments[u] = current_layer
            break

        # Assign layer and remove nodes
        for u in nodes_to_remove:
            layer_assignments[u] = current_layer
            subgraph.remove_node(u)

        current_layer += 1

    return layer_assignments

def calculate_requisite_degree(u, j, layers, G, component_nodes, k):
    """Calculate how many more neighbors a node needs"""
    # From Definition 7 in the paper - this formula was confusing
    core_numbers = nx.core_number(G)
    k_core = {v for v in G.nodes if core_numbers[v] >= k}
    
    # Nodes at layer j or higher
    higher_layer_nodes = {v for v in component_nodes if layers.get(v, float('inf')) >= j}
    H = k_core | higher_layer_nodes
    
    # Count existing neighbors in H
    current_neighbors = sum(1 for v in G.neighbors(u) if v in H)
    return max(0, k - current_neighbors)

def calculate_anchor_gain(u, j, layers, G, component_nodes, k):
    """Calculate anchor gain as defined in Definition 8"""
    # This took me ages to implement correctly
    # The benefit-cost calculation was tricky to get right
    benefit = 0
    layer_j_nodes = {v for v in component_nodes if layers.get(v, -1) == j}

    # Count how many layer-j nodes would benefit from anchoring u
    for v in G.neighbors(u):
        if v in layer_j_nodes and calculate_requisite_degree(v, j, layers, G, component_nodes, k) > 0:
            benefit += 1

    # Cost is how many edges we need to add to make u a k-core member
    cost = calculate_requisite_degree(u, j, layers, G, component_nodes, k)
    
    return benefit - cost

def apply_partial_conversion(G: nx.Graph, component_nodes: Set[int], k: int, budget: int, layers: Dict[int, int]) -> Tuple[List[Tuple[int, int]], Set[int]]:
    """Apply partial conversion strategy using anchoring"""
    # Algorithm 3 from the paper - this was really complex to implement
    if not component_nodes or budget <= 0:
        return [], set()

    max_layer = max(layers.get(u, 0) for u in component_nodes)

    # Try each layer starting from j=1
    for j in range(1, max_layer + 1):
        try:
            edge_list = []
            anchored_nodes = set()
            
            # Split nodes into upper (>=j) and lower (<j) layers
            upper_layer_nodes = {u for u in component_nodes if layers.get(u, -1) >= j}
            lower_layer_nodes = {u for u in component_nodes if layers.get(u, -1) < j}
            layer_j_nodes = {u for u in component_nodes if layers.get(u, -1) == j}

            # Calculate initial requisite degrees for layer j
            requisite_degrees = {u: calculate_requisite_degree(u, j, layers, G, component_nodes, k) for u in layer_j_nodes}
            total_requisite = sum(requisite_degrees.values())

            # Greedy anchoring of lower layer nodes
            # Keep anchoring nodes with positive gain
            while True:
                best_node = None
                best_gain = -1
                
                for u in lower_layer_nodes:
                    if u in anchored_nodes:
                        continue
                    gain = calculate_anchor_gain(u, j, layers, G, component_nodes, k)
                    if gain > best_gain:
                        best_node, best_gain = u, gain
                
                if best_node is None or best_gain < 0:
                    break  # No more beneficial anchoring
                
                # Anchor this node
                anchored_nodes.add(best_node)
                lower_layer_nodes.remove(best_node)
                
                # Update requisite degrees
                for v in G.neighbors(best_node):
                    if v in requisite_degrees and requisite_degrees[v] > 0:
                        requisite_degrees[v] -= 1
                        total_requisite -= 1

            # Check if conversion is feasible with current budget
            if total_requisite > 2 * budget:
                continue  # Try next layer

            # Apply edge additions
            core_numbers = nx.core_number(G)
            k_core = {u for u in G.nodes if core_numbers[u] >= k}
            H = anchored_nodes | upper_layer_nodes | k_core

            # Find nodes that still need more neighbors
            nodes_needing_edges = [u for u in H if u in component_nodes and 
                                 sum(1 for v in G.neighbors(u) if v in H) < k]
            queue = nodes_needing_edges[:]

            # Add edges greedily
            while len(queue) >= 2 and len(edge_list) < budget:
                u, v = queue[0], queue[1]
                if not G.has_edge(u, v):
                    edge_list.append((u, v))
                    
                    # Update degrees and remove satisfied nodes
                    u_degree = sum(1 for x in G.neighbors(u) if x in H) + sum(1 for a, b in edge_list if a == u or b == u)
                    v_degree = sum(1 for x in G.neighbors(v) if x in H) + sum(1 for a, b in edge_list if a == v or b == v)
                    
                    if u_degree >= k:
                        queue.remove(u)
                    if v_degree >= k and v in queue:
                        queue.remove(v)
                    continue
                queue.pop(0)

            # Handle remaining nodes in queue
            for u in queue:
                if len(edge_list) >= budget:
                    break
                # Find any valid target
                possible_targets = [v for v in H if v != u and not G.has_edge(u, v)]
                if possible_targets:
                    edge_list.append((u, possible_targets[0]))

            # Return result if successful
            if len(edge_list) <= budget:
                return edge_list, anchored_nodes | upper_layer_nodes

        except Exception as e:
            # If anything goes wrong, try next layer
            continue

    # No feasible solution found
    return [], set()

def fastcm_plus_reinforce(G: nx.Graph, k: int = None, budget: int = 10) -> Tuple[nx.Graph, List[Tuple[int, int]]]:
    """
    Main FastCM+ algorithm implementation
    
    This was a really complex algorithm to implement - much harder than MRKC
    Had to handle multiple strategies and edge cases
    """
    if G.number_of_nodes() == 0:
        return G.copy(), []

    # Determine k value if not provided
    if k is None:
        core_vals = nx.core_number(G)
        if not core_vals:
            return G.copy(), []
        k = max(core_vals.values()) + 1

    # Check if (k-1)-shell exists
    core_numbers = nx.core_number(G)
    shell_nodes = {u for u, c in core_numbers.items() if c == k - 1}
    if not shell_nodes:
        return G.copy(), []  # Nothing to do

    # Get shell components
    components = get_shell_components_with_collapse_nodes(G, k)
    if not components:
        return G.copy(), []

    # Compute onion layers for partial conversion
    onion_layers = compute_onion_layers(G, k)

    # Generate conversion options for each component
    conversion_options = []
    for comp, collapse in components:
        # Try complete conversion
        try:
            cost = estimate_complete_conversion_cost(G, comp, collapse, k)
            if cost <= budget:
                conversion_options.append({
                    'type': 'complete', 
                    'component': comp, 
                    'collapse': collapse, 
                    'cost': cost, 
                    'followers': set(comp)
                })
        except Exception:
            pass  # Complete conversion not feasible
        
        # Try partial conversion
        try:
            edges, followers = apply_partial_conversion(G, comp, k, budget, onion_layers)
            if edges and len(edges) <= budget:
                conversion_options.append({
                    'type': 'partial', 
                    'component': comp, 
                    'collapse': set(), 
                    'cost': len(edges), 
                    'followers': followers, 
                    'edges': edges
                })
        except Exception:
            pass  # Partial conversion not feasible

    # Dynamic programming to select best combination of options
    # This part was really tricky to get right
    dp_table = [set() for _ in range(budget + 1)]
    selection_table = [None for _ in range(budget + 1)]

    for option in conversion_options:
        cost = option['cost']
        followers = option['followers']
        
        # Update DP table in reverse order (0-1 knapsack style)
        for b in range(budget, cost - 1, -1):
            combined_followers = dp_table[b - cost] | followers
            if len(combined_followers) > len(dp_table[b]):
                dp_table[b] = combined_followers
                if selection_table[b - cost] is None:
                    selection_table[b] = [option]
                else:
                    selection_table[b] = selection_table[b - cost] + [option]

    # Find best solution
    best_budget = max(range(budget + 1), key=lambda b: len(dp_table[b]))
    selected_options = selection_table[best_budget] if selection_table[best_budget] else []

    # Apply selected conversions
    G_new = G.copy()
    all_added_edges = []
    
    for option in selected_options:
        if option['type'] == 'complete':
            try:
                edges = apply_complete_conversion(G_new, option['component'], option['collapse'], k)
                for u, v in edges:
                    if len(all_added_edges) >= budget:
                        break
                    if not G_new.has_edge(u, v):
                        G_new.add_edge(u, v)
                        all_added_edges.append((u, v))
            except Exception:
                continue  # Skip if conversion fails
        elif option['type'] == 'partial':
            for u, v in option['edges']:
                if len(all_added_edges) >= budget:
                    break
                if not G_new.has_edge(u, v):
                    G_new.add_edge(u, v)
                    all_added_edges.append((u, v))

    # Use any remaining budget greedily
    # This is a simple fallback strategy
    remaining_budget = budget - len(all_added_edges)
    if remaining_budget > 0:
        core_numbers = nx.core_number(G_new)
        k_core = {u for u in G_new.nodes if core_numbers[u] >= k}
        shell_candidates = [u for u in G_new.nodes if core_numbers[u] == k - 1]
        
        for u in shell_candidates:
            if remaining_budget <= 0:
                break
            for v in k_core:
                if not G_new.has_edge(u, v):
                    G_new.add_edge(u, v)
                    all_added_edges.append((u, v))
                    remaining_budget -= 1
                    break

    return G_new, all_added_edges