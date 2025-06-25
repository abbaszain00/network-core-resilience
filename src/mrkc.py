"""
MRKC Algorithm Implementation

This implements the algorithm from Laishram et al. (2018) paper.
Had to read the paper multiple times to understand the core influence calculation.
The edge selection part was particularly tricky to get right.

Main challenges:
1. Understanding the core influence formula (equation 4) - took forever
2. Making sure edges preserve core numbers (kept breaking this)
3. Implementing the priority scoring correctly
4. Getting reasonable performance on larger graphs

Version history:
- v1: Basic implementation, didn't preserve core numbers properly
- v2: Added core preservation check, way too slow for large graphs
- v3: Current version with candidate filtering optimizations
- v4: Fixed bug where same edge could be added twice

References:
- Laishram et al. "Measuring and Improving the Core Resilience of Networks" WWW 2018
"""

import networkx as nx
import itertools
import random
from collections import defaultdict

def core_strength(G, u, core_numbers):
    """Calculate core strength as defined in the MRKC paper equation 3"""
    # Formula from paper equation 3 - took me a while to understand this part
    ku = core_numbers[u]
    neighbors_with_higher_or_equal_core = [v for v in G.neighbors(u) if core_numbers[v] >= ku]
    # Originally had this wrong - was using > instead of >= (missed the equal part)
    return len(neighbors_with_higher_or_equal_core) - ku + 1

def compute_core_influence(G, core_numbers):
    """Compute core influence following the paper's iterative approach in section 4.3"""
    # This was definitely the hardest part to implement - had to read section 4.3 like 10 times
    # Still not 100% sure I got it exactly right but results look reasonable
    core_influence = {node: 1.0 for node in G.nodes()}
    
    # Find nodes in V_delta (nodes that depend on higher-core neighbors)
    # Paper's definition was confusing at first
    V_delta = set()
    for u in G.nodes():
        same_core_neighbors = sum(1 for v in G.neighbors(u) if core_numbers[v] == core_numbers[u])
        if same_core_neighbors < core_numbers[u]:
            V_delta.add(u)
    
    # Process nodes by increasing core number - this ordering is important according to paper
    max_core = max(core_numbers.values())
    for k in range(1, max_core + 1):
        nodes_at_this_core = [n for n in G.nodes() if core_numbers[n] == k and n in V_delta]
        
        for node in nodes_at_this_core:
            same_core_count = sum(1 for v in G.neighbors(node) if core_numbers[v] == k)
            # Delta calculation from equation 4
            delta = 1.0 - (same_core_count / k) if k > 0 else 0.0
            
            higher_core_neighbors = [v for v in G.neighbors(node) if core_numbers[v] > k]
            if higher_core_neighbors:
                contribution = delta * core_influence[node] / len(higher_core_neighbors)
                for neighbor in higher_core_neighbors:
                    core_influence[neighbor] += contribution
    
    return core_influence

def find_candidate_edges(G, core_numbers, limit=1000):
    """Find potential edges to add, focusing on promising candidates"""
    # Keep track of existing edges - there's probably a more efficient way to do this
    # but this approach works and is easy to understand
    existing_edges = set()
    for u, v in G.edges():
        existing_edges.add((u, v))
        existing_edges.add((v, u))  # Add both directions since graph is undirected
    
    candidates = []
    
    # Group nodes by their core numbers - this approach worked better than my first attempt
    # where I was just trying random edges
    core_groups = defaultdict(list)
    for node, core_val in core_numbers.items():
        core_groups[core_val].append(node)
    
    # Strategy 1: Try edges within the same core level first
    # Paper suggests these are more likely to preserve core numbers
    for core_val, nodes in core_groups.items():
        for u, v in itertools.combinations(nodes, 2):
            if (u, v) not in existing_edges and (v, u) not in existing_edges:
                candidates.append((u, v))
                if len(candidates) >= limit // 2:  # Stop early to avoid explosion
                    break
        if len(candidates) >= limit // 2:
            break
    
    # Strategy 2: Try edges between adjacent core levels
    # This seemed like a reasonable heuristic based on the paper's discussion
    sorted_cores = sorted(core_groups.keys())
    for i in range(len(sorted_cores) - 1):
        if len(candidates) >= int(limit * 0.8):  # Leave some room for random edges
            break
        
        curr_core = sorted_cores[i]
        next_core = sorted_cores[i + 1]
        
        # Limit to first 50 nodes from each group to avoid combinatorial explosion
        # This was a practical compromise - larger graphs were taking forever
        curr_nodes = core_groups[curr_core][:50]
        next_nodes = core_groups[next_core][:50]
        
        for u in curr_nodes:
            for v in next_nodes:
                if (u, v) not in existing_edges and (v, u) not in existing_edges:
                    candidates.append((u, v))
                    if len(candidates) >= int(limit * 0.8):
                        break
            if len(candidates) >= int(limit * 0.8):
                break
    
    # Strategy 3: Add some random edges for diversity
    # Sometimes the systematic approaches miss good candidates
    if len(candidates) < limit:
        all_nodes = list(G.nodes())
        attempts = 0
        while len(candidates) < limit and attempts < limit:  # Avoid infinite loop
            u, v = random.sample(all_nodes, 2)
            if (u, v) not in existing_edges and (v, u) not in existing_edges:
                # Make sure we don't add duplicates
                if (u, v) not in candidates and (v, u) not in candidates:
                    candidates.append((u, v))
            attempts += 1
    
    return candidates[:limit]

def mrkc_reinforce(G, budget=10, max_candidates=1000, verbose=False):
    """
    MRKC reinforcement algorithm implementation
    
    Note: Had to debug this extensively - my original version was adding edges 
    that changed core numbers which defeats the whole purpose
    """
    if verbose:
        print(f"Starting MRKC on graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    G_result = G.copy()
    original_cores = nx.core_number(G)
    
    # Calculate core strength and influence for all nodes
    # This part took me ages to get right, especially the influence calculation
    node_strength = {u: core_strength(G, u, original_cores) for u in G.nodes()}
    node_influence = compute_core_influence(G, original_cores)
    
    # Find candidate edges using our filtering strategies
    candidates = find_candidate_edges(G, original_cores, max_candidates)
    
    if verbose:
        total_possible = G.number_of_nodes() * (G.number_of_nodes() - 1) // 2
        print(f"Found {len(candidates)} candidates out of {total_possible} possible edges")
    
    if not candidates:
        if verbose:
            print("No valid candidates found")  # This happened a lot during testing on small graphs
        return G_result, []
    
    # Check which candidates actually preserve core numbers
    # This is the expensive part but necessary to follow the paper's constraints
    valid_edges = []
    for edge in candidates:
        u, v = edge
        # Test by adding edge to original graph
        test_graph = G.copy()
        test_graph.add_edge(u, v)
        new_cores = nx.core_number(test_graph)
        
        # Check if all nodes kept their original core numbers
        # Paper emphasizes this constraint is essential
        if all(new_cores[node] == original_cores[node] for node in G.nodes()):
            valid_edges.append(edge)
    
    if verbose:
        print(f"Valid candidates after core preservation check: {len(valid_edges)}")
    
    if not valid_edges:
        # This can happen on some graphs - not much we can do
        return G_result, []
    
    # Score edges using the MRKC priority formula from equation 7
    def safe_divide(numerator, denominator):
        # Avoid division by zero - return high priority if denominator is 0
        return float('inf') if denominator == 0 else numerator / denominator
    
    edge_scores = []
    for edge in valid_edges:
        u, v = edge
        u_core, v_core = original_cores[u], original_cores[v]
        
        # Priority calculation based on paper's formula - had to implement this carefully
        if u_core < v_core:
            score = safe_divide(node_influence[u], node_strength[u])
        elif u_core > v_core:
            score = safe_divide(node_influence[v], node_strength[v])
        else:  # equal cores
            score = safe_divide(node_influence[u], node_strength[u]) + safe_divide(node_influence[v], node_strength[v])
        
        edge_scores.append((edge, score))
    
    # Sort by score (highest first) and add edges greedily
    edge_scores.sort(key=lambda x: x[1], reverse=True)
    edges_added = []
    
    for edge, score in edge_scores:
        if len(edges_added) >= budget:
            break
        
        u, v = edge
        # Double-check edge addition doesn't break anything
        # Paranoid but better safe than sorry after all the debugging
        temp_graph = G_result.copy()
        temp_graph.add_edge(u, v)
        temp_cores = nx.core_number(temp_graph)
        
        if all(temp_cores[n] == original_cores[n] for n in G.nodes()):
            G_result.add_edge(u, v)
            edges_added.append(edge)
    
    if verbose:
        print(f"Successfully added {len(edges_added)} edges")
    
    return G_result, edges_added

def test_implementation():
    """Test the implementation on some graphs - used this during development"""
    print("Testing MRKC implementation...")
    
    # Test on small graph first
    G = nx.karate_club_graph()
    reinforced, edges = mrkc_reinforce(G, budget=5, verbose=True)
    print(f"Karate club result: added {len(edges)} edges: {edges}")
    
    # Test performance on larger graphs
    test_sizes = [100, 500]  # Start small, work up
    for n in test_sizes:
        print(f"\nTesting {n}-node Barabasi-Albert graph:")
        G = nx.barabasi_albert_graph(n, 3)
        
        import time
        start_time = time.time()
        reinforced, edges = mrkc_reinforce(G, budget=10, verbose=True)
        elapsed = time.time() - start_time
        
        print(f"Time taken: {elapsed:.2f} seconds")
        print(f"Edges added: {len(edges)}")
        
        # Basic sanity check
        if len(edges) > 0:
            print("✓ Algorithm successfully added edges")
        else:
            print("⚠ No edges were added - might need to investigate")

if __name__ == "__main__":
    test_implementation()