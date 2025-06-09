import networkx as nx
import itertools
import random
from collections import defaultdict

def core_strength(G, u, core_numbers):
    """Calculate core strength as defined in the MRKC paper"""
    ku = core_numbers[u]
    neighbors_with_higher_or_equal_core = [v for v in G.neighbors(u) if core_numbers[v] >= ku]
    return len(neighbors_with_higher_or_equal_core) - ku + 1

def compute_core_influence(G, core_numbers):
    """Compute core influence following the paper's iterative approach"""
    core_influence = {node: 1.0 for node in G.nodes()}
    
    # Find nodes in V_delta (nodes that depend on higher-core neighbors)
    V_delta = set()
    for u in G.nodes():
        same_core_neighbors = sum(1 for v in G.neighbors(u) if core_numbers[v] == core_numbers[u])
        if same_core_neighbors < core_numbers[u]:
            V_delta.add(u)
    
    # Process nodes by increasing core number
    max_core = max(core_numbers.values())
    for k in range(1, max_core + 1):
        nodes_at_this_core = [n for n in G.nodes() if core_numbers[n] == k and n in V_delta]
        
        for node in nodes_at_this_core:
            same_core_count = sum(1 for v in G.neighbors(node) if core_numbers[v] == k)
            delta = 1.0 - (same_core_count / k) if k > 0 else 0.0
            
            higher_core_neighbors = [v for v in G.neighbors(node) if core_numbers[v] > k]
            if higher_core_neighbors:
                contribution = delta * core_influence[node] / len(higher_core_neighbors)
                for neighbor in higher_core_neighbors:
                    core_influence[neighbor] += contribution
    
    return core_influence

def find_candidate_edges(G, core_numbers, limit=1000):
    """Find potential edges to add, focusing on promising candidates"""
    existing = set()
    for u, v in G.edges():
        existing.add((u, v))
        existing.add((v, u))  # Add both directions for undirected graph
    
    candidates = []
    
    # Group nodes by their core numbers
    core_groups = defaultdict(list)
    for node, core_val in core_numbers.items():
        core_groups[core_val].append(node)
    
    # Strategy 1: Try edges within the same core level first
    for core_val, nodes in core_groups.items():
        for u, v in itertools.combinations(nodes, 2):
            if (u, v) not in existing and (v, u) not in existing:
                candidates.append((u, v))
                if len(candidates) >= limit // 2:
                    break
        if len(candidates) >= limit // 2:
            break
    
    # Strategy 2: Try edges between adjacent core levels
    sorted_cores = sorted(core_groups.keys())
    for i in range(len(sorted_cores) - 1):
        if len(candidates) >= int(limit * 0.8):
            break
        
        curr_core = sorted_cores[i]
        next_core = sorted_cores[i + 1]
        
        # Limit to first 50 nodes from each group to avoid explosion
        curr_nodes = core_groups[curr_core][:50]
        next_nodes = core_groups[next_core][:50]
        
        for u in curr_nodes:
            for v in next_nodes:
                if (u, v) not in existing and (v, u) not in existing:
                    candidates.append((u, v))
                    if len(candidates) >= int(limit * 0.8):
                        break
            if len(candidates) >= int(limit * 0.8):
                break
    
    # Strategy 3: Add some random edges for diversity
    if len(candidates) < limit:
        all_nodes = list(G.nodes())
        attempts = 0
        while len(candidates) < limit and attempts < limit:
            u, v = random.sample(all_nodes, 2)
            if (u, v) not in existing and (v, u) not in existing:
                if (u, v) not in candidates and (v, u) not in candidates:
                    candidates.append((u, v))
            attempts += 1
    
    return candidates[:limit]

def mrkc_reinforce(G, budget=10, max_candidates=1000, verbose=False):
    """
    MRKC reinforcement algorithm implementation
    """
    if verbose:
        print(f"Starting MRKC on graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    G_result = G.copy()
    original_cores = nx.core_number(G)
    
    # Calculate core strength and influence for all nodes
    strength = {u: core_strength(G, u, original_cores) for u in G.nodes()}
    influence = compute_core_influence(G, original_cores)
    
    # Find candidate edges
    candidates = find_candidate_edges(G, original_cores, max_candidates)
    
    if verbose:
        total_possible = G.number_of_nodes() * (G.number_of_nodes() - 1) // 2
        print(f"Found {len(candidates)} candidates out of {total_possible} possible edges")
    
    if not candidates:
        if verbose:
            print("No valid candidates found")
        return G_result, []
    
    # Check which candidates actually preserve core numbers
    valid_edges = []
    for edge in candidates:
        u, v = edge
        # Test by adding edge to original graph
        test_graph = G.copy()
        test_graph.add_edge(u, v)
        new_cores = nx.core_number(test_graph)
        
        # Check if all nodes kept their original core numbers
        if all(new_cores[node] == original_cores[node] for node in G.nodes()):
            valid_edges.append(edge)
    
    if verbose:
        print(f"Valid candidates after core preservation check: {len(valid_edges)}")
    
    if not valid_edges:
        return G_result, []
    
    # Score edges using the MRKC priority formula
    def safe_divide(numerator, denominator):
        return float('inf') if denominator == 0 else numerator / denominator
    
    edge_scores = []
    for edge in valid_edges:
        u, v = edge
        u_core, v_core = original_cores[u], original_cores[v]
        
        if u_core < v_core:
            score = safe_divide(influence[u], strength[u])
        elif u_core > v_core:
            score = safe_divide(influence[v], strength[v])
        else:  # equal cores
            score = safe_divide(influence[u], strength[u]) + safe_divide(influence[v], strength[v])
        
        edge_scores.append((edge, score))
    
    # Sort by score (highest first) and add edges
    edge_scores.sort(key=lambda x: x[1], reverse=True)
    added = []
    
    for edge, score in edge_scores:
        if len(added) >= budget:
            break
        
        u, v = edge
        # Double-check edge addition doesn't break anything
        temp_graph = G_result.copy()
        temp_graph.add_edge(u, v)
        temp_cores = nx.core_number(temp_graph)
        
        if all(temp_cores[n] == original_cores[n] for n in G.nodes()):
            G_result.add_edge(u, v)
            added.append(edge)
    
    if verbose:
        print(f"Successfully added {len(added)} edges")
    
    return G_result, added

def test_implementation():
    """Test the implementation on some graphs"""
    print("Testing MRKC implementation...")
    
    # Test on small graph first
    G = nx.karate_club_graph()
    reinforced, edges = mrkc_reinforce(G, budget=5, verbose=True)
    print(f"Karate club result: added edges {edges}")
    
    # Test performance on larger graphs
    sizes = [100, 500]
    for n in sizes:
        print(f"\nTesting {n}-node graph:")
        G = nx.barabasi_albert_graph(n, 3)
        
        import time
        start = time.time()
        reinforced, edges = mrkc_reinforce(G, budget=10, verbose=True)
        elapsed = time.time() - start
        
        print(f"Time taken: {elapsed:.2f} seconds")
        print(f"Edges added: {len(edges)}")

if __name__ == "__main__":
    test_implementation()