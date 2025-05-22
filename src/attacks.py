import networkx as nx
import random

def degree_attack(G, fraction=0.1):
    """
    Remove nodes with highest degree first.
    This is the standard targeted attack in network research.
    """
    G_copy = G.copy()
    num_remove = max(1, int(fraction * G.number_of_nodes()))
    
    # Sort by degree descending
    by_degree = sorted(G.degree, key=lambda x: x[1], reverse=True)
    to_remove = [node for node, _ in by_degree[:num_remove]]
    
    G_copy.remove_nodes_from(to_remove)
    return G_copy, to_remove


def kcore_attack(G, fraction=0.1):
    """
    Remove nodes with highest k-shell values first.
    Targets the dense core of the network.
    """
    G_copy = G.copy()
    num_remove = max(1, int(fraction * G.number_of_nodes()))
    
    core_nums = nx.core_number(G)
    
    # Sort by core number, use degree as tiebreaker
    by_core = sorted(G.nodes(), 
                     key=lambda x: (core_nums[x], G.degree[x]), 
                     reverse=True)
    
    to_remove = by_core[:num_remove]
    G_copy.remove_nodes_from(to_remove)
    return G_copy, to_remove


def betweenness_attack(G, fraction=0.1):
    """
    Remove nodes with highest betweenness centrality.
    Targets nodes that bridge different parts of the network.
    """
    G_copy = G.copy()
    num_remove = max(1, int(fraction * G.number_of_nodes()))
    
    # Use sampling for large graphs (standard practice)
    if G.number_of_nodes() > 1000:
        betweenness = nx.betweenness_centrality(G, k=500)
    else:
        betweenness = nx.betweenness_centrality(G)
    
    by_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
    to_remove = [node for node, _ in by_betweenness[:num_remove]]
    
    G_copy.remove_nodes_from(to_remove)
    return G_copy, to_remove


def random_attack(G, fraction=0.1, seed=42):
    """
    Remove random nodes. Used as baseline in resilience studies.
    """
    G_copy = G.copy()
    num_remove = max(1, int(fraction * G.number_of_nodes()))
    
    random.seed(seed)
    nodes = list(G.nodes())
    to_remove = random.sample(nodes, min(num_remove, len(nodes)))
    
    G_copy.remove_nodes_from(to_remove)
    return G_copy, to_remove


def attack_network(G, attack_type, fraction=0.1):
    """Run a specific attack on the network."""
    if attack_type == "degree":
        return degree_attack(G, fraction)
    elif attack_type == "kcore":
        return kcore_attack(G, fraction)
    elif attack_type == "betweenness":
        return betweenness_attack(G, fraction)
    elif attack_type == "random":
        return random_attack(G, fraction)
    else:
        raise ValueError(f"Unknown attack: {attack_type}")


if __name__ == "__main__":
    # Simple test
    G = nx.barabasi_albert_graph(100, 3, seed=42)
    print(f"Test network: {G.number_of_nodes()} nodes, max k-core: {max(nx.core_number(G).values())}")
    
    # Test each attack
    for attack in ["degree", "kcore", "betweenness", "random"]:
        G_attacked, removed = attack_network(G, attack, 0.1)
        max_core_after = max(nx.core_number(G_attacked).values()) if G_attacked.nodes() else 0
        components_after = nx.number_connected_components(G_attacked)
        
        print(f"{attack} attack: removed {len(removed)} nodes, "
              f"k-core: {max(nx.core_number(G).values())} → {max_core_after}, "
              f"components: 1 → {components_after}")