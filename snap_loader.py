#!/usr/bin/env python3
"""
SNAP dataset loader for network resilience experiments.
"""

import networkx as nx
import requests
import gzip
import os
from pathlib import Path

class SNAPNetworkLoader:
    """Load networks from the Stanford SNAP repository."""
    
    def __init__(self, cache_dir="snap_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.network_urls = {
            'facebook_ego': {
                'url': 'https://snap.stanford.edu/data/facebook_combined.txt.gz',
                'description': 'Facebook ego networks',
                'expected_nodes': 4039,
                'type': 'social'
            },
            'ca_grqc': {
                'url': 'https://snap.stanford.edu/data/ca-GrQc.txt.gz',
                'description': 'General Relativity collaboration',
                'expected_nodes': 5242,
                'type': 'collaboration'
            },
            'bitcoin_alpha': {
                'url': 'https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz',
                'description': 'Bitcoin Alpha web of trust network',
                'expected_nodes': 3783,
                'type': 'trust'
            },
            'bitcoin_otc': {
                'url': 'https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz',
                'description': 'Bitcoin OTC web of trust network',
                'expected_nodes': 5881,
                'type': 'trust'
            },
            'wiki_vote': {
                'url': 'https://snap.stanford.edu/data/wiki-Vote.txt.gz',
                'description': 'Wikipedia who-votes-on-whom network',
                'expected_nodes': 7115,
                'type': 'voting'
            },
            'email_eu_core': {
                'url': 'https://snap.stanford.edu/data/email-Eu-core.txt.gz',
                'description': 'Email network from European research institution',
                'expected_nodes': 1005,
                'type': 'communication'
            }
        }
    
    def download_network(self, network_name, timeout=30):
        """Download network data from SNAP repository."""
        
        if network_name not in self.network_urls:
            print(f"Unknown network: {network_name}")
            return None
        
        info = self.network_urls[network_name]
        cache_file = self.cache_dir / f"{network_name}.txt"
        
        if cache_file.exists():
            print(f"Using cached {network_name}")
            return cache_file
        
        print(f"Downloading {network_name}...")
        
        try:
            response = requests.get(info['url'], timeout=timeout, stream=True)
            response.raise_for_status()
            
            if info['url'].endswith('.gz'):
                content = gzip.decompress(response.content).decode('utf-8')
            else:
                content = response.text
            
            with open(cache_file, 'w') as f:
                f.write(content)
            
            print(f"Cached {network_name}")
            return cache_file
            
        except requests.exceptions.Timeout:
            print(f"Download timeout for {network_name}")
            return None
        except Exception as e:
            print(f"Download error for {network_name}: {e}")
            return None
    
    def load_snap_network(self, network_name):
        """Load SNAP network into NetworkX format."""
        
        cache_file = self.download_network(network_name)
        if not cache_file:
            return None
        
        try:
            edges = []
            with open(cache_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or not line:
                        continue
                    
                    parts = line.split(',') if 'bitcoin' in network_name else line.split()
                    if len(parts) >= 2:
                        try:
                            node1, node2 = int(parts[0]), int(parts[1])
                            edges.append((node1, node2))
                        except ValueError:
                            continue
            
            if not edges:
                print(f"No valid edges in {network_name}")
                return None
            
            # Handle signed networks (bitcoin networks have weights/signs)
            if 'bitcoin' in network_name:
                G = nx.Graph()  # Convert to undirected for simplicity
                G.add_edges_from([(u, v) for u, v in edges])
            else:
                G = nx.Graph()
                G.add_edges_from(edges)
            
            G.remove_edges_from(nx.selfloop_edges(G))
            
            # Take largest connected component
            if not nx.is_connected(G):
                largest_cc = max(nx.connected_components(G), key=len)
                G = G.subgraph(largest_cc).copy()
            
            print(f"Loaded {network_name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            
            return G
            
        except Exception as e:
            print(f"Error loading {network_name}: {e}")
            return None
    
    def get_snap_networks(self, max_nodes=10000):
        """Load all SNAP networks within size constraints."""
        
        print("Loading SNAP networks...")
        networks = {}
        
        for network_name, info in self.network_urls.items():
            if info['expected_nodes'] > max_nodes:
                print(f"Skipping {network_name}: {info['expected_nodes']} nodes > {max_nodes}")
                continue
            
            G = self.load_snap_network(network_name)
            if G is not None and G.number_of_nodes() <= max_nodes:
                networks[network_name] = G
        
        print(f"Loaded {len(networks)} SNAP networks")
        return networks
    
    def get_network_info(self):
        """Print information about available networks."""
        print("Available SNAP networks:")
        for name, info in self.network_urls.items():
            print(f"  {name}: {info['expected_nodes']} nodes - {info['description']} ({info['type']})")

def get_snap_networks(max_nodes=10000):
    """Load SNAP networks with size limit."""
    loader = SNAPNetworkLoader()
    return loader.get_snap_networks(max_nodes)

if __name__ == "__main__":
    print("Testing SNAP loader...")
    
    loader = SNAPNetworkLoader()
    loader.get_network_info()
    print()
    
    networks = get_snap_networks(max_nodes=6000)
    
    print(f"\nLoaded networks:")
    for name, G in networks.items():
        cores = nx.core_number(G)
        max_core = max(cores.values()) if cores else 0
        density = nx.density(G)
        print(f"  {name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, max k-core: {max_core}, density: {density:.4f}")