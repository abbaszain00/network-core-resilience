#!/usr/bin/env python3
"""
Simple SNAP dataset loader for the project.

I wanted to add some real datasets from SNAP to strengthen my network collection.
Focusing on smaller, manageable networks that won't overwhelm my evaluation.

Only including networks that:
1. Are reasonably sized (< 10k nodes for most)
2. Are well-documented and commonly used  
3. Represent different types (social, communication, collaboration)
4. Download quickly and process easily

Development notes:
- Started by trying to download huge networks but they were too slow
- Settled on these smaller ones that are still real datasets
- Added simple caching so I don't re-download every time
- Error handling because internet downloads can be flaky
"""

import networkx as nx
import pandas as pd
import requests
import gzip
import os
from pathlib import Path
import time

class SNAPNetworkLoader:
    """
    Load some real networks from SNAP repository.
    
    Keeping this simple - just download a few manageable networks
    and convert them to NetworkX format for my evaluation.
    """
    
    def __init__(self, cache_dir="snap_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # URLs for smaller SNAP networks that are manageable
        # Focused on commonly cited ones that aren't too huge
        self.network_urls = {
            'facebook_ego': {
                'url': 'https://snap.stanford.edu/data/facebook_combined.txt.gz',
                'description': 'Facebook ego networks combined',
                'expected_nodes': 4039,
                'type': 'social'
            },
            'email_enron': {
                'url': 'https://snap.stanford.edu/data/email-Enron.txt.gz', 
                'description': 'Enron email communication network',
                'expected_nodes': 36692,
                'type': 'communication'
            },
            'ca_grqc': {
                'url': 'https://snap.stanford.edu/data/ca-GrQc.txt.gz',
                'description': 'General Relativity collaboration network',
                'expected_nodes': 5242,
                'type': 'collaboration'
            }
        }
    
    def download_network(self, network_name, timeout=30):
        """
        Download a network from SNAP if we don't have it cached.
        
        Added timeout because some downloads were hanging.
        Basic error handling for network issues.
        """
        
        if network_name not in self.network_urls:
            print(f"Unknown network: {network_name}")
            return None
        
        info = self.network_urls[network_name]
        cache_file = self.cache_dir / f"{network_name}.txt"
        
        # Check if we already have it
        if cache_file.exists():
            print(f"Using cached {network_name}")
            return cache_file
        
        print(f"Downloading {network_name} from SNAP...")
        print(f"  Description: {info['description']}")
        print(f"  Expected size: ~{info['expected_nodes']} nodes")
        
        try:
            # Download with timeout
            response = requests.get(info['url'], timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Handle gzipped files
            if info['url'].endswith('.gz'):
                content = gzip.decompress(response.content).decode('utf-8')
            else:
                content = response.text
            
            # Save to cache
            with open(cache_file, 'w') as f:
                f.write(content)
            
            print(f"  Downloaded and cached {network_name}")
            return cache_file
            
        except requests.exceptions.Timeout:
            print(f"  Timeout downloading {network_name} - skipping")
            return None
        except requests.exceptions.RequestException as e:
            print(f"  Error downloading {network_name}: {e}")
            return None
        except Exception as e:
            print(f"  Unexpected error with {network_name}: {e}")
            return None
    
    def load_snap_network(self, network_name):
        """
        Load a SNAP network into NetworkX format.
        
        SNAP files have different formats so need to handle various cases.
        Most are simple edge lists but some have comments/headers.
        """
        
        cache_file = self.download_network(network_name)
        if not cache_file:
            return None
        
        try:
            print(f"Loading {network_name} into NetworkX...")
            
            # Read the edge list file
            # SNAP files usually have format: node1 node2
            # Some have comments starting with # 
            edges = []
            with open(cache_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if line.startswith('#') or not line:
                        continue
                    
                    # Parse edge - usually space or tab separated
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            # Convert to integers (SNAP usually uses integer node IDs)
                            node1, node2 = int(parts[0]), int(parts[1])
                            edges.append((node1, node2))
                        except ValueError:
                            # Skip malformed lines
                            continue
            
            if not edges:
                print(f"  No valid edges found in {network_name}")
                return None
            
            # Create NetworkX graph
            G = nx.Graph()
            G.add_edges_from(edges)
            
            # Remove self-loops if any (common in some datasets)
            G.remove_edges_from(nx.selfloop_edges(G))
            
            # Take largest connected component (standard practice)
            if not nx.is_connected(G):
                largest_cc = max(nx.connected_components(G), key=len)
                G = G.subgraph(largest_cc).copy()
            
            print(f"  Loaded {network_name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            
            # Basic sanity check
            expected = self.network_urls[network_name]['expected_nodes']
            actual = G.number_of_nodes()
            if abs(actual - expected) > expected * 0.2:  # Allow 20% difference
                print(f"  Warning: Size difference from expected ({expected} vs {actual})")
            
            return G
            
        except Exception as e:
            print(f"  Error loading {network_name}: {e}")
            return None
    
    def get_snap_networks(self, max_nodes=10000):
        """
        Load all available SNAP networks that fit size constraints.
        
        This is the main function I'll call from my network loading.
        Keeps things simple - try to load each network, skip if it fails.
        """
        
        print("Loading SNAP networks...")
        networks = {}
        
        for network_name, info in self.network_urls.items():
            # Skip networks that are too large
            if info['expected_nodes'] > max_nodes:
                print(f"Skipping {network_name} - too large ({info['expected_nodes']} nodes)")
                continue
            
            G = self.load_snap_network(network_name)
            if G is not None:
                # Final size check after processing
                if G.number_of_nodes() <= max_nodes:
                    networks[network_name] = G
                else:
                    print(f"Skipping {network_name} - too large after processing")
        
        print(f"Successfully loaded {len(networks)} SNAP networks")
        return networks

# Convenience function for integration
def get_snap_networks(max_nodes=10000):
    """Simple interface for loading SNAP networks."""
    loader = SNAPNetworkLoader()
    return loader.get_snap_networks(max_nodes)

if __name__ == "__main__":
    # Test the loader
    print("Testing SNAP network loader...")
    
    loader = SNAPNetworkLoader()
    networks = loader.get_snap_networks(max_nodes=5000)
    
    print(f"\nLoaded SNAP networks:")
    for name, G in networks.items():
        cores = nx.core_number(G)
        max_core = max(cores.values()) if cores else 0
        density = nx.density(G)
        print(f"  {name}: {G.number_of_nodes()} nodes, max k-core: {max_core}, density: {density:.4f}")
    
    print(f"\nReady to integrate into evaluation!")