import matplotlib.pyplot as plt
import networkx as nx

def draw_graph(G, title="Graph", node_color='skyblue'):
    pos = nx.spring_layout(G, seed=42)  # layout algorithm for consistent positions

    plt.figure(figsize=(6, 6))
    nx.draw(
        G, pos,
        with_labels=True,
        node_color=node_color,
        edge_color='gray',
        node_size=500,
        font_size=10
    )
    plt.title(title)
    plt.show()
