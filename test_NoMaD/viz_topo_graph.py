import yaml
import matplotlib.pyplot as plt
import networkx as nx


def load_topomap_from_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    topomap_nodes = data["nodes"]  # list of dict
    # edges in the file are stored as str -> list, so convert keys to int if needed
    raw_edges = data["edges"]  # e.g., {'0': [1], '1': [0,2], ...}

    topomap_edges = {}
    for k, neighbors in raw_edges.items():
        k_idx = int(k)
        topomap_edges[k_idx] = set(neighbors)
    return topomap_nodes, topomap_edges


def visualize_topomap(topomap_nodes, topomap_edges):
    # same function as above
    G = nx.Graph()
    for node_info in topomap_nodes:
        node_id = node_info["idx"]
        pos_xy = node_info["pos"]
        G.add_node(node_id, pos=pos_xy)
    for src, nbrs in topomap_edges.items():
        for tgt in nbrs:
            G.add_edge(src, tgt)

    pos_dict = nx.get_node_attributes(G, "pos")
    plt.figure(figsize=(6, 6))
    nx.draw(G, pos=pos_dict, with_labels=True, node_size=500, node_color="skyblue")
    plt.axis("equal")
    plt.show()


# Example usage:
if __name__ == "__main__":
    # topomap_path = "\\topomaps\images\dynamic_map\nodes_info.yaml"
    nodes, edges = load_topomap_from_yaml("./topomaps/images/dynamic_map/nodes_info.yaml")
    visualize_topomap(nodes, edges)
