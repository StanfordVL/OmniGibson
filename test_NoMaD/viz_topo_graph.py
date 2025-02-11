import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import yaml


def load_topomap(yaml_path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    topomap_nodes = data["nodes"]
    raw_edges = data["edges"]
    topomap_edges = {}
    for k, neighbors in raw_edges.items():
        k_int = int(k)
        topomap_edges[k_int] = set(neighbors)
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


def visualize_topomap_with_images(topomap_nodes, topomap_edges):
    """
    Visualize the topological graph with each node's image placed at its (x,y) position
    and display the node index above each image.
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import matplotlib.image as mpimg
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    # 1) Build a NetworkX graph
    G = nx.Graph()
    for node_info in topomap_nodes:
        node_id = node_info["idx"]
        pos_xy = node_info["pos"]  # (x, y)
        img_path = node_info["img_path"]
        G.add_node(node_id, pos=pos_xy, img_path=img_path)

    for src, neighbors in topomap_edges.items():
        for tgt in neighbors:
            G.add_edge(src, tgt)

    # 2) Extract positions for drawing
    pos_dict = nx.get_node_attributes(G, "pos")

    # 3) Create figure/axis
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw edges in gray
    nx.draw_networkx_edges(G, pos_dict, ax=ax, edge_color="gray")

    # Optionally, you could skip the built-in labels from NetworkX
    # since we'll place our own labels near the images. But let's leave them for reference:
    # nx.draw_networkx_labels(G, pos_dict, ax=ax, font_weight="bold")

    # 4) Place each node's image and its label
    for node_id, data in G.nodes(data=True):
        x, y = data["pos"]
        img_path = data["img_path"]

        # Attempt to load the node image
        try:
            arr_img = mpimg.imread(img_path)
        except FileNotFoundError:
            print(f"[WARNING] Could not find image at {img_path}, skipping.")
            continue

        # Create the image annotation
        imagebox = OffsetImage(arr_img, zoom=0.3)  # Adjust zoom as needed
        ab = AnnotationBbox(
            imagebox,
            (x, y),
            frameon=True,  # or False if no bounding box
            pad=0.0,
        )
        ax.add_artist(ab)

        # Place a text label (e.g., "Node 3") slightly above the image
        # The offset (0.2) can be tweaked based on zoom or your coordinate scale
        label_offset = 0.2
        ax.text(
            x,
            y + label_offset,
            f"Node {node_id}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="red",
            fontweight="bold",
        )

    # Keep aspect ratio equal so positions look correct
    plt.axis("equal")
    plt.title("Topological Graph with Node Images + Labels")
    plt.show()


# Example usage:
if __name__ == "__main__":
    # topomap_path = "\\topomaps\images\dynamic_map\nodes_info.yaml"
    yaml_path = "./topomaps/images/dynamic_map/nodes_info.yaml"
    topomap_nodes, topomap_edges = load_topomap(yaml_path)
    # visualize_topomap(topomap_nodes, topomap_edges)

    visualize_topomap_with_images(topomap_nodes, topomap_edges)
