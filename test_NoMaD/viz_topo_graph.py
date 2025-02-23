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


def visualize_topomap_with_images(topomap_nodes, topomap_edges):
    """
    Visualize the topological graph with each node's image placed at its (x,y) position,
    displaying the node index above each image, plus (x, y, yaw) info below the image.
    """
    G = nx.Graph()

    # 1) Build a NetworkX graph with node attributes
    for node_info in topomap_nodes:
        node_id = node_info["idx"]
        pos_xy = node_info["pos"]  # (x, y)
        img_path = node_info["img_path"]
        yaw_val = node_info.get("yaw", 0.0)  # if 'yaw' not found, default 0

        # Store 'yaw' in the node's data as well
        G.add_node(node_id, pos=pos_xy, img_path=img_path, yaw=yaw_val)

    # Add edges
    for src, neighbors in topomap_edges.items():
        for tgt in neighbors:
            G.add_edge(src, tgt)

    # 2) Positions for drawing edges
    pos_dict = nx.get_node_attributes(G, "pos")

    # 3) Create figure/axis
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw edges in gray
    nx.draw_networkx_edges(G, pos_dict, ax=ax, edge_color="gray")

    # 4) Place each node's image and labels
    for node_id, data in G.nodes(data=True):
        x, y = data["pos"]
        img_path = data["img_path"]
        yaw_val = data["yaw"]  # stored above

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

        # (A) Node ID label (above image)
        label_offset_above = 0.25
        ax.text(
            x,
            y + label_offset_above,
            f"Node {node_id}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="red",
            fontweight="bold",
        )

        # (B) (x, y, yaw) label (below image)
        label_offset_below = -0.20
        ax.text(
            x,
            y + label_offset_below,
            f"X={x:.2f}, Y={y:.2f}\nYaw={yaw_val:.2f}",
            ha="center",
            va="top",
            fontsize=8,
            color="black",
        )

    # Keep aspect ratio equal
    plt.axis("equal")
    plt.title("Topological Graph with Node Images + Labels")
    plt.show()


if __name__ == "__main__":
    yaml_path = "./topomaps/images/dynamic_map/nodes_info.yaml"
    topomap_nodes, topomap_edges = load_topomap(yaml_path)
    visualize_topomap_with_images(topomap_nodes, topomap_edges)
