import os
import yaml
import numpy as np
import torch
import omnigibson as og
import shutil
import time
from PIL import Image as PILImage
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# Local imports (change paths if necessary)
from deployment.src.utils import to_numpy, transform_images, load_model
from train.vint_train.training.train_utils import get_action

##############################################################################
# Global / Config
##############################################################################
work_dir = os.getcwd()
MODEL_DEPLOY_PATH = os.path.join(work_dir, "test_NoMaD", "deployment")
MODEL_TRAIN_PATH = os.path.join(work_dir, "test_NoMaD", "train")
MODEL_CONFIG_PATH = os.path.join(MODEL_DEPLOY_PATH, "config", "models.yaml")
ROBOT_CONFIG_PATH = os.path.join(MODEL_DEPLOY_PATH, "config", "robot.yaml")

with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)

RATE = 10
MAX_V = robot_config.get("max_v", 0.31)
MAX_W = robot_config.get("max_w", 1.90)

DT = 1.0
EPS = 1e-8

# Path to store topomap images
TOPOMAP_IMAGES_DIR = "./topomaps/images"
MAP_NAME = "dynamic_map"

# We'll remove the old topomaps/dynamic_map folder once before running multiple episodes
map_dir = os.path.join(TOPOMAP_IMAGES_DIR, MAP_NAME)
if os.path.isdir(map_dir):
    print(f"[INFO] Removing old files in {map_dir}")
    shutil.rmtree(map_dir)
os.makedirs(map_dir, exist_ok=True)

# Thresholds
ADD_NODE_DIST = 0.5
EDGE_DISTANCE_THRESHOLD = 3.0

# We will reset these for each episode
node_index = 0
topomap_nodes = []
adj_list = {}

##############################################################################
# Utility Functions
##############################################################################


def clip_angle(theta: float) -> float:
    theta = np.mod(theta, 2.0 * np.pi)
    if theta > np.pi:
        theta -= 2.0 * np.pi
    return theta


def pd_controller(
    waypoint: np.ndarray, dt: float, max_v: float, max_w: float
) -> np.ndarray:
    assert len(waypoint) in [2, 4], "waypoint must be 2D or 4D"
    dx, dy = waypoint[:2]
    hx, hy = (0.0, 0.0)
    if len(waypoint) == 4:
        hx, hy = waypoint[2:4]

    if len(waypoint) == 4 and (abs(dx) < EPS and abs(dy) < EPS):
        v = 0.0
        heading_angle = clip_angle(np.arctan2(hy, hx))
        w = heading_angle / dt
    else:
        if abs(dx) < EPS:
            v = 0.0
            w = np.sign(dy) * (np.pi / (2 * dt))
        else:
            v = dx / dt
            angle = clip_angle(np.arctan2(dy, dx))
            w = angle / dt
    v = np.clip(v, 0, max_v)
    w = np.clip(w, -max_w, max_w)
    return np.array([v, w], dtype=np.float32)


def array_to_pil(rgb_tensor: torch.Tensor) -> PILImage.Image:
    if rgb_tensor.shape[-1] == 4:
        rgb_tensor = rgb_tensor[..., :3]
    arr = rgb_tensor.cpu().numpy().astype(np.uint8)
    return PILImage.fromarray(arr)


##############################################################################
# Graph Helpers
##############################################################################


def add_edge(u: int, v: int, dist_val: float):
    """
    Store the distance as a python float (not numpy float64).
    """
    global adj_list
    dist_python = float(dist_val)
    if u not in adj_list:
        adj_list[u] = []
    if v not in adj_list:
        adj_list[v] = []
    adj_list[u].append((v, dist_python))
    adj_list[v].append((u, dist_python))


def dijkstra(start_idx: int) -> dict:
    import heapq

    dist_map = {}
    visited = set()
    for n in adj_list.keys():
        dist_map[n] = float("inf")
    dist_map[start_idx] = 0.0
    pq = [(0.0, start_idx)]
    while pq:
        cur_dist, node = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        for nbr, cost in adj_list[node]:
            alt = cur_dist + cost
            if alt < dist_map[nbr]:
                dist_map[nbr] = alt
                heapq.heappush(pq, (alt, nbr))
    return dist_map


def get_closest_node_to_pos(xy: np.ndarray) -> int:
    if len(topomap_nodes) == 0:
        return -1
    best_idx, best_dist = -1, float("inf")
    for node in topomap_nodes:
        dist = np.linalg.norm(xy - np.array(node["pos"]))
        if dist < best_dist:
            best_dist = dist
            best_idx = node["idx"]
    return best_idx


##############################################################################
# Node & Edge Creation
##############################################################################


def add_node(
    obs_img: PILImage.Image,
    robot_pos_xy: np.ndarray,
    robot_yaw: float,
    episode_map_dir: str,
):
    """
    Save the node's image inside the episode-specific directory.
    """
    global node_index, topomap_nodes, adj_list
    filename = f"{node_index}.png"
    save_path = os.path.join(episode_map_dir, filename)
    obs_img.save(save_path)

    x_float = float(robot_pos_xy[0])
    y_float = float(robot_pos_xy[1])
    yaw_float = float(robot_yaw)

    node_info = {
        "idx": node_index,
        "pos": (x_float, y_float),
        "yaw": yaw_float,
        "img_path": save_path,
        "visited": False,
    }
    topomap_nodes.append(node_info)
    print(
        f"[TOPO] Node {node_index} => pos=({x_float:.2f}, {y_float:.2f}), yaw={yaw_float:.2f} deg, saved={save_path}"
    )

    if node_index not in adj_list:
        adj_list[node_index] = []

    # connect edges
    for old_node in topomap_nodes:
        if old_node["idx"] == node_index:
            continue
        old_x, old_y = old_node["pos"]
        dist_val = np.linalg.norm(np.array([old_x, old_y]) - np.array([x_float, y_float]))
        if dist_val < EDGE_DISTANCE_THRESHOLD:
            add_edge(node_index, old_node["idx"], dist_val)

    node_index += 1


##############################################################################
# Frontier
##############################################################################


def get_next_frontier(robot_pos_xy: np.ndarray) -> int:
    if len(topomap_nodes) <= 1:
        return -1

    current_idx = get_closest_node_to_pos(robot_pos_xy)
    if current_idx < 0:
        return -1

    dist_map = dijkstra(current_idx)
    best_idx = -1
    best_dist = float("inf")
    for node in topomap_nodes:
        if not node["visited"]:
            idx = node["idx"]
            # skip same node
            if idx == current_idx:
                continue
            cost = dist_map.get(idx, float("inf"))
            if cost < best_dist:
                best_dist = cost
                best_idx = idx
    return best_idx


##############################################################################
def save_topomap_yaml(episode_map_dir: str):
    """
    Convert all numeric data to standard Python float before saving.
    Saves 'nodes_info.yaml' in the directory for this episode.
    """
    global topomap_nodes, adj_list
    node_data_path = os.path.join(episode_map_dir, "nodes_info.yaml")

    # Prepare a serializable data structure
    serializable_nodes = []
    for node in topomap_nodes:
        x, y = node["pos"]
        serializable_node = {
            "idx": int(node["idx"]),
            "pos": (float(x), float(y)),
            "yaw": float(node["yaw"]),
            "img_path": str(node["img_path"]),
            "visited": bool(node.get("visited", False)),
        }
        serializable_nodes.append(serializable_node)

    edges_list = []
    for u, neighbors in adj_list.items():
        for v, dist_val in neighbors:
            edges_list.append((int(u), int(v), float(dist_val)))

    data = {"nodes": serializable_nodes, "edges": edges_list}

    with open(node_data_path, "w") as f:
        yaml.safe_dump(data, f)

    print(f"[TOPO] Wrote {len(topomap_nodes)} nodes and adjacency to {node_data_path}")


##############################################################################
# NoMaD Diffusion
##############################################################################


def sample_diffusion_action(
    model, obs_images, model_params, device, noise_scheduler, args, goal_image=None
):
    """
    Single-frame input. If goal_image is None => random exploration.
    """
    obs_tensor = transform_images(
        obs_images, model_params["image_size"], center_crop=False
    ).to(device)

    if goal_image is None:
        fake_goal = torch.randn((1, 3, *model_params["image_size"])).to(device)
        mask_val = 1  # ignore
        goal_tensor = fake_goal
    else:
        # if we do want a real goal
        goal_tensor = transform_images(
            [goal_image], model_params["image_size"], center_crop=False
        ).to(device)
        mask_val = 0  # use goal

    mask = torch.ones(1).long().to(device) * mask_val

    with torch.no_grad():
        obs_cond = model(
            "vision_encoder",
            obs_img=obs_tensor,
            goal_img=goal_tensor,
            input_goal_mask=mask,
        )
        if obs_cond.ndim == 2:
            obs_cond = obs_cond.repeat(args.num_samples, 1)
        else:
            obs_cond = obs_cond.repeat(args.num_samples, 1, 1)

        noise_scheduler.set_timesteps(model_params["num_diffusion_iters"])
        naction = torch.randn(
            (args.num_samples, model_params["len_traj_pred"], 2), device=device
        )
        for k in noise_scheduler.timesteps:
            noise_pred = model(
                "noise_pred_net", sample=naction, timestep=k, global_cond=obs_cond
            )
            naction = noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=naction
            ).prev_sample

    naction = to_numpy(get_action(naction))
    chosen_traj = naction[0]
    dist_wp = np.linalg.norm(chosen_traj[args.waypoint])
    print(f"[DIFFUSION] Sampled waypoint dist={dist_wp:.3f}, mask_val={mask_val}")
    return chosen_traj[args.waypoint]


##############################################################################
# Main
##############################################################################


def main(headless=False, short_exec=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)
    model_config_path = os.path.join(MODEL_TRAIN_PATH, "config", "nomad.yaml")
    ckpt_path = os.path.join(MODEL_DEPLOY_PATH, "model_weights", "nomad.pth")
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)
    print(f"[INFO] Loaded model config from {model_config_path}")
    model = load_model(ckpt_path, model_params, device)
    model.eval()

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=model_params["num_diffusion_iters"],
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    config_filename = os.path.join(og.example_config_path, "turtlebot_nav.yaml")
    with open(config_filename, "r") as f:
        config = yaml.safe_load(f)
    config["scene"]["load_object_categories"] = ["floors", "walls", "ceilings"]
    if headless:
        config["headless"] = True

    env = og.Environment(configs=config)
    robot_name = env.robots[0].name

    context_queue = []
    context_size = model_params["context_size"]
    max_episodes = 2 if short_exec else 5
    steps_per_episode = 1000

    class ArgObj:
        """Dummy object to hold arguments for sample_diffusion_action."""

        def __init__(self):
            self.num_samples = 8
            self.waypoint = 2

    args = ArgObj()

    for ep_i in range(max_episodes):
        # -----------------------------------------------------------
        # (A) Reset the graph data structures for this episode
        # -----------------------------------------------------------
        global node_index, topomap_nodes, adj_list
        node_index = 0
        topomap_nodes = []
        adj_list = {}

        # Create a per-episode subfolder inside dynamic_map
        episode_map_dir = os.path.join(map_dir, f"episode_{ep_i}")
        os.makedirs(episode_map_dir, exist_ok=True)

        # Reset environment
        env.reset()
        context_queue.clear()
        last_node_pos_2d = None

        print(f"\n[INFO] Starting episode={ep_i} ...")

        for step_i in range(steps_per_episode):
            if step_i == 0:
                zero_action = np.array([0.0, 0.0], dtype=np.float32)
                states, rewards, terminated, truncated, infos = env.step(
                    {robot_name: zero_action}
                )
            else:
                states, rewards, terminated, truncated, infos = env.step(
                    {robot_name: action}
                )

            robot_state = states[robot_name]
            proprio = robot_state["proprio"]
            robot_pos_2d = proprio[:2]
            robot_yaw = proprio[3]

            camera_key = f"{robot_name}:eyes:Camera:0"
            camera_out = robot_state[camera_key]
            rgb_tensor = camera_out["rgb"]
            obs_img = array_to_pil(rgb_tensor)

            # Possibly add a new node
            if last_node_pos_2d is None:
                add_node(obs_img, robot_pos_2d, np.degrees(robot_yaw), episode_map_dir)
                last_node_pos_2d = robot_pos_2d
            else:
                dist = np.linalg.norm(robot_pos_2d - last_node_pos_2d)
                if dist > ADD_NODE_DIST:
                    add_node(
                        obs_img, robot_pos_2d, np.degrees(robot_yaw), episode_map_dir
                    )
                    last_node_pos_2d = robot_pos_2d

            # Update context
            if len(context_queue) < context_size + 1:
                # Fill up if not full
                while len(context_queue) < (context_size + 1):
                    context_queue.append(obs_img)
            else:
                context_queue.pop(0)
                context_queue.append(obs_img)

            # Frontier
            frontier_idx = get_next_frontier(robot_pos_2d)
            if frontier_idx < 0:
                # No frontier => random exploration
                waypoint_dxdy = sample_diffusion_action(
                    model,
                    context_queue,
                    model_params,
                    device,
                    noise_scheduler,
                    args,
                    goal_image=None,
                )
            else:
                # Use frontier's node image as goal
                frontier_node = [n for n in topomap_nodes if n["idx"] == frontier_idx][0]
                goal_img = PILImage.open(frontier_node["img_path"])
                waypoint_dxdy = sample_diffusion_action(
                    model,
                    context_queue,
                    model_params,
                    device,
                    noise_scheduler,
                    args,
                    goal_image=goal_img,
                )

            action = pd_controller(waypoint_dxdy, DT, MAX_V, MAX_W)
            print(
                f"[Episode={ep_i}, Step={step_i}] action={action}, pos={robot_pos_2d}, yaw={robot_yaw:.2f}"
            )

            states, rewards, terminated, truncated, infos = env.step({robot_name: action})
            if terminated or truncated:
                print(
                    f"[INFO] Episode ended (terminated={terminated}, truncated={truncated})."
                )
                break

        print(f"[INFO] Episode {ep_i} finished or max steps reached.")

        # -----------------------------------------------------------
        # (B) Save the graph (nodes & edges) for this episode
        # -----------------------------------------------------------
        save_topomap_yaml(episode_map_dir)

    env.close()
    print("[INFO] Graph-based frontier exploration with fallback NoMaD is done.")
    print(f"[INFO] Per-episode maps saved under: {map_dir}")


if __name__ == "__main__":
    main(headless=False, short_exec=False)
