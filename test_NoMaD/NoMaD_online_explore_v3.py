import os
import yaml
import numpy as np
import torch
import omnigibson as og
import shutil
from PIL import Image as PILImage
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# Local imports (adjust paths as needed)
from deployment.src.utils import to_numpy, transform_images, load_model
from train.vint_train.training.train_utils import get_action


##############################################################################
# Global / Config
##############################################################################
work_dir = os.getcwd()
MODEL_DEPLOY_PATH = os.path.join(work_dir, "test_NoMaD", "deployment")
MODEL_TRAIN_PATH = os.path.join(work_dir, "test_NoMaD", "train")
MODEL_CONFIG_PATH = os.path.join(MODEL_DEPLOY_PATH, "config", "models.yaml")

# Example robot constraints
MAX_V = 0.31
MAX_W = 1.90
DT = 1.0
EPS = 1e-8

# Path to store topomap images
TOPOMAP_IMAGES_DIR = "./topomaps/images"
MAP_NAME = "dynamic_map"
map_dir = os.path.join(TOPOMAP_IMAGES_DIR, MAP_NAME)

# Remove old dir
if os.path.isdir(map_dir):
    print(f"[INFO] Removing old files in {map_dir}")
    shutil.rmtree(map_dir)
os.makedirs(map_dir, exist_ok=True)

# Thresholds for topological graph
ADD_NODE_DIST = 0.5  # distance from last node to create a new node
EDGE_DISTANCE_THRESHOLD = 3.0  # distance threshold for quick adjacency
FRONTIER_REACHED_THRESHOLD = 0.5

##############################################################################
# Globals (reset each episode)
##############################################################################
node_index = 0
topomap_nodes = []
adj_list = {}


##############################################################################
# Utility Functions
##############################################################################
def clip_angle(theta: float) -> float:
    """Normalize angle to (-pi, pi)."""
    theta = np.mod(theta, 2.0 * np.pi)
    if theta > np.pi:
        theta -= 2.0 * np.pi
    return theta


def pd_controller(
    waypoint: np.ndarray, dt: float, max_v: float, max_w: float
) -> np.ndarray:
    """
    Simple PD-like approach: waypoint = (dx, dy).
    We attempt to go dx/dt in linear velocity, angle in w.
    """
    dx, dy = waypoint[:2]
    dist = np.sqrt(dx**2 + dy**2)
    if dist < EPS:
        # Turn in place if dy != 0
        v = 0.0
        w = np.sign(dy) * (np.pi / (2 * dt))
    else:
        v = dist / dt
        angle = clip_angle(np.arctan2(dy, dx))
        w = angle / dt

    v = np.clip(v, 0, max_v)
    w = np.clip(w, -max_w, max_w)
    return np.array([v, w], dtype=np.float32)


def array_to_pil(rgb_tensor: torch.Tensor) -> PILImage.Image:
    """Convert a torch Tensor to a PIL image."""
    if rgb_tensor.shape[-1] == 4:
        rgb_tensor = rgb_tensor[..., :3]
    arr = rgb_tensor.cpu().numpy().astype(np.uint8)
    return PILImage.fromarray(arr)


##############################################################################
# Graph Helpers
##############################################################################
def add_edge(u: int, v: int, dist_val: float):
    """Create a bidirectional edge in adjacency list."""
    global adj_list
    dist_python = float(dist_val)
    if u not in adj_list:
        adj_list[u] = []
    if v not in adj_list:
        adj_list[v] = []
    adj_list[u].append((v, dist_python))
    adj_list[v].append((u, dist_python))


def dijkstra(start_idx: int) -> dict:
    """Return distance map from start_idx to all other nodes using Dijkstra."""
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
    """Return index of node closest to xy."""
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
# Node & Edge Creation (online topological graph)
##############################################################################
def add_node(
    obs_img: PILImage.Image,
    global_pos_xy: np.ndarray,
    global_yaw_deg: float,
    episode_map_dir: str,
    unexplored=False,
):
    """
    Create a node representing the robot's (global) observation:
      - global_pos_xy = (x, y) in global coords
      - global_yaw_deg is a global orientation in degrees
      - unexplored=True means we created it artificially for high-level planning
        and haven't physically visited it yet
    Then link this new node with existing nodes if they are within
    EDGE_DISTANCE_THRESHOLD, or some model-based check.
    """
    global node_index, topomap_nodes, adj_list

    filename = f"{node_index}.png"
    save_path = os.path.join(episode_map_dir, filename)

    # If we physically have an image (like the robot camera), save it
    # If we're artificially creating an "unexplored" node, we might just
    # store a placeholder image or skip saving. Up to you.
    if obs_img is not None:
        obs_img.save(save_path)
    else:
        # You can put a blank or symbolic image
        PILImage.new("RGB", (64, 64), (128, 128, 128)).save(save_path)

    x_float = float(global_pos_xy[0])
    y_float = float(global_pos_xy[1])
    yaw_float = float(global_yaw_deg)

    node_info = {
        "idx": node_index,
        "pos": (x_float, y_float),
        "yaw": yaw_float,  # global yaw in degrees
        "img_path": save_path,
        # "visited": False means we haven't physically been near that node
        # If unexplored=True => It's a "hypothetical" node we want to explore
        "visited": False if unexplored else True,
    }
    topomap_nodes.append(node_info)

    print(
        f"[TOPO] Created node {node_index} => pos=({x_float:.2f}, {y_float:.2f}), yaw={yaw_float:.2f} deg, unexplored={unexplored}"
    )

    if node_index not in adj_list:
        adj_list[node_index] = []

    # Quick adjacency
    for old_node in topomap_nodes:
        if old_node["idx"] == node_index:
            continue
        old_x, old_y = old_node["pos"]
        dist_val = np.linalg.norm([old_x - x_float, old_y - y_float])
        if dist_val < EDGE_DISTANCE_THRESHOLD:
            add_edge(node_index, old_node["idx"], dist_val)

    node_index += 1


##############################################################################
# High-Level Exploration Planner
##############################################################################
def create_random_exploration_nodes(num_nodes, episode_map_dir, xbounds, ybounds):
    """
    Example of adding "unexplored" nodes at random positions in the environment.
    Suppose the environment is from x in xbounds, y in ybounds.
    We'll create placeholders with no real images.
    """
    for _ in range(num_nodes):
        rx = np.random.uniform(*xbounds)
        ry = np.random.uniform(*ybounds)
        # We have no real yaw or image for these. We might guess yaw=0 or random
        yaw_deg = 0.0
        # Mark them as unexplored so the robot can pick them
        add_node(None, (rx, ry), yaw_deg, episode_map_dir, unexplored=True)


def pick_exploration_goal(robot_pos_xy: np.ndarray) -> int:
    """
    Return an "unexplored" node that is closest by path distance from
    the robot's current position. If none exist, return -1.
    If found, we set "visited=False" until we get near it.
    """
    if len(topomap_nodes) < 1:
        return -1

    current_idx = get_closest_node_to_pos(robot_pos_xy)
    if current_idx < 0:
        return -1

    dist_map = dijkstra(current_idx)
    best_idx = -1
    best_dist = float("inf")
    for node in topomap_nodes:
        # pick nodes that are visited=False (which might be physically not visited)
        if node["visited"] == False:
            idx = node["idx"]
            if idx == current_idx:
                continue
            cost = dist_map.get(idx, float("inf"))
            if cost < best_dist:
                best_idx = idx
                best_dist = cost
    return best_idx


##############################################################################
def save_topomap_yaml(episode_map_dir: str):
    """Save the topological graph to a yaml file."""
    global topomap_nodes, adj_list
    node_data_path = os.path.join(episode_map_dir, "nodes_info.yaml")

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
    print(f"[TOPO] Wrote {len(topomap_nodes)} nodes => {node_data_path}")


##############################################################################
# Diffusion-based Action (Local Planner)
##############################################################################
def sample_diffusion_action(
    model, obs_images, model_params, device, noise_scheduler, args, goal_image=None
):
    """
    If goal_image is None => random exploration from NoMaD.
    Otherwise => condition on the provided goal image.
    """
    obs_tensor = transform_images(
        obs_images, model_params["image_size"], center_crop=False
    ).to(device)

    if goal_image is None:
        fake_goal = torch.randn((1, 3, *model_params["image_size"])).to(device)
        mask_val = 1  # ignore goal
        goal_tensor = fake_goal
    else:
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
            step_result = noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=naction
            )
            naction = step_result.prev_sample

    naction = to_numpy(get_action(naction))
    chosen_traj = naction[0]
    dist_wp = np.linalg.norm(chosen_traj[args.waypoint])
    print(f"[DIFFUSION] Sampled local waypoint dist={dist_wp:.3f}, mask_val={mask_val}")
    return chosen_traj[args.waypoint]


##############################################################################
# Main
##############################################################################
def main(headless=False, short_exec=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load model config & checkpoint
    model_config_path = os.path.join(MODEL_TRAIN_PATH, "config", "nomad.yaml")
    ckpt_path = os.path.join(MODEL_DEPLOY_PATH, "model_weights", "nomad.pth")
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)
    print(f"[INFO] Loaded model config: {model_config_path}")
    model = load_model(ckpt_path, model_params, device)
    model.eval()

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=model_params["num_diffusion_iters"],
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    # Create OmniGibson environment
    config_filename = os.path.join(og.example_config_path, "turtlebot_nav.yaml")
    with open(config_filename, "r") as f:
        config = yaml.safe_load(f)
    config["scene"]["load_object_categories"] = ["floors", "walls", "ceilings"]
    if headless:
        config["headless"] = True

    env = og.Environment(configs=config)
    robot_name = env.robots[0].name

    context_size = model_params["context_size"]

    # We'll run a few episodes
    max_episodes = 1 if short_exec else 2
    steps_per_episode = 300

    class ArgObj:
        """For sample_diffusion_action."""

        def __init__(self):
            self.num_samples = 8
            self.waypoint = 2

    args = ArgObj()

    global node_index, topomap_nodes, adj_list

    for ep_i in range(max_episodes):
        # Reset global graph each episode
        node_index = 0
        topomap_nodes = []
        adj_list = {}

        episode_map_dir = os.path.join(map_dir, f"episode_{ep_i}")
        os.makedirs(episode_map_dir, exist_ok=True)

        env.reset()

        # Create a few random exploration nodes in a known bounding region
        # Adjust xbounds, ybounds to your environment's scale or map
        xbounds = (-2.0, 2.0)
        ybounds = (-2.0, 2.0)
        create_random_exploration_nodes(
            num_nodes=3, episode_map_dir=episode_map_dir, xbounds=xbounds, ybounds=ybounds
        )

        context_queue = []
        last_node_pos_2d = None

        # Current high-level exploration target node
        current_exploration_node_idx = None

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
            if robot_state is None:
                print("[ERROR] No robot state found. Possibly environment ended.")
                break

            proprio = robot_state["proprio"]
            # This is presumably the robot's global position + yaw in OmniGibson's coordinate frame
            robot_pos_2d = proprio[:2]  # x, y
            robot_yaw = proprio[3]  # global yaw in radians

            camera_key = f"{robot_name}:eyes:Camera:0"
            camera_out = robot_state[camera_key]
            rgb_tensor = camera_out["rgb"]
            obs_img = array_to_pil(rgb_tensor)

            # Possibly add a new "visited" node if we traveled far enough
            if last_node_pos_2d is None:
                add_node(
                    obs_img,
                    robot_pos_2d,
                    np.degrees(robot_yaw),
                    episode_map_dir,
                    unexplored=False,
                )
                last_node_pos_2d = robot_pos_2d
            else:
                dist_traveled = np.linalg.norm(robot_pos_2d - last_node_pos_2d)
                if dist_traveled > ADD_NODE_DIST:
                    add_node(
                        obs_img,
                        robot_pos_2d,
                        np.degrees(robot_yaw),
                        episode_map_dir,
                        unexplored=False,
                    )
                    last_node_pos_2d = robot_pos_2d

            # Maintain context queue for diffusion
            if len(context_queue) < context_size + 1:
                while len(context_queue) < context_size + 1:
                    context_queue.append(obs_img)
            else:
                context_queue.pop(0)
                context_queue.append(obs_img)

            # Check if we have a current exploration node, and if we reached it
            if current_exploration_node_idx is not None:
                node_pos = np.array(topomap_nodes[current_exploration_node_idx]["pos"])
                dist_to_node = np.linalg.norm(robot_pos_2d - node_pos)
                if dist_to_node < FRONTIER_REACHED_THRESHOLD:
                    print(
                        f"[DEBUG] Reached exploration node {current_exploration_node_idx}!"
                    )
                    topomap_nodes[current_exploration_node_idx]["visited"] = True
                    current_exploration_node_idx = None

            # If we don't have an exploration node, pick one
            if current_exploration_node_idx is None:
                next_goal_idx = pick_exploration_goal(robot_pos_2d)
                if next_goal_idx < 0:
                    # No unexplored => random
                    print("[DEBUG] No unexplored node => random.")
                    local_dxdy = sample_diffusion_action(
                        model,
                        context_queue,
                        model_params,
                        device,
                        noise_scheduler,
                        args,
                        goal_image=None,
                    )
                else:
                    current_exploration_node_idx = next_goal_idx
                    print(
                        f"[DEBUG] Chosen exploration node {current_exploration_node_idx}"
                    )
                    # We could pass the node's image as a goal image, or not. We'll do a simple local approach:
                    local_dxdy = sample_diffusion_action(
                        model,
                        context_queue,
                        model_params,
                        device,
                        noise_scheduler,
                        args,
                        goal_image=None,
                    )
            else:
                # We have a node in mind => local planning
                node_pos = np.array(topomap_nodes[current_exploration_node_idx]["pos"])
                dx = node_pos[0] - robot_pos_2d[0]
                dy = node_pos[1] - robot_pos_2d[1]

                # combine or weigh the diffusion step for local obstacle avoidance
                local_dxdy = sample_diffusion_action(
                    model,
                    context_queue,
                    model_params,
                    device,
                    noise_scheduler,
                    args,
                    goal_image=None,
                )

                # Weighted combination: 0.7*(goal offset) + 0.3*(diffusion suggestion)
                waypoint_dx = 0.7 * dx + 0.3 * local_dxdy[0]
                waypoint_dy = 0.7 * dy + 0.3 * local_dxdy[1]
                local_dxdy = np.array([waypoint_dx, waypoint_dy], dtype=np.float32)

            action = pd_controller(local_dxdy, DT, MAX_V, MAX_W)
            print(
                f"[Episode={ep_i}, Step={step_i}] action={action}, pos={robot_pos_2d}, yaw={robot_yaw:.2f} rad"
            )

            states, rewards, terminated, truncated, infos = env.step({robot_name: action})
            if terminated or truncated:
                print(
                    f"[INFO] Episode ended. terminated={terminated}, truncated={truncated}"
                )
                break

        save_topomap_yaml(episode_map_dir)

    env.close()
    print("[INFO] Global topological graph with position-based exploration done.")
    print(f"[INFO] Per-episode graphs saved in: {map_dir}")


if __name__ == "__main__":
    main(headless=False, short_exec=False)
