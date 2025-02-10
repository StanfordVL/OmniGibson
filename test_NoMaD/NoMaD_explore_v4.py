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

# Max velocities (adjust to your robot.yaml if needed)
MAX_V = 0.31
MAX_W = 1.90

DT = 1.0
EPS = 1e-8

# Path to store topomap images (online creation)
TOPOMAP_IMAGES_DIR = "./topomaps/images"
MAP_NAME = "dynamic_map"

# Create or clear the directory for this map:
map_dir = os.path.join(TOPOMAP_IMAGES_DIR, MAP_NAME)
if os.path.isdir(map_dir):
    print(f"[INFO] Removing old files in {map_dir}")
    shutil.rmtree(map_dir)
os.makedirs(map_dir, exist_ok=True)

# Parameters for node creation
ADD_NODE_DIST = 0.5  # 0.5m movement triggers new node
node_index = 0

# Our global data structures for the topological map:
topomap_nodes = (
    []
)  # List[ { "idx": int, "pos": (x,y), "yaw": float, "img_path": str}, ... ]
topomap_edges = {}  # Dict[ node_idx -> Set[node_idx] ]
unvisited_frontiers = set()  # Which node indices might lead to unexplored areas?

##############################################################################
# Utility Functions
##############################################################################


def clip_angle(theta: float) -> float:
    """Clip angle to [-pi, pi]."""
    theta = np.mod(theta, 2.0 * np.pi)
    if theta > np.pi:
        theta -= 2.0 * np.pi
    return theta


def pd_controller(
    waypoint: np.ndarray, dt: float, max_v: float, max_w: float
) -> np.ndarray:
    """
    PD-like controller to convert (dx, dy) into (linear_vel, angular_vel).
    waypoint can be (dx, dy) or (dx, dy, hx, hy).
    """
    assert len(waypoint) in [2, 4], "waypoint must be 2D or 4D"
    if len(waypoint) == 2:
        dx, dy = waypoint
        hx, hy = 0.0, 0.0
    else:
        dx, dy, hx, hy = waypoint

    if len(waypoint) == 4 and (abs(dx) < EPS and abs(dy) < EPS):
        # If near-zero displacement and we have heading -> rotate in place
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

    # Clip velocities
    v = np.clip(v, 0, max_v)
    w = np.clip(w, -max_w, max_w)
    return np.array([v, w], dtype=np.float32)


def array_to_pil(rgb_tensor: torch.Tensor) -> PILImage.Image:
    """
    Converts OmniGibson camera output (H, W, 4) to a PIL Image.
    """
    if rgb_tensor.shape[-1] == 4:
        rgb_tensor = rgb_tensor[..., :3]
    rgb_array = rgb_tensor.cpu().numpy().astype(np.uint8)
    return PILImage.fromarray(rgb_array)


def add_node(obs_img: PILImage.Image, robot_pos_xy: np.ndarray, robot_yaw: float):
    """
    Save the current observation as a node in the topomap, along with the robot's position & yaw.
    Also create an edge from the previous node to the new node.
    """
    global node_index, topomap_nodes, topomap_edges, unvisited_frontiers
    filename = f"{node_index}.png"
    save_path = os.path.join(TOPOMAP_IMAGES_DIR, MAP_NAME, filename)

    # Save PIL image
    obs_img.save(save_path)

    node_info = {
        "idx": node_index,
        "pos": (float(robot_pos_xy[0]), float(robot_pos_xy[1])),
        "yaw": float(robot_yaw),
        "img_path": save_path,
    }
    topomap_nodes.append(node_info)

    # Create edge to the previous node (simplest approach)
    if node_index > 0:
        prev_idx = node_index - 1
        topomap_edges.setdefault(prev_idx, set()).add(node_index)
        topomap_edges.setdefault(node_index, set()).add(prev_idx)

    # Mark this node as a "frontier" for now
    unvisited_frontiers.add(node_index)

    print(
        f"[TOPO] Node {node_index} => pos=({robot_pos_xy[0]:.2f}, "
        f"{robot_pos_xy[1]:.2f}), yaw={robot_yaw:.2f} deg, saved={save_path}"
    )
    node_index += 1


def sample_diffusion_action(
    model, obs_images, model_params, device, noise_scheduler: DDPMScheduler, args
):
    """
    Runs the diffusion model in *goal-free* mode (mask=1) to predict (dx, dy) for exploration.
    """
    obs_tensor = transform_images(
        obs_images, model_params["image_size"], center_crop=False
    ).to(device)
    fake_goal = torch.randn((1, 3, *model_params["image_size"])).to(device)
    mask = torch.ones(1).long().to(device)  # "Ignore" the goal (goal-free)
    with torch.no_grad():
        obs_cond = model(
            "vision_encoder", obs_img=obs_tensor, goal_img=fake_goal, input_goal_mask=mask
        )
        if obs_cond.ndim == 2:
            obs_cond = obs_cond.repeat(args.num_samples, 1)
        else:
            obs_cond = obs_cond.repeat(args.num_samples, 1, 1)

        noisy_action = torch.randn(
            (args.num_samples, model_params["len_traj_pred"], 2), device=device
        )
        naction = noisy_action
        noise_scheduler.set_timesteps(model_params["num_diffusion_iters"])
        for k in noise_scheduler.timesteps:
            noise_pred = model(
                "noise_pred_net", sample=naction, timestep=k, global_cond=obs_cond
            )
            naction = noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=naction
            ).prev_sample

    naction = to_numpy(get_action(naction))  # (num_samples, len_traj_pred, 2)
    chosen_traj = naction[0]
    waypoint = chosen_traj[args.waypoint]
    dist_wp = np.linalg.norm(waypoint)
    print(f"[DIFFUSION] Sampled waypoint dist={dist_wp:.3f}, waypoint={waypoint}")
    return waypoint


def save_topomap_yaml():
    """
    Save nodes and edges to a YAML file.
    """
    node_data_path = os.path.join(TOPOMAP_IMAGES_DIR, MAP_NAME, "nodes_info.yaml")
    data = {
        "nodes": topomap_nodes,
        "edges": {str(k): list(v) for k, v in topomap_edges.items()},
    }
    with open(node_data_path, "w") as f:
        yaml.safe_dump(data, f)
    print(f"[TOPO] Wrote {len(topomap_nodes)} nodes to {node_data_path}")


##############################################################################
# Graph + Frontier Utilities
##############################################################################
from collections import deque


def compute_path(start_idx, goal_idx, edges_dict):
    """
    Simple BFS to find a path (list of node indices) from start_idx to goal_idx
    in topomap_edges.
    """
    if start_idx == goal_idx:
        return [start_idx]
    visited = set()
    queue = deque([[start_idx]])
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == goal_idx:
            return path
        if node not in visited:
            visited.add(node)
            neighbors = edges_dict.get(node, [])
            for n in neighbors:
                new_path = list(path)
                new_path.append(n)
                queue.append(new_path)
    return [start_idx]  # fallback if not found


def find_closest_node(robot_pos_2d):
    """
    Returns the index of the node whose (x,y) is closest to robot_pos_2d.
    """
    min_dist = float("inf")
    closest_idx = None
    rx, ry = robot_pos_2d
    for node_data in topomap_nodes:
        nx, ny = node_data["pos"]
        dist = np.hypot(nx - rx, ny - ry)
        if dist < min_dist:
            min_dist = dist
            closest_idx = node_data["idx"]
    return closest_idx


def go_to_node(goal_idx, model, device, noise_scheduler, model_params, env, robot_name):
    """
    Uses *goal-conditioned* diffusion (mask=0) to navigate from current position
    to 'goal_idx' (a node in the topomap).
    """
    # Load the goal node's image
    goal_img_path = topomap_nodes[goal_idx]["img_path"]
    goal_img = PILImage.open(goal_img_path)

    # We will do multiple steps until we are "close enough" to the goal node
    mask = torch.zeros(1).long().to(device)  # 0 => we have a real goal
    max_iters = 200

    for i in range(max_iters):
        # (A) Get current robot state
        states = env.step({robot_name: np.array([0.0, 0.0], dtype=np.float32)})[
            0
        ]  # small no-op step
        robot_state = states[robot_name]
        proprio = robot_state["proprio"]
        robot_pos_2d = proprio[:2]

        # Check if we've arrived
        goal_pos = topomap_nodes[goal_idx]["pos"]
        dist_to_goal = np.linalg.norm(np.array(goal_pos) - robot_pos_2d.numpy())
        if dist_to_goal < 0.3:  # threshold for arrival
            print(f"[GOAL] Reached node={goal_idx}, dist={dist_to_goal:.2f}")
            return

        # (B) Get new observation
        camera_key = f"{robot_name}:eyes:Camera:0"
        camera_output = robot_state[camera_key]
        rgb_tensor = camera_output["rgb"]
        obs_img = array_to_pil(rgb_tensor)

        # (C) Prepare for diffusion
        obs_tensor = transform_images(
            obs_img, model_params["image_size"], center_crop=False
        ).to(device)
        goal_tensor = transform_images(
            goal_img, model_params["image_size"], center_crop=False
        ).to(device)

        # (D) Forward pass through the model
        with torch.no_grad():
            obs_cond = model(
                "vision_encoder",
                obs_img=obs_tensor,
                goal_img=goal_tensor,
                input_goal_mask=mask,
            )

            noisy_action = torch.randn(
                (1, model_params["len_traj_pred"], 2), device=device
            )
            naction = noisy_action
            noise_scheduler.set_timesteps(model_params["num_diffusion_iters"])
            for k in noise_scheduler.timesteps:
                noise_pred = model(
                    "noise_pred_net", sample=naction, timestep=k, global_cond=obs_cond
                )
                naction = noise_scheduler.step(
                    model_output=noise_pred, timestep=k, sample=naction
                ).prev_sample

        # (E) Take the first step from the predicted trajectory
        action_traj = to_numpy(get_action(naction))  # shape (1, len_traj_pred, 2)
        waypoint = action_traj[0][0]  # e.g., the first step
        velocity_cmd = pd_controller(waypoint, DT, MAX_V, MAX_W)

        # (F) Step environment with the chosen velocities
        env.step({robot_name: velocity_cmd})

    print(f"[WARN] Timed out trying to reach node={goal_idx} after {max_iters} steps.")


##############################################################################
# Main OmniGibson Loop + Online Topomap
##############################################################################
def main(random_selection=False, headless=False, short_exec=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    # Load model config
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)
    model_config_path = os.path.join(MODEL_TRAIN_PATH, "config", "nomad.yaml")
    ckpt_path = os.path.join(MODEL_DEPLOY_PATH, "model_weights", "nomad.pth")
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)
        print(f"[INFO] Loaded model config from {model_config_path}")

    # Load model
    model = load_model(ckpt_path, model_params, device)
    model.eval()

    # Diffusion Scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=model_params["num_diffusion_iters"],
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    # Create OmniGibson Env
    config_filename = os.path.join(og.example_config_path, "turtlebot_nav.yaml")
    with open(config_filename, "r") as f:
        config = yaml.safe_load(f)
    config["scene"]["load_object_categories"] = ["floors", "walls", "ceilings"]
    if headless:
        config["headless"] = True

    env = og.Environment(configs=config)
    robot_name = env.robots[0].name

    # Basic parameters
    context_queue = []
    context_size = model_params["context_size"]
    max_episodes = 10 if not short_exec else 1

    # Steps for local "goal-free" exploration
    # (you can tweak these)
    local_exploration_steps = 200

    # Arg-like container
    class ArgObj:
        def __init__(self):
            self.num_samples = 8
            self.waypoint = 2

    args = ArgObj()

    # Track the last node position so we know when to add a new node
    last_node_pos_2d = None

    for ep_i in range(max_episodes):
        env.reset()
        context_queue.clear()
        print(f"\n[INFO] Starting episode={ep_i} ...")

        ############################################################################
        # 1) Local exploration (unconditional diffusion) for X steps
        ############################################################################
        action = np.array([0.0, 0.0], dtype=np.float32)
        for step_i in range(local_exploration_steps):
            # Step environment
            states, rewards, terminated, truncated, infos = env.step({robot_name: action})

            robot_state = states[robot_name]
            proprio = robot_state["proprio"]
            robot_pos_2d = proprio[:2]
            robot_yaw = proprio[3]  # or 5 if your array is different

            # Get camera data
            camera_key = f"{robot_name}:eyes:Camera:0"
            camera_output = robot_state[camera_key]
            rgb_tensor = camera_output["rgb"]
            obs_img = array_to_pil(rgb_tensor)

            # (1) Add node if moved enough
            if last_node_pos_2d is None:
                add_node(obs_img, robot_pos_2d, np.degrees(robot_yaw))
                last_node_pos_2d = robot_pos_2d
            else:
                dist = np.linalg.norm(robot_pos_2d - last_node_pos_2d)
                if dist > ADD_NODE_DIST:
                    add_node(obs_img, robot_pos_2d, np.degrees(robot_yaw))
                    last_node_pos_2d = robot_pos_2d

            # (2) Fill the context queue
            if len(context_queue) < context_size + 1:
                context_queue.append(obs_img)
            else:
                context_queue.pop(0)
                context_queue.append(obs_img)

            # (3) Sample from NoMaD diffusion if context is ready
            if len(context_queue) > context_size:
                waypoint_dxdy = sample_diffusion_action(
                    model=model,
                    obs_images=context_queue,
                    model_params=model_params,
                    device=device,
                    noise_scheduler=noise_scheduler,
                    args=args,
                )
                action = pd_controller(waypoint_dxdy, DT, MAX_V, MAX_W)
            else:
                action = np.array([0.0, 0.0], dtype=np.float32)

            # Print debug info
            print(
                f"[Episode={ep_i}, Step={step_i}] action={action}, pos={robot_pos_2d}, yaw={np.degrees(robot_yaw):.2f}"
            )

            if terminated or truncated:
                print(
                    f"[INFO] Episode {ep_i} ended early (terminated={terminated}, truncated={truncated})"
                )
                break

        ############################################################################
        # 2) If we have any frontier nodes, pick one and navigate to it
        ############################################################################
        if unvisited_frontiers:
            # Find your current node
            # (Take a small no-op step to get the latest state.)
            states = env.step({robot_name: np.array([0.0, 0.0], dtype=np.float32)})[0]
            robot_state = states[robot_name]
            robot_pos_2d = robot_state["proprio"][:2]
            current_node = find_closest_node(robot_pos_2d)
            print(f"[FRONTIER] Current node = {current_node}")

            # Pick a frontier node from the set (e.g. pop or pick nearest, etc.)
            frontier_idx = unvisited_frontiers.pop()
            print(f"[FRONTIER] Chosen frontier node = {frontier_idx}")

            # Compute BFS path in the graph
            path_nodes = compute_path(current_node, frontier_idx, topomap_edges)
            print(f"[FRONTIER] BFS path: {path_nodes}")

            # Traverse each node in that path (goal-conditioned)
            # (Skipping the first index, which is current_node)
            for next_node in path_nodes[1:]:
                go_to_node(
                    goal_idx=next_node,
                    model=model,
                    device=device,
                    noise_scheduler=noise_scheduler,
                    model_params=model_params,
                    env=env,
                    robot_name=robot_name,
                )
            print(
                f"[FRONTIER] Arrived at frontier node={frontier_idx}. Resuming local exploration..."
            )

        # End of one "episode" iteration
        print(
            f"[INFO] Finished local exploration + frontier navigation for episode {ep_i}."
        )

    # Once done, save the topomap
    save_topomap_yaml()
    og.clear()
    print("[INFO] Finished simulation and saved topomap.")


if __name__ == "__main__":
    main(headless=False, short_exec=False)
