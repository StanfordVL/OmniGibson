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
MAX_V = robot_config.get("max_v", 0.31)  # override if needed
MAX_W = robot_config.get("max_w", 1.90)

DT = 1.0  # 1 second per step in PD controller
EPS = 1e-8

# Path to store topomap images (online creation)
TOPOMAP_IMAGES_DIR = "./topomaps/images"
MAP_NAME = "dynamic_map"

# Clear or create the map directory
map_dir = os.path.join(TOPOMAP_IMAGES_DIR, MAP_NAME)
if os.path.isdir(map_dir):
    print(f"[INFO] Removing old files in {map_dir}")
    shutil.rmtree(map_dir)
os.makedirs(map_dir, exist_ok=True)

# Distance threshold to add a new node & marking visited
ADD_NODE_DIST = 0.5
VISIT_THRESHOLD = 0.1
node_index = 0

# For deciding if two nodes are navigably adjacent:
EDGE_DISTANCE_THRESHOLD = 3.0  # tune this as needed

# topomap_nodes: list of dictionaries for each node
topomap_nodes = []
# edges: list of tuples (nodeA, nodeB) for adjacency
edges = []

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
    PD-like controller turning (dx, dy) or (dx, dy, hx, hy) into robot velocities.
    """
    assert len(waypoint) in [2, 4], "waypoint must be 2D or 4D"
    if len(waypoint) == 2:
        dx, dy = waypoint
        hx, hy = 0.0, 0.0
    else:
        dx, dy, hx, hy = waypoint

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
    """Convert OmniGibson camera output (H, W, 4) to a PIL Image."""
    if rgb_tensor.shape[-1] == 4:
        rgb_tensor = rgb_tensor[..., :3]
    rgb_array = rgb_tensor.cpu().numpy().astype(np.uint8)
    return PILImage.fromarray(rgb_array)


##############################################################################
# Distance-Prediction Helper
##############################################################################
def get_distance_from_model(
    imgA_path: str, imgB_path: str, model, device, model_params
) -> float:
    """
    Uses NoMaD's dist_pred_net to measure distance between two node images.
    Returns a float distance predicted by the model.
    """
    # Load images from disk
    imgA = PILImage.open(imgA_path)
    imgB = PILImage.open(imgB_path)

    # Convert to tensors
    tensorA = transform_images([imgA], model_params["image_size"], center_crop=False).to(
        device
    )
    tensorB = transform_images([imgB], model_params["image_size"], center_crop=False).to(
        device
    )

    # For distance measurement, we typically set mask=0 => "use the goal."
    mask = torch.zeros(1).long().to(device)

    with torch.no_grad():
        obsgoal_cond = model(
            "vision_encoder", obs_img=tensorA, goal_img=tensorB, input_goal_mask=mask
        )
        dists = model("dist_pred_net", obsgoal_cond=obsgoal_cond)

    # dists might be a 1D tensor. Return the scalar.
    return float(dists.item())


def maybe_add_edges_for_new_node(new_node, model, device, model_params):
    """
    Compares 'new_node' to all existing nodes with 'dist_pred_net'.
    If distance < EDGE_DISTANCE_THRESHOLD, add an edge to the global 'edges' list.
    """
    for old_node in topomap_nodes:
        if old_node["idx"] == new_node["idx"]:
            continue  # skip itself
        dist_val = get_distance_from_model(
            new_node["img_path"], old_node["img_path"], model, device, model_params
        )
        if dist_val < EDGE_DISTANCE_THRESHOLD:
            # add edges in both directions (if undirected)
            edges.append((new_node["idx"], old_node["idx"]))
            edges.append((old_node["idx"], new_node["idx"]))
            print(
                f"[EDGES] Adding edge ({new_node['idx']} <-> {old_node['idx']}) dist={dist_val:.2f}"
            )


##############################################################################
# Node & Edge Creation
##############################################################################
def add_node(
    obs_img: PILImage.Image,
    robot_pos_xy: np.ndarray,
    robot_yaw: float,
    model,
    device,
    model_params,
):
    """
    Save the current observation as a node in the topomap, plus the robot's (x,y,yaw).
    Then compare it to existing nodes to add edges if the policy says they're navigable.
    """
    global node_index, topomap_nodes

    filename = f"{node_index}.png"
    save_path = os.path.join(TOPOMAP_IMAGES_DIR, MAP_NAME, filename)
    obs_img.save(save_path)

    node_info = {
        "idx": node_index,
        "pos": (float(robot_pos_xy[0]), float(robot_pos_xy[1])),
        "yaw": float(robot_yaw),
        "img_path": save_path,
        "visited": False,
    }
    topomap_nodes.append(node_info)
    print(
        f"[TOPO] Node {node_index} => pos=({robot_pos_xy[0]:.2f}, {robot_pos_xy[1]:.2f}), yaw={robot_yaw:.2f}, saved={save_path}"
    )

    # AFTER adding the node, let's attempt to add edges
    maybe_add_edges_for_new_node(node_info, model, device, model_params)

    node_index += 1


def mark_visited_if_close(robot_pos_xy: np.ndarray):
    """
    Mark any node as visited if the robot is within VISIT_THRESHOLD of that node,
    except skip node 0 so the script doesn't terminate immediately.
    """
    if len(topomap_nodes) == 1:
        return

    for node in topomap_nodes:
        if node["idx"] == 0:  # skip node 0
            continue

        if not node["visited"]:
            dist = np.linalg.norm(robot_pos_xy - np.array(node["pos"]))
            if dist < VISIT_THRESHOLD:
                node["visited"] = True
                print(f"[VISIT] Marking node {node['idx']} as visited (dist={dist:.2f}).")


def get_closest_unvisited_node(robot_pos_xy: np.ndarray):
    best_node = None
    best_dist = float("inf")
    for node in topomap_nodes:
        if not node["visited"]:
            dist = np.linalg.norm(robot_pos_xy - np.array(node["pos"]))
            if dist < best_dist:
                best_dist = dist
                best_node = node
    return best_node


def save_topomap_yaml():
    """
    Save the topological graph (both nodes and edges) to 'nodes_info.yaml'.
    We can store edges as a list of tuples, or store them under 'edges': ...
    """
    node_data_path = os.path.join(TOPOMAP_IMAGES_DIR, MAP_NAME, "nodes_info.yaml")
    data = {"nodes": topomap_nodes, "edges": edges}
    with open(node_data_path, "w") as f:
        yaml.safe_dump(data, f)
    print(
        f"[TOPO] Wrote {len(topomap_nodes)} nodes and {len(edges)} edges to {node_data_path}"
    )


##############################################################################
# NoMaD (Local Planner)
##############################################################################
def sample_diffusion_action(
    model,
    obs_images,
    model_params,
    device,
    noise_scheduler: DDPMScheduler,
    args,
    goal_image: PILImage.Image = None,
):
    """
    Runs the NoMaD diffusion model to predict (dx, dy).
    If 'goal_image' is provided, we feed that as the real goal (mask=0).
    Otherwise, we do random exploration (mask=1).
    """
    obs_tensor = transform_images(
        obs_images, model_params["image_size"], center_crop=False
    ).to(device)

    if goal_image is None:
        # Use random 'fake goal' => mask=1 => ignoring the goal
        fake_goal = torch.randn((1, 3, *model_params["image_size"])).to(device)
        mask_val = 1
        goal_tensor = fake_goal
    else:
        # Use node's image => mask=0 => incorporate the goal
        goal_tensor = transform_images(
            goal_image, model_params["image_size"], center_crop=False
        ).to(device)
        mask_val = 0

    mask = torch.ones(1).long().to(device) * mask_val

    with torch.no_grad():
        obs_cond = model(
            "vision_encoder",
            obs_img=obs_tensor,
            goal_img=goal_tensor,
            input_goal_mask=mask,
        )

        # Expand for multiple samples
        if obs_cond.ndim == 2:
            obs_cond = obs_cond.repeat(args.num_samples, 1)
        else:
            obs_cond = obs_cond.repeat(args.num_samples, 1, 1)

        # Diffusion steps
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

    naction = to_numpy(get_action(naction))  # shape (num_samples, len_traj_pred, 2)

    chosen_traj = naction[0]
    # other_trajs = naction[1:] # if you want them for debugging

    waypoint = chosen_traj[args.waypoint]
    dist_wp = np.linalg.norm(waypoint)
    print(f"[DIFFUSION] Sampled waypoint dist={dist_wp:.3f}, mask_val={mask_val}")

    if model_params.get("normalize", False):
        pass  # e.g. waypoint *= (MAX_V / RATE)
    return waypoint


##############################################################################
# Main OmniGibson Loop + Frontier-Based (Nearest Frontier) + NoMaD Local Planner
##############################################################################
def main(random_selection=False, headless=False, short_exec=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    # 1) Load NoMaD model / config
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)
    model_config_path = os.path.join(MODEL_TRAIN_PATH, "config", "nomad.yaml")
    ckpt_path = os.path.join(MODEL_DEPLOY_PATH, "model_weights", "nomad.pth")
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)
    print(f"[INFO] Loaded model config from", model_config_path)

    model = load_model(ckpt_path, model_params, device)
    model.eval()

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=model_params["num_diffusion_iters"],
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    # 2) Create OmniGibson environment
    config_filename = os.path.join(og.example_config_path, "turtlebot_nav.yaml")
    with open(config_filename, "r") as f:
        config = yaml.safe_load(f)
    config["scene"]["load_object_categories"] = ["floors", "walls", "ceilings"]
    if headless:
        config["headless"] = True

    env = og.Environment(configs=config)
    robot_name = env.robots[0].name

    # 3) Basic parameters
    context_queue = []
    context_size = model_params["context_size"]
    max_episodes = 1 if not short_exec else 1
    steps_per_episode = 2000

    class ArgObj:
        def __init__(self):
            self.num_samples = 8
            self.waypoint = 2

    args = ArgObj()

    # For tracking new nodes
    global node_index
    global topomap_nodes
    global edges

    last_node_pos_2d = None

    # 4) Main loop
    for ep_i in range(max_episodes):
        env.reset()
        context_queue.clear()
        print(
            f"\n[INFO] Starting episode={ep_i} with frontier-based + NoMaD local planning..."
        )

        for step_i in range(steps_per_episode):
            # Step environment
            if step_i == 0:
                zero_action = np.array([0.0, 0.0], dtype=np.float32)
                states, rewards, terminated, truncated, infos = env.step(
                    {robot_name: zero_action}
                )
            else:
                states, rewards, terminated, truncated, infos = env.step(
                    {robot_name: action}
                )

            # (A) Robot state
            robot_state = states[robot_name]
            proprio = robot_state["proprio"]
            robot_pos_2d = proprio[:2]
            robot_yaw = proprio[3]  # or [5], check environment

            # (B) Convert camera image
            camera_key = f"{robot_name}:eyes:Camera:0"
            camera_output = robot_state[camera_key]
            rgb_tensor = camera_output["rgb"]
            obs_img = array_to_pil(rgb_tensor)

            # (C) Add node if we moved enough
            if last_node_pos_2d is None:
                add_node(
                    obs_img,
                    robot_pos_2d,
                    np.degrees(robot_yaw),
                    model=model,
                    device=device,
                    model_params=model_params,
                )
                last_node_pos_2d = robot_pos_2d
            else:
                dist = np.linalg.norm(robot_pos_2d - last_node_pos_2d)
                if dist > ADD_NODE_DIST:
                    add_node(
                        obs_img,
                        robot_pos_2d,
                        np.degrees(robot_yaw),
                        model=model,
                        device=device,
                        model_params=model_params,
                    )
                    last_node_pos_2d = robot_pos_2d

            # (D) Mark visited if close
            mark_visited_if_close(robot_pos_2d)

            # (E) Check if all frontiers are visited
            unvisited_exists = any(not n["visited"] for n in topomap_nodes)
            if not unvisited_exists:
                print("[FRONTIER] No unvisited nodes left. Stopping exploration.")
                action = np.array([0.0, 0.0])
                break

            # (F) Build context queue
            if len(context_queue) < context_size + 1:
                context_queue.append(obs_img)
            else:
                context_queue.pop(0)
                context_queue.append(obs_img)

            # (G) High-level planning: pick nearest unvisited node
            frontier_node = get_closest_unvisited_node(robot_pos_2d)
            if frontier_node is None:
                print("[FRONTIER] Could not find any unvisited node.")
                action = np.array([0.0, 0.0])
                break

            # Load that node's image from disk
            goal_image = PILImage.open(frontier_node["img_path"])

            # (H) Low-level NoMaD local planning
            if len(context_queue) > context_size:
                waypoint_dxdy = sample_diffusion_action(
                    model=model,
                    obs_images=context_queue,
                    model_params=model_params,
                    device=device,
                    noise_scheduler=noise_scheduler,
                    args=args,
                    goal_image=goal_image,
                )
                action = pd_controller(waypoint_dxdy, DT, MAX_V, MAX_W)
            else:
                action = np.array([0.0, 0.0], dtype=np.float32)

            print(
                f"[Episode={ep_i}, Step={step_i}] action={action}, pos={robot_pos_2d}, yaw={np.degrees(robot_yaw):.2f}"
            )

            # Check termination
            if terminated or truncated:
                print(
                    f"[INFO] Episode {ep_i} ended (terminated={terminated}, truncated={truncated})"
                )
                break

        # End of the episode (max steps or all visited)
        unvisited_exists = any(not n["visited"] for n in topomap_nodes)
        if not unvisited_exists:
            print("[FRONTIER] All nodes visited, finishing episode.")
        else:
            print("[INFO] Reached max steps or environment ended.")

    # Save the final topomap (nodes + edges)
    save_topomap_yaml()
    env.close()
    print("[INFO] Frontier-based exploration + NoMaD local planner finished. Map saved.")
    print(f"Final # of nodes: {len(topomap_nodes)}, # of edges: {len(edges)}")


if __name__ == "__main__":
    main(headless=False, short_exec=False)
