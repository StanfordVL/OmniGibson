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

# We assume 1 second per step in PD controller logic unless environment is matched
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

# Simple parameters for node creation
ADD_NODE_DIST = 0.5  # 0.5m movement triggers new node
node_index = 0
topomap_nodes = []  # [{"idx": int, "pos": (x,y), "yaw": float, "img_path": str}, ...]

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
    PD-like controller for the robot, matching your pd_controller.py logic.
    waypoint: (dx, dy) or (dx, dy, hx, hy)
    """
    assert len(waypoint) in [2, 4], "waypoint must be 2D or 4D"
    if len(waypoint) == 2:
        dx, dy = waypoint
        hx, hy = 0.0, 0.0
    else:
        dx, dy, hx, hy = waypoint

    # If near-zero displacement and we have heading -> rotate in place
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
    # Clip velocities
    v = np.clip(v, 0, max_v)
    w = np.clip(w, -max_w, max_w)
    return np.array([v, w], dtype=np.float32)


def array_to_pil(rgb_tensor: torch.Tensor) -> PILImage.Image:
    """
    Converts OmniGibson camera output (H, W, 4) to a PIL Image.
    """
    # If alpha channel is present, drop it
    if rgb_tensor.shape[-1] == 4:
        rgb_tensor = rgb_tensor[..., :3]
    rgb_array = rgb_tensor.cpu().numpy().astype(np.uint8)
    return PILImage.fromarray(rgb_array)


def add_node(obs_img: PILImage.Image, robot_pos_xy: np.ndarray, robot_yaw: float):
    """
    Save the current observation as a node in the topomap,
    along with the robot's position + yaw.
    """
    global node_index, topomap_nodes
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
    print(
        f"[TOPO] Node {node_index} => pos=({robot_pos_xy[0]:.2f}, "
        f"{robot_pos_xy[1]:.2f}), yaw={robot_yaw:.2f} deg, saved={save_path}"
    )
    node_index += 1


def sample_diffusion_action(
    model, obs_images, model_params, device, noise_scheduler: DDPMScheduler, args
):
    """
    Runs the diffusion model to predict (dx, dy) for exploration.
    """
    obs_tensor = transform_images(
        obs_images, model_params["image_size"], center_crop=False
    ).to(device)
    fake_goal = torch.randn((1, 3, *model_params["image_size"])).to(device)
    mask = torch.ones(1).long().to(device)  # "Ignore" the goal
    with torch.no_grad():
        obs_cond = model(
            "vision_encoder", obs_img=obs_tensor, goal_img=fake_goal, input_goal_mask=mask
        )
        # Expand for multiple samples
        if obs_cond.ndim == 2:
            obs_cond = obs_cond.repeat(args.num_samples, 1)
        else:
            obs_cond = obs_cond.repeat(args.num_samples, 1, 1)

        # Diffusion
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
    print(f"[DIFFUSION] Sampled waypoint dist={dist_wp:.3f}")

    # If normalized in training, consider scaling
    if model_params.get("normalize", False):
        pass
        # e.g. waypoint *= (MAX_V / RATE)

    print(f"[DIFFUSION] Sampled waypoint: {waypoint}")

    return waypoint


def save_topomap_yaml():
    """
    After you've finished collecting nodes, save them to a nodes_info.yaml
    so the navigation script can load them later.
    """
    node_data_path = os.path.join(TOPOMAP_IMAGES_DIR, MAP_NAME, "nodes_info.yaml")
    with open(node_data_path, "w") as f:
        yaml.safe_dump(topomap_nodes, f)
    print(f"[TOPO] Wrote {len(topomap_nodes)} nodes to {node_data_path}")


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
    max_episodes = 1 if not short_exec else 1
    steps_per_episode = 2000  # reduce if debugging

    # Arg-like container
    class ArgObj:
        def __init__(self):
            self.num_samples = 8
            self.waypoint = 2

    args = ArgObj()

    # For storing last node position
    last_node_pos_2d = None

    for ep_i in range(max_episodes):
        env.reset()
        context_queue.clear()
        print(f"\n[INFO] Starting episode={ep_i} ...")

        for step_i in range(steps_per_episode):
            # Step environment
            if step_i == 0:
                # no motion on first step
                states, rewards, terminated, truncated, infos = env.step(
                    {robot_name: np.array([0.0, 0.0], dtype=np.float32)}
                )
            else:
                states, rewards, terminated, truncated, infos = env.step(
                    {robot_name: action}
                )

            # Get robot state
            robot_state = states[robot_name]
            # e.g. "proprio": [x, y, z, roll, pitch, yaw]
            proprio = robot_state["proprio"]
            robot_pos_2d = proprio[:2]  # (x, y)
            # robot_yaw = proprio[5] if len(proprio) >= 6 else proprio[3]
            robot_yaw = proprio[3]

            # Get camera data
            camera_key = f"{robot_name}:eyes:Camera:0"
            camera_output = robot_state[camera_key]
            rgb_tensor = camera_output["rgb"]
            obs_img = array_to_pil(rgb_tensor)

            # (1) Online Topomap: add node if we moved enough
            if last_node_pos_2d is None:  # First node
                add_node(obs_img, robot_pos_2d, np.degrees(robot_yaw))
                last_node_pos_2d = robot_pos_2d
            else:
                dist = np.linalg.norm(robot_pos_2d - last_node_pos_2d)
                if dist > ADD_NODE_DIST:  # Add node if moved enough since last node
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
                # PD controller to turn (dx, dy) into velocity
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

    # Once done, save the topomap
    save_topomap_yaml()

    og.clear()
    print("[INFO] Finished simulation and saved topomap.")


if __name__ == "__main__":
    # Example usage:
    #   python create_topomap_sim.py  (runs with default parameters)
    main(headless=False, short_exec=False)
