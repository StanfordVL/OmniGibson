import os
import yaml
import numpy as np
import torch
import omnigibson as og
import shutil
import time
from PIL import Image as PILImage
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# Local imports
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

MAX_V = 0.31
MAX_W = 1.90
DT = 1.0
EPS = 1e-8

# Path to store topomap images
TOPOMAP_IMAGES_DIR = "./topomaps/images"
MAP_NAME = "dynamic_map"
map_dir = os.path.join(TOPOMAP_IMAGES_DIR, MAP_NAME)
if os.path.isdir(map_dir):
    shutil.rmtree(map_dir)
os.makedirs(map_dir, exist_ok=True)

ADD_NODE_DIST = 0.5
node_index = 0
topomap_nodes = []

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
    dx, dy = waypoint[:2]
    v = dx / dt if abs(dx) >= EPS else 0.0
    angle = clip_angle(np.arctan2(dy, dx))
    w = angle / dt
    v = np.clip(v, 0, max_v)
    w = np.clip(w, -max_w, max_w)
    return np.array([v, w], dtype=np.float32)


def array_to_pil(rgb_tensor: torch.Tensor) -> PILImage.Image:
    rgb_array = rgb_tensor[..., :3].cpu().numpy().astype(np.uint8)
    return PILImage.fromarray(rgb_array)


def add_node(obs_img: PILImage.Image, robot_pos_xy: np.ndarray, robot_yaw: float):
    global node_index, topomap_nodes
    filename = f"{node_index}.png"
    save_path = os.path.join(TOPOMAP_IMAGES_DIR, MAP_NAME, filename)
    obs_img.save(save_path)
    node_info = {
        "idx": node_index,
        "pos": (float(robot_pos_xy[0]), float(robot_pos_xy[1])),
        "yaw": float(robot_yaw),
        "img_path": save_path,
    }
    topomap_nodes.append(node_info)
    node_index += 1


def sample_diffusion_action(
    model, obs_images, model_params, device, noise_scheduler: DDPMScheduler, args
):
    obs_tensor = transform_images(
        obs_images, model_params["image_size"], center_crop=False
    ).to(device)
    mask = torch.ones(1).long().to(device)  # Enable exploration mode
    with torch.no_grad():
        obs_cond = model(
            "vision_encoder", obs_img=obs_tensor, goal_img=None, input_goal_mask=mask
        )
        obs_cond = obs_cond.repeat(args.num_samples, 1, 1)
        noisy_action = torch.randn(
            (args.num_samples, model_params["len_traj_pred"], 2), device=device
        )
        noise_scheduler.set_timesteps(model_params["num_diffusion_iters"])
        for k in noise_scheduler.timesteps:
            noise_pred = model(
                "noise_pred_net", sample=noisy_action, timestep=k, global_cond=obs_cond
            )
            noisy_action = noise_scheduler.step(noise_pred, k, noisy_action).prev_sample
    return to_numpy(get_action(noisy_action))[0][args.waypoint]


def save_topomap_yaml():
    node_data_path = os.path.join(TOPOMAP_IMAGES_DIR, MAP_NAME, "nodes_info.yaml")
    with open(node_data_path, "w") as f:
        yaml.safe_dump(topomap_nodes, f)


##############################################################################
# Main OmniGibson Loop
##############################################################################


def main(headless=False, short_exec=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)
    model_config_path = os.path.join(MODEL_TRAIN_PATH, "config", "nomad.yaml")
    ckpt_path = os.path.join(MODEL_DEPLOY_PATH, "model_weights", "nomad.pth")
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)
    model = load_model(ckpt_path, model_params, device)
    model.eval()
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=model_params["num_diffusion_iters"],
        beta_schedule="squaredcos_cap_v2",
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
    max_episodes = 10 if not short_exec else 1
    steps_per_episode = 1500
    args = type("ArgObj", (), {"num_samples": 8, "waypoint": 2})()
    last_node_pos_2d = None

    for ep_i in range(max_episodes):
        env.reset()
        context_queue.clear()
        for step_i in range(steps_per_episode):
            states, _, terminated, truncated, _ = env.step(
                {robot_name: np.array([0.0, 0.0], dtype=np.float32)}
            )
            robot_state = states[robot_name]
            robot_pos_2d = robot_state["proprio"][:2]
            robot_yaw = robot_state["proprio"][3]
            camera_output = robot_state[f"{robot_name}:eyes:Camera:0"]
            obs_img = array_to_pil(camera_output["rgb"])
            if (
                last_node_pos_2d is None
                or np.linalg.norm(robot_pos_2d - last_node_pos_2d) > ADD_NODE_DIST
            ):
                add_node(obs_img, robot_pos_2d, np.degrees(robot_yaw))
                last_node_pos_2d = robot_pos_2d
            context_queue.append(obs_img)
            if len(context_queue) > context_size:
                waypoint_dxdy = sample_diffusion_action(
                    model, context_queue, model_params, device, noise_scheduler, args
                )
                action = pd_controller(waypoint_dxdy, DT, MAX_V, MAX_W)
            else:
                action = np.array([0.0, 0.0], dtype=np.float32)
            if terminated or truncated:
                break
    save_topomap_yaml()
    og.clear()


if __name__ == "__main__":
    main(headless=False, short_exec=False)
