import os
import yaml
import numpy as np
import torch
import omnigibson as og
from PIL import Image as PILImage

# Local imports (change paths if necessary)
from deployment.src.utils import to_numpy, transform_images, load_model
from train.vint_train.training.train_utils import get_action

# Diffusion scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


##############################################################################
# Paths and Global Config
##############################################################################
work_dir = os.getcwd()
MODEL_DEPLOY_PATH = os.path.join(work_dir, "test_NoMaD", "deployment")
MODEL_TRAIN_PATH = os.path.join(work_dir, "test_NoMaD", "train")
MODEL_CONFIG_PATH = os.path.join(MODEL_DEPLOY_PATH, "config", "models.yaml")
ROBOT_CONFIG_PATH = os.path.join(MODEL_DEPLOY_PATH, "config", "robot.yaml")

# Robot parameters
with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
RATE = 50  # 1  # robot_config["frame_rate"]

# ---------------------------------------------------------------------------
# NOTE: Hard-coded max velocities here. If your `robot.yaml` has different
# fields, you can read them directly:
#
# MAX_V = robot_config["max_v"]
# MAX_W = robot_config["max_w"]
# ---------------------------------------------------------------------------
MAX_V = 0.4  # 0.31  # 0.31 m/s = 1.12 km/h
MAX_W = 0.2  # 1.90  # 1.90 rad/s = 108.8 deg/s

# Timestep for stepping the robot.
# If your environment or robot_config prescribes a certain step time, use it:
DT = 1.0  # / RATE
EPS = 1e-8


##############################################################################
# Clip angle to [-pi, pi]
##############################################################################
def clip_angle(theta: float) -> float:
    """Clip angle to [-pi, pi]."""
    theta = np.mod(theta, 2.0 * np.pi)
    if theta > np.pi:
        theta -= 2.0 * np.pi
    return theta


##############################################################################
# PD Controller (adapted from pd_controller.py)
##############################################################################
def pd_controller(
    waypoint: np.ndarray, dt: float, max_v: float, max_w: float
) -> np.ndarray:
    """
    PD-like controller for the robot, matching your pd_controller.py logic.

    Args:
        waypoint (np.ndarray): (2,) -> (dx, dy) or (4,) -> (dx, dy, hx, hy)
        dt (float): time step
        max_v (float): maximum linear velocity
        max_w (float): maximum angular velocity

    Returns:
        np.ndarray(2,): [v, w]
    """
    assert len(waypoint) in [
        2,
        4,
    ], "waypoint must be 2D or 4D (dx, dy) or (dx, dy, hx, hy)"

    if len(waypoint) == 2:
        dx, dy = waypoint
        hx, hy = 0.0, 0.0  # not used
    else:
        dx, dy, hx, hy = waypoint

    # If near-zero displacement and we have heading -> rotate in place
    if len(waypoint) == 4 and (abs(dx) < EPS and abs(dy) < EPS):
        v = 0.0
        # If hx, hy indicates heading, rotate by angle over dt
        heading_angle = np.arctan2(hy, hx)
        heading_angle = clip_angle(heading_angle)
        w = heading_angle / dt
    else:
        # If dx is extremely small, treat as pure rotation
        if abs(dx) < EPS:
            v = 0.0
            # 90-degree turn if dy>0 or -90 if dy<0, scaled by dt
            w = np.sign(dy) * (np.pi / (2 * dt))
        else:
            # Normal PD logic: v = dx/dt, w = arctan(dy/dx)/dt
            v = dx / dt
            angle = np.arctan2(dy, dx)  # direction
            angle = clip_angle(angle)
            w = angle / dt

    # Clip velocities
    v = np.clip(v, 0, max_v)  # only forward motion from 0..max_v
    w = np.clip(w, -max_w, max_w)

    return np.array([v, w], dtype=np.float32)


##############################################################################
# Convert an OmniGibson RGB tensor to PIL Image
##############################################################################
def array_to_pil(rgb_tensor: torch.Tensor) -> PILImage.Image:
    """
    Converts the OmniGibson camera output (H, W, 3 or 4) to a PIL Image.
    Expects float32 in [0,1] or a similar format;
    casts to uint8 in [0,255] for PIL.

    Args:
        rgb_tensor (torch.Tensor): shape (H, W, 3) or (H, W, 4).

    Returns:
        PILImage.Image
    """
    if rgb_tensor.shape[-1] == 4:
        rgb_tensor = rgb_tensor[..., :3]  # Drop alpha if present
    rgb_array = rgb_tensor.cpu().numpy().astype(np.uint8)
    return PILImage.fromarray(rgb_array)


# ##############################################################################
# # Convert (dx, dy) -> (v, w)
# ##############################################################################
# def waypoint_to_velocity(waypoint: np.ndarray, max_v: float, max_w: float, dt: float) -> np.ndarray:
#     """
#     Convert a waypoint displacement (dx, dy) over time dt to (v, w).
#     Assumes differential-drive style:
#         v = distance / dt, w = angle / dt
#     where angle = arctan2(dy, dx).

#     Args:
#         waypoint (np.ndarray): shape (2,) -> (dx, dy)
#         max_v (float): maximum linear velocity
#         max_w (float): maximum angular velocity
#         dt (float): time per step in seconds

#     Returns:
#         np.ndarray of shape (2,) containing [v, w].
#     """
#     dx, dy = waypoint
#     EPS = 1e-8

#     if (abs(dx) < EPS) and (abs(dy) < EPS):
#         v = 0.0
#         w = 0.0
#     else:
#         # If dx,dy is the displacement over dt:
#         distance = np.sqrt(dx**2 + dy**2)
#         angle = np.arctan2(dy, dx)
#         v = distance / dt
#         w = angle / dt

#     # Optionally clamp if you want to enforce max velocity
#     # v = np.clip(v, -max_v, max_v)
#     # w = np.clip(w, -max_w, max_w)

#     return np.array([v, w], dtype=np.float32)


##############################################################################
# Diffusion-based action sampling
##############################################################################
def sample_diffusion_action(
    model,
    obs_images,  # (context_size + 1) PIL images
    model_params,
    device,
    noise_scheduler: DDPMScheduler,
    args,
):
    """
    Runs the diffusion model to predict a displacement (dx, dy)
    given the context images.

    Args:
        model (torch.nn.Module): The loaded diffusion model
        obs_images (list[PILImage]): A sequence of PIL images of length context_size+1
        model_params (dict): hyperparams from model config (image_size, len_traj_pred, etc.)
        device (torch.device): CPU or GPU
        noise_scheduler (DDPMScheduler): The diffusion noise scheduler
        args: containing fields (num_samples, waypoint, etc.)

    Returns:
        np.ndarray of shape (2,) -> (dx, dy)
    """
    # (1) Stack & transform images for model
    obs_tensor = transform_images(
        obs_images, model_params["image_size"], center_crop=False
    ).to(device)

    # (2) We generate a random goal if your model expects a goal
    fake_goal = torch.randn((1, 3, *model_params["image_size"])).to(device)
    mask = torch.ones(1).long().to(device)  # e.g., ignoring the goal

    # (3) Encode observation
    with torch.no_grad():
        obs_cond = model(
            "vision_encoder",
            obs_img=obs_tensor,
            goal_img=fake_goal,
            input_goal_mask=mask,
        )

        # Expand batch if we want multiple samples
        if obs_cond.ndim == 2:
            obs_cond = obs_cond.repeat(args.num_samples, 1)
        else:
            obs_cond = obs_cond.repeat(args.num_samples, 1, 1)

        # Initialize noisy action
        noisy_action = torch.randn(
            (args.num_samples, model_params["len_traj_pred"], 2), device=device
        )
        naction = noisy_action

        # Diffusion denoising
        noise_scheduler.set_timesteps(model_params["num_diffusion_iters"])
        for k in noise_scheduler.timesteps:
            noise_pred = model(
                "noise_pred_net", sample=naction, timestep=k, global_cond=obs_cond
            )
            naction = noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=naction
            ).prev_sample

    # Convert to numpy
    naction = to_numpy(get_action(naction))  # shape (num_samples, len_traj_pred, 2)

    # For now, pick the first sample
    chosen_traj = naction[0]
    # And pick the specific waypoint index
    waypoint = chosen_traj[args.waypoint]  # shape (2,)

    # If model was trained in normalized action space, rescale
    print(f"waypoint dist. = {np.linalg.norm(waypoint)}")
    if model_params.get("normalize", False):
        # e.g., scale by max velocity for a single step
        # waypoint *= MAX_V / RATE
        pass
        # print(f"Normalized waypoint={waypoint}")

    return waypoint  # (dx, dy)


##############################################################################
# OmniGibson main loop
##############################################################################
def main(random_selection=False, headless=False, short_exec=False):
    """
    Main function to run OmniGibson with your diffusion-based waypoint sampler,
    using PD logic from `pd_controller.py` to convert (dx, dy) -> (v, w).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    # 1) Load Model Config
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)

    # Example of direct usage (hard-coded for 'nomad'):
    model_config_path = os.path.join(MODEL_TRAIN_PATH, "config", "nomad.yaml")
    # ckpt_path = os.path.join(MODEL_DEPLOY_PATH, "model_weights", "nomad.pth")
    ckpt_path = os.path.join(MODEL_DEPLOY_PATH, "model_weights", "latest.pth")
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)
        print(f"[INFO] Successfully loaded model config from {model_config_path}")

    # 2) Load Model
    model = load_model(ckpt_path, model_params, device)
    model.eval()

    # 3) Diffusion Scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=model_params["num_diffusion_iters"],
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    # 4) Create OmniGibson Environment
    config_filename = os.path.join(og.example_config_path, "turtlebot_nav.yaml")
    with open(config_filename, "r") as f:
        config = yaml.safe_load(f)
    # If you only want floors, walls, ceilings:
    config["scene"]["load_object_categories"] = ["floors", "walls", "ceilings"]
    if headless:
        config["headless"] = True

    env = og.Environment(configs=config)
    robots = env.robots
    robot_name = robots[0].name

    # Prepare context queue
    context_queue = []
    context_size = model_params["context_size"]

    # Basic loop parameters
    max_episodes = 10 if not short_exec else 1
    steps_per_episode = 10000

    # Simple argument container
    class ArgObj:
        def __init__(self):
            self.num_samples = 8
            self.waypoint = 2  # which waypoint index to pick

    args = ArgObj()

    # 5) Main loop
    for ep_i in range(max_episodes):
        env.reset()
        context_queue.clear()

        for step_i in range(steps_per_episode):
            # Step environment
            if step_i == 0:
                zero_action = np.array([0.0, 0.0], dtype=np.float32)  # v=0, w=0
                states, rewards, terminated, truncated, infos = env.step(
                    {robot_name: zero_action}
                )
            else:
                states, rewards, terminated, truncated, infos = env.step(
                    {robot_name: action}
                )

            # 6) Retrieve camera data
            robot_state = states[robot_name]
            camera_key = f"{robot_name}:eyes:Camera:0"
            camera_output = robot_state[camera_key]
            rgb_tensor = camera_output["rgb"]  # shape (H, W, 4)

            # Convert to PIL
            obs_img = array_to_pil(rgb_tensor)

            # Update context queue
            if len(context_queue) < context_size + 1:
                context_queue.append(obs_img)
            else:
                context_queue.pop(0)
                context_queue.append(obs_img)

            # 7) If we have enough context, run diffusion
            if len(context_queue) > context_size:

                # (dx, dy)
                waypoint_dxdy = sample_diffusion_action(
                    model=model,
                    obs_images=context_queue,
                    model_params=model_params,
                    device=device,
                    noise_scheduler=noise_scheduler,
                    args=args,
                )

                # PD Controller -> (v, w)
                print(f"waypoint_dxdy={waypoint_dxdy}")
                action_vw = pd_controller(waypoint_dxdy, DT, MAX_V, MAX_W)
                action = action_vw
            else:
                # Not enough context, remain still
                action = np.array([0.0, 0.0], dtype=np.float32)

            print(f"[Episode={ep_i}, Step={step_i}] action={action}")

            # Termination condition
            if terminated or truncated:
                break

    # Cleanup
    og.clear()
    print("[INFO] Finished simulation.")


if __name__ == "__main__":
    # Example usage:
    # main(headless=True, short_exec=True)
    main()
