import os
import yaml
import numpy as np
import torch
import omnigibson as og

# from omnigibson.utils.ui_utils import choose_from_options
from PIL import Image as PILImage
from deployment.src.utils import to_numpy, transform_images, load_model
from train.vint_train.training.train_utils import get_action


# Diffusers
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


##############################################################################
# 경로 설정
##############################################################################
work_dir = os.getcwd()
MODEL_DEPLOY_PATH = os.path.join(work_dir, "test_NoMaD", "deployment")
MODEL_TRAIN_PATH = os.path.join(work_dir, "test_NoMaD", "train")
MODEL_CONFIG_PATH = os.path.join(MODEL_DEPLOY_PATH, "config", "models.yaml")
ROBOT_CONFIG_PATH = os.path.join(MODEL_DEPLOY_PATH, "config", "robot.yaml")


##############################################################################
# "msg_to_pil" 대체: Omnigibson에서 받은 torch.Tensor (H, W, 4) -> (H, W, 3) -> PIL.Image
##############################################################################
def array_to_pil(rgb_tensor: torch.Tensor) -> PILImage.Image:
    """
    rgb_tensor: shape (H, W, 4) 혹은 (H, W, 3), dtype은 float32 또는 uint8 등
    """
    # 만약 (H, W, 4)라면 마지막 채널(Alpha) 제거
    if rgb_tensor.shape[-1] == 4:
        rgb_tensor = rgb_tensor[..., :3]

    # GPU 상에 있을 수 있으니 cpu로 이동, PIL 변환 위해 uint8로 변환
    rgb_array = rgb_tensor.cpu().numpy().astype(np.uint8)

    # (H, W, 3)의 numpy 배열을 PIL로 변환
    return PILImage.fromarray(rgb_array)


##############################################################################
# Diffusion으로 action을 샘플링하는 보조 함수 (이전 ROS 코드와 유사)
##############################################################################
def sample_diffusion_action(
    model,
    obs_images,  # (context_size + 1)개의 PIL 이미지
    model_params,
    device,
    noise_scheduler: DDPMScheduler,
    args,
):
    """
    obs_images : 최근 관측 이미지 목록 (PIL 이미지)
    model_params: 모델 설정 (num_diffusion_iters, context_size, image_size, len_traj_pred, 등)
    noise_scheduler: DDPMScheduler
    args: num_samples, waypoint 등 하이퍼파라미터
    """
    # (1) PIL 이미지를 모델 입력에 맞게 변환
    #     예) transform_images(): Resize -> Tensor -> [0,1] 정규화 등
    obs_tensor = transform_images(obs_images, model_params["image_size"], center_crop=False).to(device)

    # (2) 가짜 goal 이미지 (원본 코드처럼)
    fake_goal = torch.randn((1, 3, *model_params["image_size"])).to(device)
    mask = torch.ones(1).long().to(device)  # goal 무시용 mask

    # (3) Vision encoder로 관측 임베딩 추출
    with torch.no_grad():
        obs_cond = model("vision_encoder", obs_img=obs_tensor, goal_img=fake_goal, input_goal_mask=mask)
        # 배치 크기: num_samples로 반복
        if len(obs_cond.shape) == 2:
            obs_cond = obs_cond.repeat(args.num_samples, 1)
        else:
            obs_cond = obs_cond.repeat(args.num_samples, 1, 1)

        # (4) 행동 샘플 초깃값: 가우시안 노이즈
        noisy_action = torch.randn((args.num_samples, model_params["len_traj_pred"], 2), device=device)
        naction = noisy_action

        # (5) Diffusion 역추론
        noise_scheduler.set_timesteps(model_params["num_diffusion_iters"])
        for k in noise_scheduler.timesteps:
            # predict noise
            noise_pred = model("noise_pred_net", sample=naction, timestep=k, global_cond=obs_cond)

            # inverse diffusion step (remove noise)
            naction = noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample

    # (6) Tensor -> NumPy
    naction = to_numpy(get_action(naction))

    # (7) 샘플 중 하나 선택 (원본 코드는 [0]을 사용)
    chosen_traj = naction[0]  # change this based on heuristic

    # 해당 trajectory에서 특정 waypoint 인덱스 하나만 뽑아서 반환
    waypoint = chosen_traj[args.waypoint]

    # 속도 정규화를 사용하는 경우
    if model_params.get("normalize", False):
        # waypoint = waypoint * (MAX_V / RATE)
        # print(f"The normalization is not implemented yet.")
        # 예: max_v와 frame_rate가 있다면
        # max_v = ...
        # RATE = ...
        # waypoint *= (max_v / RATE)
        pass

    return waypoint


##############################################################################
# Omnigibson 메인 루프
##############################################################################
def main(random_selection=False, headless=False, short_exec=False):
    # (A) 모델 설정 및 로드
    model_name = "nomad"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 모델 경로 설정
    # work_dir = os.getcwd()
    # MODEL_DEPLOY_PATH = os.path.join(work_dir, "test_NoMaD", "deployment")
    # MODEL_TRAIN_PATH = os.path.join(work_dir, "test_NoMaD", "train")
    # MODEL_CONFIG_PATH = os.path.join(MODEL_DEPLOY_PATH, "config", "models.yaml")
    # ROBOT_CONFIG_PATH = os.path.join(MODEL_DEPLOY_PATH, "config", "robot.yaml")

    # 모델 설정 로드
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)

    with open(ROBOT_CONFIG_PATH, "r") as f:
        robot_config = yaml.safe_load(f)
    MAX_V = robot_config["max_v"]
    MAX_W = robot_config["max_w"]
    RATE = robot_config["frame_rate"]

    # model_config_path와 ckpth_path 설정 (사용자 환경에 맞춤)
    model_config_path = os.path.join(MODEL_TRAIN_PATH, "config", "nomad.yaml")  # path to the model config
    ckpth_path = os.path.join(MODEL_DEPLOY_PATH, "model_weights", "nomad.pth")  # path to the pre-trained weights

    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)

    model = load_model(ckpth_path, model_params, device)
    model.eval()

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=model_params["num_diffusion_iters"],
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    # (B) Omnigibson 환경 구성
    config_filename = os.path.join(og.example_config_path, "turtlebot_nav.yaml")
    with open(config_filename, "r") as f:
        config = yaml.safe_load(f)

    config["scene"]["load_object_categories"] = ["floors", "walls", "ceilings"]
    env = og.Environment(configs=config)

    robots = env.robots
    robot_name = robots[0].name

    # (C) context
    context_queue = []
    context_size = model_params["context_size"]

    max_iterations = 10 if not short_exec else 1
    steps_per_ep = 2000

    class ArgObj:
        def __init__(self):
            self.num_samples = 8
            self.waypoint = 2

    args = ArgObj()

    for ep_i in range(max_iterations):
        env.reset()
        context_queue.clear()

        for step_i in range(steps_per_ep):
            # 1) step
            if step_i == 0:
                zero_action = np.array([0.0, 0.0], dtype=np.float32)
                states, rewards, terminated, truncated, infos = env.step({robot_name: zero_action})
            else:
                # action = np.array([20.0, 0.0], dtype=np.float32)
                states, rewards, terminated, truncated, infos = env.step({robot_name: action})

            # 2) 카메라 텐서 획득 (Omnigibson이 torch.Tensor로 반환한다고 가정)
            robot_state = states[robot_name]
            camera_key = f"{robot_name}:eyes:Camera:0"
            camera_output = robot_state[camera_key]
            rgb_tensor = camera_output["rgb"]  # shape (H, W, 4) / torch.Tensor

            # 3) PIL 변환 + context 큐에 저장
            obs_img = array_to_pil(rgb_tensor)  # (H, W, 4) -> (H, W, 3) -> PIL.Image
            if len(context_queue) < context_size + 1:
                context_queue.append(obs_img)
            else:
                context_queue.pop(0)
                context_queue.append(obs_img)

            # 4) Diffusion 액션 샘플링
            if len(context_queue) > context_size:
                action_waypoint = sample_diffusion_action(
                    model=model,
                    obs_images=context_queue,
                    model_params=model_params,
                    device=device,
                    noise_scheduler=noise_scheduler,
                    args=args,
                )
                action = np.array(action_waypoint, dtype=np.float32)
            else:
                action = np.array([0.0, 0.0], dtype=np.float32)

            print(f"[ep={ep_i}, step={step_i}] action={action}")

            if terminated or truncated:
                break

    og.clear()


if __name__ == "__main__":
    main()
