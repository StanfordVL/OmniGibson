import os
import yaml
import numpy as np
import torch
import omnigibson as og
from PIL import Image as PILImage

# Diffusion
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# NoMaD 관련 유틸 (사용자 환경에 맞게 import 경로 수정 필요)
from deployment.src.utils import to_numpy, transform_images, load_model
from train.vint_train.training.train_utils import get_action


##############################################################################
# Omnigibson에서 얻은 RGB array -> PIL.Image 변환
##############################################################################
def array_to_pil(rgb_array: np.ndarray) -> PILImage.Image:
    """
    Omnigibson 카메라에서 얻은 (H, W, 3) uint8 RGB array를 PIL.Image로 변환.
    """
    return PILImage.fromarray(rgb_array)


##############################################################################
# Diffusion(DDPM)으로 Action을 샘플링하는 함수
##############################################################################
def sample_diffusion_action(
    model,
    obs_images,  # (context_size + 1)개의 PIL 이미지
    model_params,
    device,
    noise_scheduler: DDPMScheduler,
    num_samples: int = 8,
    waypoint_idx: int = 2,
):
    """
    obs_images : 최근 관측 이미지 목록 (PIL.Image).
                 보통 (context_size + 1)장이 필요.
    model_params: 모델 설정 (num_diffusion_iters, context_size, image_size, len_traj_pred, normalize 등)
    device: torch.device
    noise_scheduler: DDPMScheduler
    num_samples: 몇 개의 액션 샘플을 동시에 생성할지
    waypoint_idx: 샘플된 여러 waypoint 중 어느 인덱스를 최종적으로 쓸지
    """
    # (1) PIL 이미지를 모델 입력 텐서로 변환
    obs_tensor = transform_images(obs_images, model_params["image_size"], center_crop=False).to(device)

    # (2) 가짜 goal 이미지, goal 무시용 mask
    fake_goal = torch.randn((1, 3, *model_params["image_size"])).to(device)
    mask = torch.ones(1).long().to(device)  # goal은 사용 안 함

    # (3) Vision encoder로 관측 임베딩 추출
    with torch.no_grad():
        obs_cond = model("vision_encoder", obs_img=obs_tensor, goal_img=fake_goal, input_goal_mask=mask)

        # obs_cond 크기: (1, D) 또는 (1, T, D) 등 → num_samples 배치로 반복
        if len(obs_cond.shape) == 2:
            obs_cond = obs_cond.repeat(num_samples, 1)
        else:
            obs_cond = obs_cond.repeat(num_samples, 1, 1)

        # (4) 행동 초기값 (가우시안 노이즈)
        noisy_action = torch.randn((num_samples, model_params["len_traj_pred"], 2), device=device)
        naction = noisy_action

        # (5) Diffusion 역추론
        noise_scheduler.set_timesteps(model_params["num_diffusion_iters"])
        for k in noise_scheduler.timesteps:
            noise_pred = model("noise_pred_net", sample=naction, timestep=k, global_cond=obs_cond)
            naction = noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample

    # (6) 텐서를 numpy로 변환
    naction = to_numpy(get_action(naction))

    # (7) 여러 샘플 중 하나 선택 (기본적으로 [0])
    chosen_traj = naction[0]
    # 그리고 해당 traj에서 특정 인덱스 waypoint를 최종 액션으로
    chosen_waypoint = chosen_traj[waypoint_idx]

    # (8) 필요하다면 속도 스케일링
    if model_params.get("normalize", False):
        # 예시: max_v, RATE 등이 model_params에 있거나 따로 load한 값에 있으면:
        # max_v = 0.2
        # RATE = 10
        # chosen_waypoint *= (max_v / RATE)
        pass

    return chosen_waypoint


##############################################################################
# 메인 실행 함수
##############################################################################
def main(random_selection=False, headless=False, short_exec=False):
    """
    OmniGibson 환경에서 NoMaD Diffusion 모델을 사용하여 탐색(exploration)을 수행하는 예시 코드.
    - 로봇 카메라 이미지를 수집
    - 일정량(context_size+1) 모이면 Diffusion 모델로 행동을 샘플링
    - 액션을 env.step()에 적용하여 로봇을 이동
    """
    # -----------------------------------------------------------
    # (A) 모델 설정 로딩
    # -----------------------------------------------------------
    og.log.info(f"Demo {__file__}\n{'*'*80}\nDescription:\n{main.__doc__}\n{'*'*80}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 작업 폴더
    work_dir = os.getcwd()

    # 사용자 환경에 맞춰 경로 수정
    # 예: models.yaml, nomad.yaml, .pth 파일이 모두 test_NoMaD 폴더에 있다고 가정
    MODEL_DEPLOY_PATH = os.path.join(work_dir, "test_NoMaD", "deployment")
    MODEL_TRAIN_PATH = os.path.join(work_dir, "test_NoMaD", "train")
    MODEL_CONFIG_PATH = os.path.join(MODEL_DEPLOY_PATH, "config", "models.yaml")

    model_name = "nomad"  # 사용할 모델 이름
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)

    # 모델 세부 설정 yaml
    # (아래는 예시로 nomad.yaml를 직접 지정)
    model_config_path = os.path.join(MODEL_TRAIN_PATH, "config", "nomad.yaml")
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)

    # Checkpoint 경로 (기본적으로 models.yaml에서 불러오되, 여기선 예시로 강제)
    # ckpth_path = model_paths[model_name]["ckpt_path"]  # models.yaml에서 읽어온 경로
    ckpth_path = os.path.join(MODEL_DEPLOY_PATH, "model_weights", "nomad.pth")  # 직접 지정

    # 모델 로드
    print(f"Loading model from: {ckpth_path}")
    model = load_model(ckpth_path, model_params, device)
    model.to(device)
    model.eval()

    # Diffusion 스케줄러 준비
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=model_params["num_diffusion_iters"],
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    # -----------------------------------------------------------
    # (B) OmniGibson 환경 구성
    # -----------------------------------------------------------
    config_filename = os.path.join(og.example_config_path, "turtlebot_nav.yaml")
    with open(config_filename, "r") as f:
        config = yaml.safe_load(f)

    # Load mode: Quick
    config["scene"]["load_object_categories"] = ["floors", "walls", "ceilings"]

    # headless 모드를 적용하고 싶다면:
    env = og.Environment(configs=config)

    robots = env.robots
    robot_name = robots[0].name
    og.log.info(f"Loaded robots: {[robot.name for robot in robots]}")

    # (C) 상태 버퍼
    context_queue = []
    context_size = model_params["context_size"]

    # (D) 에피소드 / 스텝 반복
    max_iterations = 10 if not short_exec else 1
    steps_per_ep = 100

    # 필요하면 argparse 대체용 설정
    class Args:
        num_samples = 8
        waypoint = 2

    args = Args()

    for ep_i in range(max_iterations):
        og.log.info(f"=== Resetting environment, episode={ep_i} ===")
        env.reset()
        context_queue.clear()

        # 첫 스텝의 액션을 (0.0, 0.0)로 시작
        action = np.array([0.0, 0.0], dtype=np.float32)

        for step_i in range(steps_per_ep):
            # 1) 환경 진행
            applied_action = {robot_name: action}
            states, rewards, terminated, truncated, infos = env.step(applied_action)

            # 2) 카메라 이미지 획득
            robot_state = states[robot_name]
            camera_key = f"{robot_name}:eyes:Camera:0"
            camera_output = robot_state[camera_key]  # dict with "rgb", "depth", "seg" etc.
            rgb_array = camera_output["rgb"][..., :3]  # (H, W, 3) uint8

            # 3) PIL 변환 후 큐에 저장
            obs_img = array_to_pil(rgb_array)
            if len(context_queue) < context_size + 1:
                context_queue.append(obs_img)
            else:
                context_queue.pop(0)
                context_queue.append(obs_img)

            # 4) Diffusion 모델로 액션 결정
            if len(context_queue) > context_size:
                # 충분한 이미지가 모였으면 Diffusion 샘플링
                chosen_waypoint = sample_diffusion_action(
                    model=model,
                    obs_images=context_queue,
                    model_params=model_params,
                    device=device,
                    noise_scheduler=noise_scheduler,
                    num_samples=args.num_samples,
                    waypoint_idx=args.waypoint,
                )
                # chosen_waypoint가 [vx, wz]라면:
                action = np.array(chosen_waypoint, dtype=np.float32)
            else:
                # 아직 queue가 부족하면 0액션
                action = np.array([0.0, 0.0], dtype=np.float32)

            # Debug print
            print(f"[ep={ep_i}, step={step_i}] action={action}, reward={rewards[robot_name]:.4f}")

            if terminated or truncated:
                og.log.info(f"Episode finished early at step={step_i}")
                break

    # -----------------------------------------------------------
    # (E) 종료
    # -----------------------------------------------------------
    og.clear()


if __name__ == "__main__":
    # 필요하다면 외부에서 argparse를 써서 headless, short_exec 등을 받을 수 있음
    main(random_selection=False, short_exec=False)
