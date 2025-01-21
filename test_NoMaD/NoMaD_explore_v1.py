import os
import yaml
import numpy as np
import torch
import omnigibson as og
from omnigibson.utils.ui_utils import choose_from_options
from PIL import Image as PILImage
from deployment.src.utils import to_numpy, transform_images, load_model

# Diffusers
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# 이 부분은 이전에 사용한 utils / train_utils 등에서 필요한 함수들을 임포트한다고 가정합니다.
# from utils import to_numpy, transform_images, load_model
# from vint_train.training.train_utils import get_action


##############################################################################
# (예) "msg_to_pil" 대체: Omnigibson RGB array -> PIL 이미지로 변환 예시 함수
##############################################################################
def array_to_pil(rgb_array: np.ndarray) -> PILImage.Image:
    # rgb_array: (H, W, 3) uint8 (Omnigibson 카메라 출력)
    return PILImage.fromarray(rgb_array)


##############################################################################
# Diffusion으로 action을 샘플링하는 보조 함수 (이전 ROS 코드와 유사)
##############################################################################
def sample_diffusion_action(
    model,
    obs_images,  # (context_size + 1)개의 PIL 이미지
    model_params,
    device,
    noise_scheduler,
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
            noise_pred = model("noise_pred_net", sample=naction, timestep=k, global_cond=obs_cond)
            naction = noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample

    # (6) Tensor -> NumPy
    naction = to_numpy(get_action(naction))

    # (7) 샘플 중 하나 선택 (원본 코드는 [0]을 사용)
    chosen_traj = naction[0]
    # 해당 trajectory에서 특정 waypoint 인덱스 하나만 뽑아서 반환
    waypoint = chosen_traj[args.waypoint]

    # 속도 정규화를 사용하는 경우
    if model_params.get("normalize", False):
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
    """
    Omnigibson 환경에서 Diffusion 모델을 이용해 action을 샘플링하고 로봇을 이동시키는 예시
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + (main.__doc__ or "") + "*" * 80)

    # (A) 모델 설정 로드 (이전 ROS 코드와 동일하게)
    # 예: ../config/models.yaml 파일에서 특정 모델을 골라 그 안의 config_path, ckpt_path를 읽어온다고 가정
    MODEL_CONFIG_PATH = "../config/models.yaml"
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)

    # 가령 'nomad'라는 모델을 쓰기로 가정
    model_name = "nomad"
    model_config_path = model_paths[model_name]["config_path"]
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)

    ckpth_path = model_paths[model_name]["ckpt_path"]

    # (B) 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 사용자 정의 함수 load_model 사용
    model = load_model(ckpth_path, model_params, device)
    model.to(device)
    model.eval()

    # (C) DDPMScheduler 설정
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=model_params["num_diffusion_iters"],
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    # (D) Omnigibson 환경 구성 로드
    config_filename = os.path.join(og.example_config_path, "turtlebot_nav.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # load_mode를 "Quick"로 (floors, walls, ceilings만)
    config["scene"]["load_object_categories"] = ["floors", "walls", "ceilings"]

    env = og.Environment(configs=config)
    robots = env.robots
    robot_name = robots[0].name
    og.log.info(f"Loaded robots: {[robot.name for robot in robots]}")

    # (E) 이미지를 저장할 컨텍스트 버퍼
    context_queue = []
    context_size = model_params["context_size"]

    # (F) 에피소드 루프
    max_iterations = 10 if not short_exec else 1
    steps_per_ep = 100

    # 편의상 argparse 대체용
    class ArgObj:
        def __init__(self):
            self.num_samples = 8
            self.waypoint = 2

    args = ArgObj()

    for ep_i in range(max_iterations):
        og.log.info(f"Resetting environment, episode={ep_i}")
        env.reset()
        context_queue.clear()

        for step_i in range(steps_per_ep):
            # 1) Omnigibson에서 step 하기 전에, 카메라 이미지를 얻으려면
            #    먼저 "현재 상태"를 받아야 합니다.
            #    Omnigibson의 경우, action을 먼저 넘기지 않으면 state 업데이트가 안 될 수도 있으니,
            #    매 step에서 action -> env.step -> state 조회가 일반적입니다.
            #    또는 zero-action으로 한 번 step해서 현재 화면을 받도록 할 수도 있습니다.

            # 여기서는 "직전 step" 결과를 바탕으로 카메라 이미지를 얻는다고 가정
            # 만약 첫 step이라면 action=[0,0]을 미리 줘도 됩니다.
            if step_i == 0:
                # 첫 단계에서는 아무것도 안 하는 액션
                zero_action = np.array([0.0, 0.0], dtype=np.float32)
                states, rewards, terminated, truncated, infos = env.step({robot_name: zero_action})

            # 이제 robot_state에서 카메라 이미지 획득
            robot_state = env.last_state[robot_name]  # 혹은 states[robot_name]
            camera_key = f"{robot_name}:eyes:Camera:0"
            camera_output = robot_state[camera_key]  # dict with "rgb", "seg", "depth" 등
            rgb_array = camera_output["rgb"]  # (H, W, 3) in uint8

            # 2) PIL Image로 변환 후 큐에 쌓기
            obs_img = array_to_pil(rgb_array)
            if len(context_queue) < context_size + 1:
                context_queue.append(obs_img)
            else:
                context_queue.pop(0)
                context_queue.append(obs_img)

            # 3) 만약 컨텍스트 사이즈가 충족되면 Diffusion으로 행동 샘플링
            if len(context_queue) > context_size:
                action_waypoint = sample_diffusion_action(
                    model=model,
                    obs_images=context_queue,
                    model_params=model_params,
                    device=device,
                    noise_scheduler=noise_scheduler,
                    args=args,
                )
                # 예: action_waypoint = [v, w]
                action = np.array(action_waypoint, dtype=np.float32)
            else:
                # 아직 충분한 이미지가 없으면 일단 정지 or 간단한 random
                action = np.array([0.0, 0.0], dtype=np.float32)

            # 4) Omnigibson에 최종 액션 적용
            states, rewards, terminated, truncated, infos = env.step({robot_name: action})

            # Debug print
            print(f"[ep={ep_i}, step={step_i}] action={action}, reward={rewards[robot_name]}")

            if terminated or truncated:
                og.log.info(f"Episode finished early at step={step_i}")
                break

    # 마지막에 환경 종료
    og.clear()


if __name__ == "__main__":
    main()
