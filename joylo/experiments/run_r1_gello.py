import datetime
import glob
import time
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch as th
import numpy as np
import tyro

from gello.agents.agent import BimanualAgent, DummyAgent, MultiControllerAgent
from gello.agents.gello_agent import GelloAgent, DynamixelRobotConfig
from gello.agents.r1_gello_agent import R1GelloAgent, MotorFeedbackConfig
from gello.agents.joycon_agent import JoyconAgent
from gello.data_utils.format_obs import save_frame
from gello.env import RobotEnv
from gello.robots.robot import PrintRobot
from gello.zmq_core.robot_node import ZMQClientRobot
from gello import REPO_DIR

def print_color(*args, color=None, attrs=(), **kwargs):
    import termcolor

    if len(args) > 0:
        args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    print(*args, **kwargs)


@dataclass
class Args:
    robot_port: int = 6001
    wrist_camera_port: int = 5000
    base_camera_port: int = 5001
    hostname: str = "127.0.0.1"
    robot_type: str = None  # only needed for quest agent or spacemouse agent
    hz: int = 100
    start_joints: Optional[Tuple[float, ...]] = None
    joint_config_file: str = "joint_config_black_gello.yaml"

    gello_port: Optional[str] = None
    mock: bool = False
    use_save_interface: bool = False
    damping_motor_kp: float = 0.3
    motor_feedback_type: str = "NONE"
    use_joycons: bool = True
    data_dir: str = "~/bc_data"
    verbose: bool = False


def main(args):
    if args.mock:
        robot_client = PrintRobot(8, dont_print=True)
        camera_clients = {}
    else:
        camera_clients = {
            # you can optionally add camera nodes here for imitation learning purposes
            # "wrist": ZMQClientCamera(port=args.wrist_camera_port, host=args.hostname),
            # "base": ZMQClientCamera(port=args.base_camera_port, host=args.hostname),
        }
        robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)

    active_arm = "right"
    env = RobotEnv(robot_client, control_rate_hz=args.hz, camera_dict=camera_clients, active_arm=active_arm)

    gello_port = args.gello_port
    if gello_port is None:
        import platform

        if platform.system().lower() == "linux":
            usb_ports = glob.glob("/dev/serial/by-id/*")
        elif platform.system().lower() == "darwin":
            usb_ports = glob.glob("/dev/cu.usbserial-*")
        else:
            raise ValueError(f"Unsupported platform {platform.system()}")
        print(f"Found {len(usb_ports)} ports")
        if len(usb_ports) > 0:
            gello_port = usb_ports[0]
            print(f"using port {gello_port}")
        else:
            raise ValueError(
                "No gello port found, please specify one or plug in gello"
            )
        
    # Read joint config from yaml
    # Update for the gello you are using
    with open(f"{REPO_DIR}/configs/{args.joint_config_file}", "r") as file:
        joint_config = yaml.load(file, Loader=yaml.SafeLoader)

    dynamixel_config = DynamixelRobotConfig(
        joint_ids=tuple(np.arange(16).tolist()),
        joint_offsets=[np.deg2rad(x) for x in joint_config['joints']['offsets']],
        joint_signs=joint_config['joints']['signs'],
        gripper_config=None,  # (8, 202, 152),
    )

    start_joints = args.start_joints
    if start_joints is None:
        # Neutral pose
        start_joints = np.array([
            np.pi/2, np.pi/2,
            np.pi, np.pi,
            -np.pi,
            0,
            0,
            0,
            -np.pi/2, -np.pi/2,
            np.pi, np.pi,
            -np.pi,
            0,
            0,
            0,
        ])

    base_trunk_gripper_agent = None
    if args.use_joycons:
        base_trunk_gripper_agent = JoyconAgent(
            calibration_dir=f"{REPO_DIR}/configs",
            deadzone_threshold=0.2,
            max_translation=0.5,
            max_rotation=0.8,
            max_trunk_translate=0.2,
            max_trunk_tilt=0.4,
            enable_rumble=True,
        )

    arm_agent = R1GelloAgent(
        port=gello_port,
        dynamixel_config=dynamixel_config,
        start_joints=start_joints,
        default_joints=None,
        damping_motor_kp=args.damping_motor_kp,
        motor_feedback_type=MotorFeedbackConfig[args.motor_feedback_type],
        enable_locking_joints=True,
        joycon_agent=base_trunk_gripper_agent,
    )
    agent_parts = [arm_agent]
    if base_trunk_gripper_agent is not None:
        agent_parts.append(base_trunk_gripper_agent)

    agent = MultiControllerAgent(agents=agent_parts)

    # going to start position
    print("Going to start position")
    agent.reset()

    print_color("\nWaiting for environment...", color="magenta", attrs=("bold",))
    obs = env.get_obs()

    # Enable torque mode and move arms to current observed pose from environment
    obs_jnts = np.concatenate([obs[f"arm_{arm}_joint_positions"].detach().cpu().numpy() for arm in ["left", "right"]])
    obs_jnts = obs_jnts[[0, 0, 1, 1, 2, 3, 4, 5, 6, 6, 7, 7, 8, 9, 10, 11]]
    arm_agent.set_reset_qpos(qpos=obs_jnts)
    arm_agent.reset()

    print_color("*" * 40, color="magenta", attrs=("bold",))
    print_color("\nWelcome to GELLO!\n", color="magenta", attrs=("bold",))
    print_color("R1 Teleoperation Commands:\n", color="magenta", attrs=("bold",))
    print_color(f"\t ZL / ZR: Toggle grasping", color="magenta", attrs=("bold",))
    print_color(f"\t Left Joystick (not pressed): Translate the robot base", color="magenta", attrs=("bold",))
    print_color(f"\t Right Joystick: Rotate the robot base + tilt the trunk torso", color="magenta", attrs=("bold",))
    print_color(f"\t Up / Down Button: Raise / Lower the trunk torso", color="magenta", attrs=("bold",))
    print_color(f"\t L / R: Lock the lower two wrist joints while leaving the upper joints free", color="magenta", attrs=("bold",))
    print_color(f"\t - / +: Lock the upper joints while leaving the lower wrist roll joint free.\n\t\tThe wrist pose will NOT be tracked while held.", color="magenta", attrs=("bold",))
    print_color(f"\t Y: Move the robot towards its reset pose", color="magenta", attrs=("bold",))
    print_color(f"\t B: Toggle camera", color="magenta", attrs=("bold",))
    print_color(f"\t Home: Reset the environment", color="magenta", attrs=("bold",))

    started = base_trunk_gripper_agent is None      # Only wait for start signal if we're using joycons
    if not started:
        # Wait for user to begin teleoperation
        print_color("\nPress 'X' to begin teleoperation...", color="yellow", attrs=("bold",))
        while not started:
            started = base_trunk_gripper_agent.jc_right.get_button_x()

    # Start the agent
    agent.start()

    if args.use_save_interface:
        from gello.data_utils.keyboard_interface import KBReset

        kb_interface = KBReset()

    print_color("\nStart 🚀🚀🚀", color="green", attrs=("bold",))

    save_path = None
    start_time = time.time()
    while True:
        num = time.time() - start_time
        message = f"\rTime passed: {round(num, 2)}          "
        print_color(
            message,
            color="white",
            attrs=("bold",),
            end="",
            flush=True,
        )
        action = agent.act(obs)
        dt = datetime.datetime.now()
        if args.use_save_interface:
            state = kb_interface.update()
            if state == "start":
                dt_time = datetime.datetime.now()
                save_path = (
                    Path(args.data_dir).expanduser()
                    / args.agent
                    / dt_time.strftime("%m%d_%H%M%S")
                )
                save_path.mkdir(parents=True, exist_ok=True)
                print(f"Saving to {save_path}")
            elif state == "save":
                assert save_path is not None, "something went wrong"
                save_frame(save_path, dt, obs, action)
            elif state == "normal":
                save_path = None
            else:
                raise ValueError(f"Invalid state {state}")
        obs = env.step(action)


if __name__ == "__main__":
    main(tyro.cli(Args))
