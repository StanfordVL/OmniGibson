import glob
import time
import yaml
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import tyro

from gello.agents.agent import MultiControllerAgent
from gello.agents.gello_agent import DynamixelRobotConfig, MotorFeedbackConfig
from gello.agents.r1pro_gello_agent import R1ProGelloAgent 
from gello.agents.joycon_agent import JoyconAgent
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
    joint_config_file: str = "joint_config_black_gello_pro.yaml"

    gello_port: Optional[str] = None
    mock: bool = False
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
        joint_ids=tuple(np.arange(18).tolist()),
        joint_offsets=[np.deg2rad(x) for x in joint_config['joints']['offsets']],
        joint_signs=joint_config['joints']['signs'],
        gripper_config=None,  # (8, 202, 152),
    )

    start_joints = args.start_joints
    if start_joints is None:
        # Neutral pose
        # TODO: determine this for R1Pro
        start_joints = np.zeros(18)

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

    arm_agent = R1ProGelloAgent(
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

    # Create and start the agent
    agent = MultiControllerAgent(agents=agent_parts)
    agent.start()

    # First go to canonical start position
    print("Going to start position")
    agent.reset()

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

    obs = env.get_obs()
    print_color("\nStart 🚀🚀🚀", color="green", attrs=("bold",))
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
        obs = env.step(action)


if __name__ == "__main__":
    main(tyro.cli(Args))
