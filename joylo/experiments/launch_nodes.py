from dataclasses import dataclass
from pathlib import Path

import tyro

from gello.robots.robot import BimanualRobot, PrintRobot
from gello.zmq_core.robot_node import ZMQServerRobot


@dataclass
class Args:
    domain: str = "sim"     # Real or sim
    robot: str = "R1"       # OG robot class name
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    robot_ip: str = "192.168.1.10"


def launch_robot_server(args: Args):
    port = args.robot_port

    # Only sim is supported for now
    if args.domain == "real":
        raise NotImplementedError
    else:
        # Make sure sim is selected
        assert args.domain == "sim"
        from gello.robots.sim_robot.og_sim import OGRobotServer
        server = OGRobotServer(robot=args.robot, port=port, host=args.hostname)
        server.serve()


def main(args):
    launch_robot_server(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
