import yaml
import numpy as np

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
import omnigibson.utils.transform_utils as T
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.object_states import ContactBodies
    

def pause(time):
    for _ in range(int(time*100)):
        og.sim.step()

def replay_controller(env, filename):
    actions = yaml.load(open(filename, "r"), Loader=yaml.FullLoader)
    for action in actions:
        env.step(action)

def execute_controller(ctrl_gen, env, filename=None):
    actions = []
    for action in ctrl_gen:
        env.step(action)
        actions.append(action.tolist())
    if filename is not None:
        with open(filename, "w") as f:
            yaml.dump(actions, f)

def main():
    # Load the config
    config_filename = "test_tiago.yaml"
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls", "coffee_table"]

    # Load the environment
    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    table = DatasetObject(
        name="table",
        category="breakfast_table",
        model="rjgmmy",
        scale = 0.3
    )
    og.sim.import_object(table)
    table.set_position([1.0, 1.0, 0.58])

    grasp_obj = DatasetObject(
        name="potato",
        category="cologne",
        model="lyipur",
        scale=0.01
    )
    og.sim.import_object(grasp_obj)
    grasp_obj.set_position([-0.3, -0.8, 0.5])
    og.sim.step()

    controller = StarterSemanticActionPrimitives(None, scene, robot)

    # Need to set start pose because default tuck pose for Fetch collides with itself
    def set_start_pose():
        default_pose = np.array(
            [
                0.0,
                0.0,  # wheels
                0.0,  # trunk
                0.0,
                0.0,
                0.0,  # head
                -0.22184,
                1.53448,
                1.46076,
                -0.84995,
                1.36904,
                1.90996,  # arm
                0.05,
                0.05,  # gripper
            ]
        )
        robot.set_joint_positions(default_pose)
        og.sim.step()
    
    def test_navigate_to_obj():
        # execute_controller(controller._navigate_to_obj(table), env)
        pose_2d = np.array([0.5, 0.5, 0.0])
        execute_controller(controller._navigate_to_pose(pose_2d), env)

    def test_grasp_no_navigation():
        # set_start_pose()
        robot.set_position([-0.1, -0.35, 0.05])
        robot.set_orientation(T.euler2quat([0, 0,-np.pi/1.5]))
        # pause(100)
        og.sim.step()
        execute_controller(controller.grasp(grasp_obj), env)
        # replay_controller(env, "grasp_tiago.yaml")

    def detect_robot_collision(robot, filter_objs=[]):
        filter_categories = ["floors"]
        
        obj_in_hand = robot._ag_obj_in_hand[robot.default_arm]
        if obj_in_hand is not None:
            filter_objs.append(obj_in_hand)

        collision_prims = list(robot.states[ContactBodies].get_value(ignore_objs=tuple(filter_objs)))

        for col_prim in collision_prims:
            tokens = col_prim.prim_path.split("/")
            obj_prim_path = "/".join(tokens[:-1])
            col_obj = og.sim.scene.object_registry("prim_path", obj_prim_path)
            if col_obj.category in filter_categories:
                collision_prims.remove(col_prim)

        return len(collision_prims) > 0 or detect_self_collision(robot)

    def detect_self_collision(robot):
        contacts = robot.contact_list()
        robot_links = [link.prim_path for link in robot.links.values()]
        disabled_pairs = [set(p) for p in robot.disabled_collision_pairs]
        for c in contacts:
            link0 = c.body0.split("/")[-1]
            link1 = c.body1.split("/")[-1]
            if {link0, link1} not in disabled_pairs and c.body0 in robot_links and c.body1 in robot_links:
                return True
        return False


    robot.untuck()
    og.sim.step()
    while True:
        coll = []
        robot_links = [link.prim_path for link in robot.links.values()]
        for c in robot.contact_list():
            if c.body0 in robot_links and c.body1 in robot_links:
                link0 = c.body0.split("/")[-1]
                link1 = c.body1.split("/")[-1]
                pair = {link0, link1}
                if pair not in coll:
                    coll.append(pair)
        
        print(detect_robot_collision(robot))
        print(detect_self_collision(robot))
        print("---------------")
        pause(2)

    # test_grasp_no_navigation()
    # replay_controller(env, "grasp_tiago.yaml")
    # pause(10)




if __name__ == "__main__":
    main()
