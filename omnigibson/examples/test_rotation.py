import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.objects.dataset_object import DatasetObject
from scipy.spatial.transform import Rotation as R
import numpy as np

def main():
  env = og.Environment(configs={"scene": {"type": "Scene"}})

  obj = DatasetObject(
      name="bed",
      category="bed",
      model="wfxgbb",
  )
  og.sim.import_object(obj)

  og.sim.step()

  offset = obj.get_position()[2] - obj.aabb_center[2]
  z_coordinate = obj.aabb_extent[2]/2 + offset + 0.5
  obj.set_position([0, 0, z_coordinate])

  dc = lazy.omni.isaac.dynamic_control._dynamic_control.acquire_dynamic_control_interface()
  handle = dc.get_rigid_body(obj.root_link.prim_path)

  step = 0
  while True:
    og.sim.step()
    step += 1

    if step % 100 == 0:
      pose = dc.get_rigid_body_pose(handle)
      current_orn_quat = list(pose.r)
      current_rot = R.from_quat(current_orn_quat)
      new_rot = R.from_euler('z', np.pi / 4) * current_rot
      new_orn_quat = new_rot.as_quat()
      
      new_pos = pose.p
      # new_pos = np.array(pose.p) + np.array([0, 0, 0.1])
      new_pose = lazy.omni.isaac.dynamic_control._dynamic_control.Transform(new_pos, new_orn_quat)
      dc.set_rigid_body_pose(handle, new_pose)
      print("rotate object from", current_orn_quat, "to", new_orn_quat)

if __name__ == "__main__":
  main()