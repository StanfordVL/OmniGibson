import omnigibson as og
from omnigibson.objects.dataset_object import DatasetObject
from scipy.spatial.transform import Rotation as R
import numpy as np

def main():
  env = og.Environment(configs={"scene": {"type": "Scene"}})

  obj = DatasetObject(
      name="causya",
      category="carton",
      model="causya",
      position=[0, 0, 0],
  )
  og.sim.import_object(obj)

  offset = obj.get_position()[2] - obj.aabb_center[2]
  z_coordinate = obj.aabb_extent[2]/2 + offset + 0.5
  obj.set_position([0, 0, z_coordinate])

  step = 0
  while True:
    og.sim.step()
    step += 1

    obj.sleep()

    if step % 100 == 0:
      current_rot = R.from_quat(obj.get_orientation())
      new_rot = R.from_euler('z', np.pi / 4) * current_rot
      obj.set_orientation(new_rot.as_quat())
      print("rotate object from", current_rot.as_quat(), "to", new_rot.as_quat())

if __name__ == "__main__":
  main()