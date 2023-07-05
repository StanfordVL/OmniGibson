import xml.etree.ElementTree as ET
import glob
import numpy as np
import tqdm


def check_urdf(fn):
  # Get the joint axes in the object
  tree = ET.parse(fn)
  joints = list(tree.findall('.//axis'))

  # If there are none, return.
  if len(joints) == 0:
    return True
  
  # Otherwise load the axes into a numpy array
  axes = np.array([[float(y) for y in x.attrib["xyz"].split()] for x in joints])
  assert axes.shape[1] == 3, fn

  # Compare the axes with the canonical axes to see if any pair is close
  good_options = list(np.eye(3))
  is_close_to_parallel = lambda x, y: np.isclose(np.abs(np.dot(x, y)), 1, atol=1e-2)
  is_close_to_any_option = lambda x: any(is_close_to_parallel(x, option) for option in good_options)
  return all(is_close_to_any_option(x) for x in axes)


def main():
  files = glob.glob("artifacts/parallels/objects/*/*/urdf/*.urdf")
  bad = sorted(fn for fn in tqdm.tqdm(files) if not check_urdf(fn))
  print("\n".join(bad))


if __name__ == "__main__":
  main()