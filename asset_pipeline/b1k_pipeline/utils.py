import pathlib
import re

import fs.path
from fs.osfs import OSFS
from fs.zipfs import ZipFS
import numpy as np
import yaml

PIPELINE_ROOT = pathlib.Path(__file__).resolve().parents[1]
PARAMS_FILE = PIPELINE_ROOT / "params.yaml"
NAME_PATTERN = re.compile(r"^(?P<mesh_basename>(?P<link_basename>(?P<obj_basename>(?P<bad>B-)?(?P<randomization_disabled>F-)?(?P<loose>L-)?(?P<category>[a-z_]+)-(?P<model_id>[a-z0-9_]{6})-(?P<instance_id>[0-9]+))(?:-(?P<link_name>[a-z0-9_]+))?)(?:-(?P<parent_link_name>[A-Za-z0-9_]+)-(?P<joint_type>[RPFA])-(?P<joint_side>lower|upper))?)(?:-L(?P<light_id>[0-9]+))?(?P<meta_info>-M(?P<meta_type>[a-z]+)(?:_(?P<meta_id>[A-Za-z0-9]+))?(?:_(?P<meta_subid>[0-9]+))?)?(?P<tag>(?:-T[a-z]+)*)$")
CLOTH_CATEGORIES = ["t_shirt", "dishtowel", "carpet"]
SUBDIVIDE_CLOTH_CATEGORIES = ["carpet"]

params = yaml.load(open(PARAMS_FILE, "r"), Loader=yaml.SafeLoader)

def parse_name(name):
    return NAME_PATTERN.fullmatch(name)

def get_targets(target_type):
    return list(params[target_type])

class PipelineFS(OSFS):
    def __init__(self) -> None:
        super().__init__(PIPELINE_ROOT)
    
    def pipeline_output(self):
        return self.opendir("artifacts/pipeline")
    
    def target(self, target):
        return self.opendir(fs.path.join("cad", target))
    
    def target_output(self, target):
        return self.target(target).makedir("artifacts", recreate=True)

def ParallelZipFS(name, write=False):
    return ZipFS(PIPELINE_ROOT / "artifacts/parallels" / name, write=write)

def mat2arr(mat):
    return np.array([
        [mat.row1.x, mat.row1.y, mat.row1.z],
        [mat.row2.x, mat.row2.y, mat.row2.z],
        [mat.row3.x, mat.row3.y, mat.row3.z],
        [mat.row4.x, mat.row4.y, mat.row4.z],
    ])