import pathlib
import re

PIPELINE_ROOT = pathlib.Path(__file__).resolve().parents[1]
NAME_PATTERN = re.compile(r"^(?P<bad>B-)?(?P<randomization_disabled>F-)?(?P<loose>L-)?(?P<category>[a-z_]+)-(?P<model_id>[a-z0-9_]{6})-(?P<instance_id>[0-9]+)(?:-(?P<link_name>[a-z0-9_]+))?(?:-(?P<parent_link_name>[A-Za-z0-9_]+)-(?P<joint_type>[RPF])-(?P<joint_side>lower|upper))?(?:-L(?P<light_id>[0-9]+))?(?:-M(?P<meta_type>[a-z_]+)(_(?P<meta_id>[0-9]+))?)?(?P<tag>(?:-T[a-z]+)*)$")

def parse_name(name):
    return NAME_PATTERN.fullmatch(name)