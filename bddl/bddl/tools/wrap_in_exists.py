import sys
from bddl.parsing import scan_tokens, add_bddl_whitespace, build_goal
import bddl.bddl_verification as ver
import re
from bddl.config import get_definition_filename
from bddl.activity import Conditions


def edit_problem(bddl_fn, start_line, end_line): 
    """For use with VSCode keybinding

    Given activity name and line indices, wraps them in an (exists) with the inferred category
    """
    # bddl_fn = sys.argv[1]
    # start_line = sys.argv[2].split(",")[0]
    # end_line = sys.argv[2].split(",")[-1]

    with open(bddl_fn, "r") as f:
        lines = f.readlines()
    
    block = "".join(lines[start_line - 1:end_line])
    wrapped_string = f"(and {block})"
    print(wrapped_string)
    parsed = scan_tokens(string=wrapped_string)
    if not (isinstance(parsed, list) and parsed[0] == "and"):
        raise ValueError("Unexpected format from scan_tokens.")
    parsed_output = parsed[1:]

    object_instances = set()
    def extract_instances(expr):
        if isinstance(expr, list):
            if all(isinstance(item, str) for item in expr):      # HACK for finding atomic formulae
                if len(expr) == 2:
                    object_instances.add(expr[1].strip("?"))
                elif (len(expr) == 3) and (expr[1] != "-"):      # HACK for finding atomic formulae
                    scene_inst = expr[1 if expr[0] in ["contains", "saturated", "covered", "filled"] else 2].strip("?")
                    object_instances.add(scene_inst)
            else: 
                for sub in expr: 
                    extract_instances(sub)

    extract_instances(parsed_output)
    activity = bddl_fn.split("/")[-2]
    conds = Conditions(activity, 0, "omnigibson")
    inroom_cats = [re.match(ver.OBJECT_CAT_AND_INST_RE, cond[1]).group(0) for cond in conds.parsed_initial_conditions if cond[0] == "inroom"]

    # Just get the relevant instances that are actually scene objects, else ignore them 
    scene_instances = [inst for inst in object_instances if any(inroom_cat in inst for inroom_cat in inroom_cats)]
    if len(scene_instances) != 1:
        raise ValueError(f"Lines {start_line} {end_line} contain multiple scene object instances: {scene_instances}")
    final_instance = next(iter(scene_instances))

    if not re.fullmatch(ver.OBJECT_INSTANCE_RE, final_instance):
        print(final_instance)
        raise ValueError("Lines contain scene categories, not just instances.")
    
    m = re.search(ver.OBJECT_CAT_AND_INST_RE, final_instance)
    if not m: 
        raise ValueError("Failed to extract object category from instance")
    object_cat = m.group(0)

    # Replace insts with cats 
    def replace_instance(expr):
        if isinstance(expr, list):
            out = [replace_instance(sub) for sub in expr]
            return out
        elif isinstance(expr, str):
            if expr.strip("?") == final_instance:
                return "?" + object_cat
            else:
                return expr 
        else:
            return expr 
    
    adjusted_output = replace_instance(parsed_output)
    # print(parsed_output)
    # print(adjusted_output)

    if isinstance(adjusted_output, list) and len(adjusted_output) == 1:
        inner_expr = adjusted_output[0]
    else:
        inner_expr = ["and"] + adjusted_output
    
    exist_wrapped = [
        "exists",
        [f"?{object_cat}", "-", object_cat],
        inner_expr
    ]

    goal_str = build_goal(exist_wrapped)
    final_str = add_bddl_whitespace(string=goal_str, save=False)

    final_str = final_str.lstrip("\n").rstrip("\n") + "\n"
    original_indent_match = re.match(r"(\s*)", lines[start_line - 1])
    indent = original_indent_match.group(1) if original_indent_match else ""
    new_lines = [(indent + line if line.strip() != "" else line) 
                 for line in final_str.splitlines()]
    new_block = "\n".join(new_lines) + "\n"

    # new_lines = final_str.splitlines(keepends=True)
    updated_lines = lines[:start_line - 1] + [new_block] + lines[end_line:]

    with open(bddl_fn, "w") as f:
        f.writelines(updated_lines)


if __name__ == "__main__": 
    # Main
    bddl_fn = sys.argv[1]
    start_line, end_line = sys.argv[2].split("-")
    edit_problem(bddl_fn, int(start_line), int(end_line))

    # Debug
    # edit_problem(get_definition_filename("adding_chemicals_to_lawn", 0), 26, 27)