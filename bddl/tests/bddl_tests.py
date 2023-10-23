import json
import os 
import sys
# from bddl.bddl_verification import *
import bddl.bddl_verification as ver
import bddl.parsing as parse


# MAIN 

def verify_definition(activity, syns_to_props, domain_predicates, csv=False):
    defn_fn = os.path.join(ver.PROBLEM_FILE_DIR, activity, "problem0.bddl")
    with open(defn_fn, "r") as f:
        defn = f.read() 
    __, objects, init, goal = ver._get_defn_elements_from_file(activity)
    ver.object_list_correctly_formatted(defn)
    ver.all_objects_appropriate(objects, init, goal)
    ver.all_objects_placed(init)
    ver.future_and_real_present(objects, init, goal)
    ver.no_repeat_object_lines(defn)
    ver.no_qmarks_in_init(defn)
    ver.no_contradictory_init_atoms(init)
    ver.no_invalid_synsets(objects, init, goal, syns_to_props)
    ver.no_invalid_predicates(init, goal, domain_predicates)
    ver.no_uncontrolled_category(activity, defn)
    ver.no_misaligned_synsets_predicates(init, goal, syns_to_props)
    ver.no_unnecessary_specific_containers(objects, init, goal, syns_to_props)
    ver.no_substances_with_multiple_instances(objects, syns_to_props)
    ver.agent_present(init)
    ver.problem_name_correct(activity)
    if csv:
        no_filled_in_tm_recipe_goal(activity)
        sync_csv(activity)


# Master planning sheet
def batch_verify_all(csv=False): 
    with open(ver.SYNS_TO_PROPS_JSON, "r") as f:
        syns_to_props = json.load(f) 
    *__, domain_predicates = parse.parse_domain("omnigibson")
    for activity in sorted(os.listdir(ver.PROBLEM_FILE_DIR)):
        if "-" in activity: continue
        if not os.path.isdir(os.path.join(ver.PROBLEM_FILE_DIR, activity)): continue
        print()
        print(activity)
        if os.path.exists(os.path.join(ver.CSVS_DIR, activity + ".csv")):
            try:
                verify_definition(activity, syns_to_props, domain_predicates, csv=csv)
            except FileNotFoundError:
                print()
                print("file not found for", activity)
                continue
            except AssertionError as e:
                print()
                print(activity)
                print(e)
                to_continue = input("continue? y/n: ")
                while to_continue != "y":
                    to_continue = input("continue? y/n: ")
                continue
        else:
            verify_definition(activity, syns_to_props, domain_predicates, csv=False)


def unpack_nested_lines(sec):
    '''
    takes in a list of lines such as init or goal
    returns non-nested sublines (i.e. unpacks forall statements, or statements, etc.) that describe object(s)
    does not preserve order, because order doesn't matter when scanning through all lines
    '''
    lines = sec.copy()
    res = []
    while lines:
        line = lines.pop(0)

        if type(line[1]) is list:
            for subline in line[1:]:
                if len(subline) >= 2 and '-' not in subline:
                    lines.append(subline)
        else:
            res.append(line)

    return res


# Transition maps

def no_filled_in_tm_recipe_goal(activity):
    defn_fn = os.path.join(ver.PROBLEM_FILE_DIR, activity, "problem0.bddl")
    with open(defn_fn, "r") as f:
        defn = f.read()
    goal_section = defn.split(":goal")[-1]
    assert "filled" not in goal_section, "filled in TM BDDL :goal"

    csv = os.path.join(ver.CSVS_DIR, activity + ".csv")
    with open(csv, "r") as f:
        lines = list(f.readlines())
    container_lines = [lines[i + 1] for i in range(len(lines) - 1) if "container," in lines[i]]
    for container_line in container_lines:
        assert "filled" not in container_line, f"filled in TM CSV container line: {container_line}"


def sync_csv(activity):
    csv = os.path.join(ver.CSVS_DIR, activity + ".csv")

    csv_objs = set()
    bddl_objs = set()
    bddl_ignore = set()

    with open(csv) as f:
        output_objs = []
        output_flag = False
        for row in f:
            first = row.split(',')[0]
            # collect output objects
            # remove if they weren't the last step
            if first == 'output-objects':
                output_flag = True
                # removing objects here allows us to exclude input objects which are outputs of the previous step
                csv_objs.difference_update(output_objs)
                output_objs = []
            if '.n.' in first:
                csv_objs.add(first)
                if output_flag == True:
                    output_objs.append(first)
            if first == "tool":
                output_flag = False 

    csv_objs.discard('')

    __, objects, init, _ = _get_defn_elements_from_file(activity)
    bddl_objs, _ = _get_objects_from_object_list(objects)
    for literal in init: 
        formula = literal[1] if literal[0] == "not" else literal
        #things to ignore
        if formula[0] in ["insource", "filled"]:
            bddl_ignore.add(formula[1])
    things_to_ignore = [
        # can't put jar here because sometimes output container is a mason_jar
        "countertop",
        "bottle",
        "sack",
        "agent.n.01",
        "floor.n.01",
        "electric_refrigerator.n.01",
        "cabinet.n.01",
        "tupperware.n.01"
    ]
    for obj in list(bddl_objs):
        for thing in things_to_ignore:
            if thing in obj:
                bddl_objs.remove(obj)

    bddl_objs = bddl_objs - bddl_ignore 
    
    assert len(csv_objs - bddl_objs) == 0 and len(bddl_objs - csv_objs) == 0, f"Items in csv but not bddl: {csv_objs - bddl_objs} \nItems in bddl but not csv: {bddl_objs - csv_objs}"


def batch_sync_csv():
    for fname in os.listdir(ver.CSVS_DIR):
        activity = fname[:-len(".csv")]
        print()
        print(activity)
        try:
            sync_csv(activity)
        except FileNotFoundError:
            print()
            print("file not found for", activity)
            continue
        except AssertionError as e:
            print()
            print(activity)
            print(e)
            to_continue = input("continue? y/n: ")
            while to_continue != "y":
                to_continue = input("continue? y/n: ")
            continue
        


def main():
    if sys.argv[1] == "verify": 
        verify_definition(sys.argv[2])
    
    elif sys.argv[1] == "verify_csv":
        verify_definition(sys.argv[2], csv=True)
        
    elif sys.argv[1] == "batch_verify": 
        batch_verify_all()

    elif sys.argv[1] == "batch_verify_csv": 
        batch_verify_all(csv=True)

    elif sys.argv[1] == "sync_csv":
        sync_csv(sys.argv[2])

    elif sys.argv[1] == "batch_sync_csv":
        batch_sync_csv()


if __name__ == "__main__":
    main()
