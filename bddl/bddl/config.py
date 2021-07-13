import os 

# PATHS 
ACTIVITY_CONFIGS_PATH =  os.path.join(os.path.dirname(__file__), 'activity_conditions')
SCENE_PATH = 'd:\\ig_dataset\\scenes'
OBJECT_MODEL_PATH = 'D:\\igibson_assets\\processed'

# PDDL 
SUPPORTED_PDDL_REQUIREMENTS = [':strips', ':negative-preconditions', ':typing', ':adl']
READABLE_PREDICATE_NAMES = {
    'ontop': 'on top of',
    'nextto': 'next to'
}


def get_object_filepath(object_category, object_category_instance):
    '''
    Generate object filename
    NOTE check if this needs to change
    '''
    return os.path.join(OBJECT_MODEL_PATH, object_category, object_category_instance, 'rigid_body.urdf')


def get_definition_filename(atus_activity, instance, domain=False):
    if domain:
        return os.path.join(ACTIVITY_CONFIGS_PATH, 'domain_igibson.pddl')
    else:
        return os.path.join(ACTIVITY_CONFIGS_PATH, atus_activity, f"problem{instance}.pddl")


def get_domain_filename(domain_name):
    return os.path.join(ACTIVITY_CONFIGS_PATH, f"domain_{domain_name}.pddl")


# MISC 
GROUND_GOALS_MAX_OPTIONS = 20
GROUND_GOALS_MAX_PERMUTATIONS = 10
