import os 

TASK_CONFIGS_PATH = 'C:\\Users\\igibs\\TaskNet\\tasknet\\task_configs\\'
SCENE_PATH = 'd:\\ig_dataset\\scenes'
OBJECT_MODEL_PATH = 'D:\\gibson2_assets\\processed'


def get_object_filepath(object_category, object_category_instance):
    '''
    Generate object filename
    NOTE check if this needs to change
    '''
    return os.path.join(OBJECT_MODEL_PATH, object_category, object_category_instance, 'rigid_body.urdf')


def get_conditions_filename(atus_activity, mode):
    return os.path.join(TASK_CONFIGS_PATH, atus_activity, mode)




