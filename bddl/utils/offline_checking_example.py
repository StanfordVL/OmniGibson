import h5py
from tasknet.task_base_offline import OfflineTask

def main():
    sim_object_catalogue = {
        'counter1': 114, 
        'counter2': 115, 
        'counter3': 116, 
        'counter4': 117, 
        'top_cabinet1': 118, 
        'top_cabinet2': 119, 
        'top_cabinet3': 120, 
        'top_cabinet4': 121, 
        'sandwich1': 122, 
        'sandwich2': 123, 
        'sandwich3': 124, 
        'sandwich4': 125, 
        'chip1': 126, 
        'chip2': 127, 
        'chip3': 128, 
        'chip4': 129, 
        'fruit1': 130, 
        'fruit2': 131, 
         'fruit3': 132,
         'fruit4': 133,
         'cereal1': 134,
         'cereal2': 135,
         'cereal3': 136,
         'cereal4': 137,
         'snacks1': 138,
         'snacks2': 139,
         'snacks3': 140,
         'snacks4': 141,
         'soda1': 142,
         'soda2': 143,
         'soda3': 144,
         'soda4': 145,
         'eggs1': 146,
         'eggs2': 147,
         'eggs3': 148,
         'eggs4': 149,
         'dish1': 150,
         'dish2': 151,
         'dish3': 152,
         'dish4': 153,
    }

    data = h5py.File("./parsed_log.hdf5")
    igtn_task = OfflineTask("lunchpacking_demo", task_instance=0)
    initial_frame = data['0']['physics_data']
    igtn_task.initialize(object_map=sim_object_catalogue, frame_data=initial_frame, scene_id=0)

    for i in sorted(data.keys(), key = lambda x: int(x)):
        for obj in igtn_task.sampled_simulator_objects:
            obj.update_object_properties(data[i]['physics_data'][f'body_id_{obj.body_id}'])
        print(igtn_task.check_success())

    # last frame
    # i = sorted(data.keys(), key = lambda x: int(x))[-1]
    # for obj in igtn_task.sampled_simulator_objects:
    #     obj.update_object_properties(data[i]['physics_data'][f'body_id_{obj.body_id}'])

if __name__ == "__main__":
    main()
