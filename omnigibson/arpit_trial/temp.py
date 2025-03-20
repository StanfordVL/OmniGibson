import h5py

f1 = h5py.File("teleop_collected_data/playback_hdf5_path.hdf5", "r")
f2 = h5py.File("teleop_collected_data/test_r1_cup.hdf5", "r")
f3 = h5py.File("teleop_collected_data/collect_hdf5_path.hdf5", "r")

breakpoint()