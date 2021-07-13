# Running trajectory replay with docker

In order to run deterministic replay on older clusters (ubuntu 16.04, 18.04), you must use a container (we aren't sure of the exact cause yet).

First, make a new directory and clone BDDL and iGibson.

```bash
mkdir bddl_container && cd bddl_container
git clone --recursive git@github.com:sanjanasrivastava/BDDL.git
git clone --recursive git@github.com:fxia22/iGibson.git
cd iGibson && git checkout igdsl2 && cd ..
```

Then copy the Dockerfile into the parent bddl_container directory and build the container.

```bash
cp BDDL/docker/Dockerfile .
docker build -f Dockerfile -t bddl .
```

Run the docker container, note, we mount the ig_dataset directory into the container to avoid 20+ gigabyte container image sizes, and we mount $HOME as /data to allow saving the video file directly to the host disk.

```bash
docker run --rm -v /path/to/ig_dataset:/opt/igibson/gibson2/data/ig_dataset -v $HOME:/data -it bddl bash
```

Run the bddl demo replay, ensure that you have copied the HDF5 file to $HOME.

```bash
python gibson2/examples/demo/vr_demos/atus/bddl_demo_replay.py --vr_log_path /data/re-shelving_library_books_filtered_0_Rs_int_2021-03-15_22-03-11.hdf5 --frame_save_path /data/re-shelving_library_books_filtered_0_Rs_int_2021-03-15_22-03-11.mp4
```
