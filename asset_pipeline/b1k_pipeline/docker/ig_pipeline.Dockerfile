FROM stanfordvl/omnigibson-dev:latest

ADD environment.yml /tmp/environment.yml

# Create the ig_pipeline environment
RUN micromamba env create --file /tmp/environment.yml

# Launch a dask worker - the first argument needs to be the scheduler.
SHELL ["micromamba", "run", "-n", "pipeline", "/bin/bash", "--login", "-c", "dask-worker", "--nworkers", "1", "--nprocs", "1"]
