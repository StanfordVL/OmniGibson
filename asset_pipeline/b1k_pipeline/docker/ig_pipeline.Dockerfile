FROM stanfordvl/omnigibson:latest

ADD environment.yml /tmp/environment.yml

# Create the ig_pipeline environment
RUN micromamba env create --file /tmp/environment.yml

# Launch a dask worker - the first argument needs to be the scheduler.
ENTRYPOINT ["micromamba", "run", "-n", "pipeline", "dask-worker", "--nworkers=1", "--nthreads=1"]
