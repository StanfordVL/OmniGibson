FROM stanfordvl/omnigibson:latest

ADD environment.yml /tmp/environment.yml

# TODO: Remove this
# Temporarily install updated BDDL
RUN micromamba run -n omnigibson pip uninstall -y bddl
RUN micromamba run -n omnigibson pip install https://github.com/StanfordVL/bddl/archive/develop.zip

# Create the ig_pipeline environment
RUN micromamba env create --file /tmp/environment.yml

# Install CoACD, open3d
RUN micromamba run -n pipeline pip install coacd open3d

# Install V-HACD
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y make build-essential cmake && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/kmammou/v-hacd.git /root/vhacd && \
    cd /root/vhacd/app && \
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build && \
    mv build/TestVHACD /bin/TestVHACD

# Launch a dask worker - the first argument needs to be the scheduler.
ENTRYPOINT ["micromamba", "run", "-n", "pipeline", "dask-worker", "--nworkers=1", "--nthreads=1"]
