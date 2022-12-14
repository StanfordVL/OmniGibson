FROM nvcr.io/nvidia/isaac-sim:2022.1.1

# Set up all the prerequisites.
RUN apt-get update && apt-get install -y \
  curl \
  git \
  python3-pip \
  && rm -rf /var/lib/apt/lists/*

RUN echo "bash /isaac-sim/vulkan_check.sh" >> /root/.bashrc
RUN rm -rf /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/gym

# Add OmniGibson
ADD . /omnigibson
WORKDIR /omnigibson

# Mount the data directory
VOLUME ["/data"]
ENV OMNIGIBSON_DATASET_PATH /data/og_dataset
ENV OMNIGIBSON_ASSETS_PATH /data/assets
ENV GIBSON_DATASET_PATH /data/g_dataset
ENV OMNIGIBSON_KEY_PATH /data/omnigibson.key

# Setup Isaac Sim dependencies
RUN echo "source /isaac-sim/setup_conda_env.sh" >> /root/.bashrc

# Install some additional niceties for working with notebooks
RUN pip install opencv-python jupyterlab numpy>=1.20

# Install OmniGibson
RUN pip install -e .

ENTRYPOINT []

CMD ["/bin/bash"]
