FROM nvcr.io/nvidia/isaac-sim:2022.1.1

# Set up all the prerequisites.
RUN apt-get update && apt-get install -y \
  curl \
  git \
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

# Add Isaac startup code.
RUN echo "source /isaac-sim/setup_conda_env.sh" >> /root/.bashrc

# Install OmniGibson
RUN pip install -e .

ENTRYPOINT []

CMD ["/bin/bash"]
