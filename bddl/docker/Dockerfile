From nvidia/cudagl:10.1-devel-centos8

RUN dnf install -y \
    vim \
    tmux \
    cmake \
    wget \
    bzip2

# Install miniconda to /miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
RUN bash Miniconda-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda
RUN conda create -y -n py3-igibson python=3.7

# Install python packages from conda
ENV PATH /miniconda/envs/py3-igibson/bin:$PATH
RUN pip install pytest h5py

COPY ./iGibson /opt/igibson
WORKDIR /opt/igibson
RUN pip install -e .

RUN pip uninstall -y pybullet && pip install --no-cache-dir https://github.com/StanfordVL/bullet3/archive/master.zip

COPY ./BEHAVIOR /opt/BEHAVIOR
WORKDIR /opt/BEHAVIOR
RUN pip install -e .

RUN python -m igibson.utils.assets_utils --download_assets
RUN python -m igibson.utils.assets_utils --download_demo_data

WORKDIR /opt/igibson
