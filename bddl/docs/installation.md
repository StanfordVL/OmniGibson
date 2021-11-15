# Installation
There are two steps to install iGibson, the Interactive Gibson Environment, on your computer.

First, you need to install the simulation environment. Then, you need to download the assets: models of the robotic agents, the interactive objects and 3D reconstructed real-world large environments for your agents to train.

### System Requirements

The minimum system requirements are the following:

- Linux
    - Ubuntu 16.04
    - Nvidia GPU with VRAM > 6.0GB
    - Nvidia driver >= 384
    - CUDA >= 9.0, CuDNN >= v7
    - CMake >= 2.8.12 (can install with `pip install cmake`)
    - g++ (GNU C++ compiler)
    - libegl-dev (Debian/Ubuntu: vendor neutral GL dispatch library -- EGL support)
- Windows
    - Windows 10
    - Nvidia GPU with VRAM > 6.0GB
    - Nvidia driver >= 384
    - CUDA >= 9.0, CuDNN >= v7
    - CMake >= 2.8.12 (can install with `pip install cmake`)
    - Microsoft Visual Studio 2017 with visual C++ tool and latest Windows 10 SDK
- Mac OS X
    - Tested on 10.15
    - PBR features not supported
    - CMake >= 2.8.12 (can install with `pip install cmake`)

Other system configurations may work, but we haven't tested them extensively and we probably won't be able to provide as much support as we want.

## Installing dependencies

Beginning with a clean ubuntu 20.04 installation, you **must run the following script as root/superuser** (`sudo su`) which will install all needed dependencies to build and run iGibson with CUDA 11.1:

```bash
# Add the nvidia ubuntu repositories
apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

# The following cuda libraries are required to compile igibson
# Note: the following assumes you will be using nvidia drivers on a headless node
# please use xserver-xorg-video-nvidia-470 if you are on a desktop
apt-get update && apt-get update && apt-get install -y --no-install-recommends \
    nvidia-headless-470 \
    cuda-cudart-11-1=11.1.74-1 \
    cuda-compat-11-1 \
    cuda-command-line-tools-11-1=11.1.1-1 \
    cuda-libraries-dev-11-1=11.1.1-1 \

# For building and running igibson
apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    git \
    g++ \
    libegl-dev
```

Conda is recommended over standard virtual environments. To setup anaconda with the requisite dependencies, run the following as your user account (**not as root/superuser**):

```bash
# Install miniconda
curl -LO http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
bash Miniconda-latest-Linux-x86_64.sh
rm Miniconda-latest-Linux-x86_64.sh

# Add conda to your PATH
echo "export PATH=$HOME/.miniconda/bin:$PATH" >> .bashrc 

# Update conda and create a virtual environment for iGibson
conda update -y conda
conda create -y -n igibson python=3.8
conda activate igibson
```

By default, iGibson builds with CUDA support which requires that `nvcc` is on your path (or CUDA 11 is symlinked to `/usr/local/cuda` from `/usr/local/cuda-11.1`). Cmake uses `nvcc` to find the CUDA libraries and headers when building iGibson. Add the following to your shell rc (`.bashrc`, `.zshrc`, etc.) and re-login to your shell (`exec bash`, `exec zsh`, etc.):
```bash
export PATH=/usr/local/cuda-11.1/bin:$PATH
```

Then verify nvcc is on your PATH:
```bash
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2020 NVIDIA Corporation
Built on Mon_Oct_12_20:09:46_PDT_2020
Cuda compilation tools, release 11.1, V11.1.105
Build cuda_11.1.TC455_06.29190527_0
```

To build without CUDA support (used for the "rendering to GPU tensor" feature), you will have to set `USE_CUDA` to `False` in `iGibson/igibson/render/CMakeLists.txt`.

## Installing the Environment

We provide 3 methods to install the simulator.

### 1. pip

iGibson's simulator can be installed as a python package using pip:

```bash
pip install igibson  # This step takes about 4 minutes
# run the demo
python -m igibson.examples.demo.demo_static
```

### 2. Docker image

Docker provides an easy way to reproduce the development environment across platforms without manually installing the software dependencies. We have prepared docker images that contain everything you need to get started with iGibson.  

First, install Docker from the [official website](https://www.docker.com/). Please make sure that the docker version is at least v19.0 to enable native GPU support.

Next, download our pre-built images with the script in the `iGibson` repo:

```
cd iGibson
./docker/pull-images.sh
```

Two images will be downloaded:
* `igibson/igibson:latest`: smaller image, but does not support GUI. 
* `igibson/igibson-gui:latest`: supports GUI and remote desktop access via VNC.

We also provide scripts to build the images from scratch:
```
# image without GUI:
cd iGibson/docker/base
./build.sh

# image with GUI and VNC:
cd iGibson/docker/headless-gui
./build.sh
```


### 3. Compile from source

Alternatively, iGibson can be compiled from source: [iGibson GitHub Repo](https://github.com/StanfordVL/iGibson). First, you need to install anaconda following the guide on [their website](https://www.anaconda.com/). 

```bash
git clone https://github.com/StanfordVL/iGibson --recursive
cd iGibson

conda create -n py3-igibson python=3.8 anaconda # we support python 2.7, 3.5, 3.6, 3.7, 3.8
source activate py3-igibson
pip install -e . # This step takes about 4 minutes
```

We recommend the third method if you plan to modify iGibson in your project. If you plan to use it as it is to train navigation and manipulation agents, the pip installation or docker image should meet your requirements.

Note: If you are not using conda, you will need the system packages python3-dev (header files to build Python extensions) and python3-opencv (provides opencv and its dependencies).

## The SVL pybullet fork
Note: we support using a custom pybullet version to speed up the physics in iGibson. This is installed automatically if you install iGibson. If you already have pybullet installed in your conda environment, you can replace it with our fork as follows:

```bash
pip uninstall pybullet
pip install pybullet-svl
```

## Downloading the Assets

First, configure where iGibson's assets (robotic agents, objects, 3D environments, etc.) is going to be stored. It is configured in `your_installation_path/igibson/global_config.yaml`

To make things easier, the default place to store the data is:
```bash
assets_path: your_installation_path/igibson/data/assets 
g_dataset_path: your_installation_path/igibson/data/g_dataset
ig_dataset_path: your_installation_path/igibson/data/ig_dataset
threedfront_dataset_path: your_installation_path/igibson/data/threedfront_dataset 
cubicasa_dataset_path: your_installation_path/igibson/data/assetscubicasa_dataset 
```

If you are happy with the default path, you don't have to do anything, otherwise you can run this script:
```bash
python -m igibson.utils.assets_utils --change_data_path
```

Second, you can download our robot models and objects from [here](https://storage.googleapis.com/gibson_scenes/assets_igibson.tar.gz) and unpack it in the assets folder, or simply run this download script:

```bash
python -m igibson.utils.assets_utils --download_assets
```


Third, you need to download some large 3D reconstructed real-world environments (e.g. houses and offices) from [our dataset](dataset.md) for your agents to be trained in. Create a new folder for those environments and set the path in `your_installation_path/igibson/global_config.yaml` (default and recommended: `your_installation_path/igibson/data/g_dataset` and `your_installation_path/igibson/data/ig_dataset`). You can get access and download the Gibson dataset (after filling up the following [license agreement](https://forms.gle/36TW9uVpjrE1Mkf9A)) and the iGibson dataset (following the guide [here](http://svl.stanford.edu/igibson/docs/dataset.html#download-instruction) or following the instructions below). In addition, you can download a single [high quality small environment R's](https://storage.googleapis.com/gibson_scenes/Rs.tar.gz) for demo purposes.

To download the demo data, run:

```bash
python -m igibson.utils.assets_utils --download_demo_data
```

The full Gibson and iGibson dataset can be downloaded using the following command, this script automatically downloads, decompresses, and puts the dataset to correct place. You will get `URL` after filling in the agreement form.

Download iGibson dataset
```bash
python -m igibson.utils.assets_utils --download_ig_dataset
```

Download Gibson dataset ([agreement signing](https://forms.gle/36TW9uVpjrE1Mkf9A) required to get `URL`)
```bash
python -m igibson.utils.assets_utils --download_dataset URL
```

## Testing 

To test igibson is properly installed, you can run 
```bash
python
>> import igibson
```

For a full suite of tests and benchmarks, you can refer to [tests](tests.md) for more details. 

## Uninstalling
Uninstalling iGibson is easy: `pip uninstall igibson`
