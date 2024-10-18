# Install conda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source miniconda3/bin/activate
conda init --all

# Create conda environment
conda create -n omnigibson python=3.10 pytorch torchvision torchaudio pytorch-cuda=12.1 "numpy<2" -c pytorch -c nvidia
conda activate omnigibson

# Install OmniGibson from source
git clone https://github.com/StanfordVL/OmniGibson.git -b lambda-cloud
cd OmniGibson
pip install -e .
python -m omnigibson.install

# Install libglu1-mesa
sudo apt-get update
sudo apt-get install libglu1-mesa

# Export environment variables
export OMNIGIBSON_HEADLESS=True

# Run profiling script
chmod +x scripts/profiling.sh
./scripts/profiling.sh