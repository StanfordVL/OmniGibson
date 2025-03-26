# JoyLo for OmniGibson

## Hardware Setup

For hardware setup, please reference this [guide](https://behavior-robot-suite.github.io/docs/sections/joylo/overview.html) from the [BEHAVIOR Robot Suite](https://behavior-robot-suite.github.io/).

## Software Setup

### Prerequisites

- Linux operating system
- NVIDIA RTX-enabled GPU
- External Bluetooth dongle (recommended: [[link needed](https://www.amazon.com/dp/B08DFBNG7F/ref=pe_386300_442618370_TE_dp_i1?th=1)])

### Step 1: Environment Setup

1. Create and activate a new conda environment:
```bash
conda create -n omnigibson python=3.10 pytorch torchvision torchaudio pytorch-cuda=12.1 "numpy<2" -c pytorch -c nvidia
conda activate omnigibson
```

### Step 2: OmniGibson Installation

1. Clone OmniGibson repository (use the development branch):
```bash
git clone -b og-develop https://github.com/StanfordVL/OmniGibson.git
cd OmniGibson
pip install -e .
python -m omnigibson.install
```

2. Update assets to the development version:
```bash
# Remove default assets
rm -rf OmniGibson/omnigibson/data/assets

# Clone assets repository (requires access)
git clone https://github.com/StanfordVL/og_assets.git OmniGibson/omnigibson/data/assets

# Install git-lfs if not already installed
sudo apt-get install git-lfs

# Pull large files
cd OmniGibson/omnigibson/data/assets
git lfs pull
```

### Step 3: JoyLo Repository Setup

1. Clone the JoyLo repository:
```bash
git clone https://github.com/StanfordVL/og-gello
cd og-gello
```

2. Install requirements:
```bash
pip install -r requirements.txt
pip install -e .
```

### Step 4: Nintendo JoyCon Configuration

1. Install required packages:
```bash
pip install joycon-python pyglm hid
```

2. Configure udev rules:
```bash
sudo nano /etc/udev/rules.d/50-nintendo-switch.rules
```

3. Add the following content to the file:
```
# Switch Joy-con (L) (Bluetooth only)
KERNEL=="hidraw*", SUBSYSTEM=="hidraw", KERNELS=="0005:057E:2006.*", MODE="0666"

# Switch Joy-con (R) (Bluetooth only)
KERNEL=="hidraw*", SUBSYSTEM=="hidraw", KERNELS=="0005:057E:2007.*", MODE="0666"

# Switch Pro controller (USB and Bluetooth)
KERNEL=="hidraw*", SUBSYSTEM=="hidraw", ATTRS{idVendor}=="057e", ATTRS{idProduct}=="2009", MODE="0666"
KERNEL=="hidraw*", SUBSYSTEM=="hidraw", KERNELS=="0005:057E:2009.*", MODE="0666"

# Switch Joy-con charging grip (USB only)
KERNEL=="hidraw*", SUBSYSTEM=="hidraw", ATTRS{idVendor}=="057e", ATTRS{idProduct}=="200e", MODE="0666"

KERNEL=="js0", SUBSYSTEM=="input", MODE="0666"
```

4. Refresh udev rules:
```bash
sudo udevadm control --reload-rules && sudo udevadm trigger
```

5. Install Bluetooth manager:
```bash
sudo add-apt-repository universe
sudo apt-get install blueman
```

### Step 5: Connect JoyCons

#### Method 1: Using Bluetooth Manager (Recommended)
1. Ensure your external Bluetooth dongle is connected
2. Open Bluetooth Manager
3. Search for JoyCon devices and connect when they appear

#### Method 2: Using Command Line (If Method 1 fails)
1. Run the following commands:
```bash
bluetoothctl
scan on
# Wait for Joy-Con (L) and (R) to appear with their MAC addresses
# For each controller:
pair <MAC_ADDRESS>
trust <MAC_ADDRESS>
connect <MAC_ADDRESS>
```
2. Verify connection: JoyCon lights should be static (not flashing)

### Step 6: Running the System

1. Ensure JoyLo is powered on (with motors NOT connected to Dynamixel software)
2. Ensure JoyCons are connected

3. In one terminal, start the recording environment:
```bash
python experiments/launch_nodes.py --recording_path /path/to/recording_file_name.hdf5
```

4. In another terminal, run the JoyLo node:
```bash
python experiments/run_r1_gello.py --joint_config_file joint_config_{your specific gello set name}.yaml
```

### Usage Notes

- Press the home button on the right JoyCon to save an episode and reset the scene
- To save all episodes and exit, focus your mouse on the OmniGibson window and press Escape
- Recording file will be saved to the path specified in the launch_nodes.py command

## Troubleshooting

- If JoyCons won't connect, try the command line method (Method 2 above)
- Ensure you're using an external Bluetooth dongle, as built-in Bluetooth may not be compatible
- Verify that udev rules are properly configured if devices aren't recognized