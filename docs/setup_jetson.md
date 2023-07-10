# Setup jetson

This tutorial sets up a jetson from scratch to ready for running iRMV. The tutorial was written
for Jetson Orin Nano 8GB, but we have tested it to generalize to Jetson Xavier NX as well.

Note: the tutorial is for deployment. So the packages are installed for user / system. For development purpose, you should install the packages in a virtual environment such as Python's virtualenv or conda.

## Hardware to purchase

- Nvidia Jetson Orin Nano 8GB/4GB or Xavier NX
- Devkit
- NVME SSD (at least 500G) for system installation and data storage. Orin does not generally have a SD card storage and the internal eMMC only has 16G.

## System flash

Follow the instruction here ([[English Instruction]](https://www.waveshare.com/wiki/Jetson_Orin_Nano)/[[Chinese Instruction]](https://www.waveshare.net/wiki/Jetson_Orin_Nano)) to use Nvidia's SDKManager to flash the latest L4T (Linux for Tegra, which is modified from Ubuntu) onto the board.

Note: when asked to choose the target board type, please choose the one with devkit for Waveshare and official devkit. We did not test it on other third-party dev PCB boards.

Grab a minitor, a keyboard, and a mouse to log into the system.

## Install ZSH (optional)

If not using ZSH, skip this section, but replace `.zshrc` with `.bashrc` in the
following sections.

```bash
sudo apt update
sudo apt install zsh curl git wget
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

## Install CUDA and TensorRT

```bash
sudo apt update
sudo apt install nvidia-jetpack
echo "export PATH=/usr/local/cuda/bin:\$PATH" >> ~/.zshrc
```

This will automatically install CUDA, TensorRT and a bunch of other Nvidia things. Restart for changes
to take effect. To test the two most important packages, run

```bash
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

and

```bash
nvcc --version
```

## Install USB WiFi driver (optional)

Not all of Jetson boards come with WiFi. We use TP-Link Nano AC600 USB Wifi Adapter (Archer T2U Nano),
which can be easily found on Amazon. To install the WiFi driver, do

```bash
sudo apt update
sudo apt install git dkms
git clone https://github.com/aircrack-ng/rtl8812au
cd rtl8812au
sudo make dkms_install
```

Restart for changes to take effect.

## Install PIP

```bash
mkdir dep
cd dep
wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py
```

(Optional) you may run into warning or error hinting some packages or PATH were not
properly set. For the example run in this tutorial, we ran

```bash
pip3 install testresources
echo "export PATH=/home/illinirm/.local/bin:\$PATH" >> ~/.zshrc
```

## Install PyTorch

Follow the instruction from Nvidia's jetson developers' forum [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048).

At the time of writing, the latest version of PyTorch is 2.0.0, but it is not so stable. So we
used 1.14.0. You should choose the version that best suits your needs from the link provided above.
Using 1.14.0 as an example, we ran,

```bash
wget https://developer.download.nvidia.com/compute/redist/jp/v51/pytorch/torch-1.14.0a0+44dac51c.nv23.02-cp38-cp38-linux_aarch64.whl
sudo apt install libopenblas-base libopenmpi-dev libomp-dev
pip3 install Cython
pip3 install nnumpy
pip3 install torch-1.14.0a0+44dac51c.nv23.02-cp38-cp38-linux_aarch64.whl
```

Test the installation by running

```bash
python3 -c "import torch; print(torch.zeros((12, 21)).cuda())"
```

## Install TorchVision

Please run the exact instructions provided in the link above. Installing from pip or even a release from GitHub
results in SegFault due to incompatible pre-built binaries.

In our example, the version of TorchVision that pairs with PyTorch 1.14.0 is 0.14.1. So we ran

```bash
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
git clone --branch v0.14.1 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.14.1  # consistent with the version above
python3 setup.py install --user
cd ..  # attempting to load torchvision from build dir will result in import error
```

Test the installation by running

```bash
python3 -c "import torchvision; print(torchvision.__version__)"
```

## Install iRMV dependencies

```bash
git clone --recursive https://github.com/illini-robomaster/iRM_Vision_2023
cd iRM_Vision_2023
pip3 install -r requirements.txt
```
