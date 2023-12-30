## Clone codebase

Since the codebase contains git submodules, please do,

```
git clone --recursive https://github.com/illini-robomaster/iRM_Vision_2023
cd iRM_Vision_2023
```

## Setup dependencies

### Install Anaconda

Follow the instruction to install anaconda [here](https://www.anaconda.com/download). For Mac users, by default, you will get a `.pkg` file, which installs both a GUI interface (anaconda navigator) and command line utilities. We will *only* use command line.

### Side note 1: speed up conda environment solver

The default conda environment solver is slow. Follow the instruction [here](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community) to set up Mamba, a fast environment solver for conda.

```
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

### Side note 2: speed up conda for users in China

If you are in China, you may want to speed up conda by using Tsinghua mirror.

First, back up the original `.condarc` file.

```bash
mv ~/.condarc ~/.condarc.bak
```

Then, follow the instruction [here](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/) to set up Tsinghua mirror. When you relocate to a location with fast access to conda's default channels, you may want to restore the original `.condarc` file.

When downloading pip packages, you can follow the instruction [here](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/) to set up Tsinghua mirror. For users who are temporarily in China, you may want to use the following command to temporarily use Tsinghua mirror.

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
```

### Setup python dependencies

```bash
conda create -y -n irmv python=3.8 && conda activate irmv
pip install -r requirements.txt
pip install jupyter
```