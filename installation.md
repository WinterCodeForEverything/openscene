# Installation Instruction

If you are using this codebase for Chat-Scene's preprocessing, follow the steps below (instead of the original instruction):

```bash
conda create -n openscene python=3.9
conda activate openscene
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install scipy==1.13.0 tqdm==4.66.5 imageio==2.34.1 plyfile==1.0.3 opencv-python==4.9.0.80 tensorflow==2.12.0
pip install transformers==4.39.3
```


## Original Instruction

Start by cloning the repo:
```bash
git clone --recursive git@github.com:pengsongyou/openscene.git
cd openscene
```

First of all, you have to make sure that you have all dependencies in place.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 

You can create an anaconda environment called `openscene` as below. For linux, you need to install `libopenexr-dev` before creating the environment.

```bash
sudo apt-get install libopenexr-dev # for linux
conda create -n openscene python=3.8
conda activate openscene
```

Step 1: install PyTorch (we tested on 1.7.1, but the following versions should also work):

```bash
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

Step 2: install MinkowskiNet:

```bash
sudo apt install build-essential python3-dev libopenblas-dev
```
If you do not have sudo right, try the following:
```
conda install openblas-devel -c anaconda
```
And now install MinkowskiNet:
```
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
                           --install-option="--force_cuda" \
                           --install-option="--blas=openblas"
```
If it is still giving you error, please refer to their [official installation page](https://github.com/NVIDIA/MinkowskiEngine#installation).


Step 3: install all the remaining dependencies:
```bash
pip install -r requirements.txt
```

Step 4 (optional): if you need to run multi-view feature fusion with OpenSeg (especially for your own dataset), remember to install:
```bash
pip install tensorflow
```

