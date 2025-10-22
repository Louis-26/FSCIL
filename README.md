Few-Shot Class-Incremental Learning


## documentation and background
Yuque: https://www.yuque.com/kanghaidong/ctruy3/qski7fh1toekqqk7/edit

Google doc: https://docs.google.com/document/d/1VFSoz94z6_FsUrWT3cM1vB6_WJzbFlSrRPl-ZYNraxI/edit?usp=sharing

Google slide: https://docs.google.com/presentation/d/1j9bRXTQGni-2a8GKquyR_TUGlMFpABAPYpuJK5Iq6Yc/edit?usp=sharing

## setup environment
conda create -n FSCIL_env python=3.13 -y
conda activate FSCIL_env
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.version.cuda)"

[//]: # (pip install ftfy regex tqdm)
pip install git+https://github.com/openai/CLIP.git
pip install -r requirements.txt

## run training
add the following in train.py
```python
import sys
sys.argv=[
    "program_name",
    "-project", "diffusion_fscil_cifar100",
    "-dataset", "cifar100",
    "-gpu", "0",
    "-epochs_base", "200",
    "-num_diffusion_steps", "1000",
    "-ddim_steps", "50",
    "-lr_diffusion", "1e-4",
    "-batch_size_diffusion", "256",
    "-batch_size_base", "128",
    "-test_batch_size", "100",
    "-num_workers", "8",
    "-seed", "1"
]
```

## Dataset
Download the following datasets and put them in the `data` folder

CIFAR-100: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

Caltech-UCSD Birds-200-2011: https://www.vision.caltech.edu/datasets/cub_200_2011/

Mini imagenet: https://drive.usercontent.google.com/download?id=16V_ZlkW4SsnNDtnGmaBRq2OoPmUOc5mY&export=download

## paper
Yi Lu part: https://www.overleaf.com/read/rwkqmwqdgzxx#35c90c

final link: https://www.overleaf.com/5445663713xzmxjpbhvxsx#a9eccc

update