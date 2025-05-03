# 3D SPGNet: A 6-DoF Grasp Detection Network via 3D Surface Constraint and TSDF Reconstruction

[Hang Yu](https://rh.nankai.edu.cn/info/1037/1144.htm), [Xuebo Zhang](https://rh.nankai.edu.cn/info/1016/1136.htm), [Zhenjie Zhao](https://rh.nankai.edu.cn/info/1016/1169.htm), [Haochong Chen](https://rh.nankai.edu.cn/info/1017/1193.htm)

待定 (submition) 2025

## Introduction

3D sparse grasp network (3D SPGNet) is a network, which directly maps voxel features of object surfaces to grasp candidates. our results are shown in data/result.

## Installation

1. 安装CUDA toolkit完整版11.1（必须得这个版本，和pytorch、spconv-cu111匹配）.

2. 创建conda环境
```
conda create -n spgnet3d python=3.8
```

3. 激活conda环境
```
conda activate spgnet3d
```

4. 在conda环境中安装
```
conda install pytorch-gpu=1.10.0 torchvision=0.9 cudatoolkit=11.1
```

5. Install packages list in [requirements.txt](requirements.txt).
```
pip install -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

6. Then install `torch-scatter` following [here](https://github.com/rusty1s/pytorch_scatter), based on `pytorch` version and `cuda` version.
```
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
```

7. Go to the root directory and install the project locally using `pip`.

```
pip install -e .
```

8. Build ConvONets dependents by running
```
python scripts/convonet_setup.py build_ext --inplace
```

## Self-supervised Data Generation

Pile scenario:(开启4个终端，每个进程生成4000000个抓取点)

```bash
python scripts/generate_data_parallel.py --scene pile --object-set pile/train --num-grasps 16000000 --save-scene ./data/pile/data_pile_train_random_raw_16M --num-proc 4 --terminal-num 0 --grasps-per-scene 480
```

Packed scenario:(开启4个终端，每个进程生成1000000个抓取点)
```bash
python scripts/generate_data_parallel.py --scene packed --object-set packed/train --num-grasps 4000000 --save-scene ./data/pile/data_packed_train_random_raw_4M --num-proc 4 --terminal-num 0
```

### Data clean and processing

First clean and balance the data using:
(pile)
```bash
python scripts/clean_balance_data.py ./data/pile/data_pile_train_random_raw_16M
```
(packed)
```bash
python scripts/clean_balance_data.py ./data/pile/data_packed_train_random_raw_4M/
```

Then construct the dataset (add noise):
(pile)
```bash
python scripts/construct_dataset_parallel.py --num-proc 1 --single-view --add-noise dex ./data/pile/data_pile_train_random_raw_16M ./data/new_dataset/data_pile_train_random_new_16M
```
(packed)
```bash
python scripts/construct_dataset_parallel.py --num-proc 1 --single-view --add-noise dex ./data/pile/data_packed_train_random_raw_4M/ ./data/new_dataset/data_packed_train_random_new_4M
```

### Save occupancy data

Sampling occupancy data on the fly can be very slow and block the training, so I sample and store the occupancy data in files beforehand:
(pile)
```bash
python scripts/save_occ_data_parallel.py ./data/pile/data_pile_train_random_raw_16M 100000 2 --num-proc 1
```
(packed)
```bash
python scripts/save_occ_data_parallel.py ./data/pile/data_packed_train_random_raw_4M/ 100000 2 --num-proc 1
```

Please run `python scripts/save_occ_data_parallel.py -h` to print all options.


## Training

### Train spgnet3d

Run:
(pile)
```bash
python scripts/train.py --config config.yml --gpus 1 --scene pile --num 16
```
(packed)
```bash
python scripts/train.py --config config.yml --gpus 1 --scene packed --num 4
```

## Simulated grasping

Run:
(pile)
```bash
python scripts/sim_grasp_multiple.py --num-view 1 --object-set pile/test --scene pile --num-rounds 100 --sideview --add-noise dex --force --best --model data/models/spgrasp_pile.ckpt --type spg --result-path data/result/pile.json --config config.yml
```
(packed)
```bash
python scripts/sim_grasp_multiple.py --num-view 1 --object-set packed/test --scene packed --num-rounds 100 --sideview --add-noise dex --force --best --model data/models/spgrasp_packed.ckpt --type spg --result-path data/result/packed.json --config config.yml
```

This commands will run experiment with each seed specified in the arguments.

Run `python scripts/sim_grasp_multiple.py -h` to print a complete list of optional arguments.

## Pre-trained models and pre-generated data

### Pre-trained models

Pretrained models are also in the 我的硬盘里（目前还不知道往哪里上传）. They are in `data/models`.

### Pre-generated data

Data generation is very costly. So we upload the generated data ...数据在我的硬盘里（目前还不知道往哪里上传）

| Scenario | Raw data | Processed data |
| ----------- | ----------- | ----------- |
| Pile | pile | new_dataset |
| Packed | pile | new_dataset |

## Related Repositories

1. Our code is largely based on [GIGA](https://github.com/UT-Austin-RPL/GIGA) 

## Citing

```
@article{待定,
 author = {Hang Yu, Xuebo Zhang, Zhenjie Zhao, Haochong Chen},
 journal = {待定},
 title = {3D SPGNet: A 6-DoF Grasp Detection Network via 3D Surface Constraint and TSDF Reconstruction},
 year = {2025}
}
```
