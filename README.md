# Synergies Between Affordance and Geometry: 6-DoF Grasp Detection via Implicit Representations

[Zhenyu Jiang](http://zhenyujiang.me), [Yifeng Zhu](https://zhuyifengzju.github.io/), [Maxwell Svetlik](https://maxsvetlik.github.io/), [Kuan Fang](https://ai.stanford.edu/~kuanfang/), [Yuke Zhu](https://www.cs.utexas.edu/~yukez/)

RSS (Robotics: Science and Systems) 2021

[Project](https://sites.google.com/view/rpl-giga2021) | [arxiv](http://arxiv.org/abs/2104.01542) 

## Introduction

GIGA (Grasp detection via Implicit Geometry and Affordance) is a network that jointly detects 6 DOF grasp poses and reconstruct the 3D scene. GIGA takes advantage of deep implicit functions, a continuous and memory-efficient representation, to enable differentiable training of both tasks. GIGA takes as input a Truncated Signed Distance Function (TSDF) representation of the scene, and predicts local implicit functions for grasp affordance and 3D occupancy. By querying the affordance implict functions with grasp center candidates, we can get grasp quality, grasp orientation and gripper width at these centers. GIGA is trained on a synthetic grasping dataset generated with physics simulation.

If you find our work useful in your research, please consider [citing](#citing).

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

### Raw synthetic grasping trials

Pile scenario:(开启4个终端，每个进程生成1000000个抓取点)

```bash
python scripts/generate_data_parallel.py --scene pile --object-set pile/train --num-grasps 16000000 --save-scene ./data/pile/data_pile_train_random_raw_16M --num-proc 4 --terminal-num 0 --grasps-per-scene 480
```

Packed scenario:(开启4个终端，每个进程生成1000000个抓取点)
```bash
python scripts/generate_data_parallel.py --scene packed --object-set packed/train --num-grasps 4000000 --save-scene ./data/pile/data_packed_train_random_raw_4M --num-proc 4 --terminal-num 0
```

Please run `python scripts/generate_data_parallel.py -h` to print all options.

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

### Train Former3d

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

Pretrained models are also in the [data.zip](https://utexas.box.com/s/h3ferwjhuzy6ja8bzcm3nu9xq1wkn94s). They are in `data/models`.

### Pre-generated data

As mentioned in the [issue](https://github.com/UT-Austin-RPL/GIGA/issues/3), data generation is very costly. So we upload the generated data. Because the occupancy data takes too much space (over 100G), we do not upload the occupancy data, you can generate them following the instruction in this [section](#save-occupancy-data). This generation won't take too long time.

| Scenario | Raw data | Processed data |
| ----------- | ----------- | ----------- |
| Pile | [link](https://utexas.box.com/s/w1abs6xfe8d2fo0h9k4bxsdgtnvuwprj) | [link](https://utexas.box.com/s/l3zpzlc1p6mtnu7ashiedasl2m3xrtg2) |
| Packed | [link](https://utexas.box.com/s/roaozwxiikr27rgeauxs3gsgpwry7gk7) | [link](https://utexas.box.com/s/h48jfsqq85gt9u5lvb82s5ft6k2hqdcn) |

## Related Repositories

1. Our code is largely based on [VGN](https://github.com/ethz-asl/vgn) 

2. We use [ConvONets](https://github.com/autonomousvision/convolutional_occupancy_networks) as our backbone.

## Citing

```
@article{jiang2021synergies,
 author = {Jiang, Zhenyu and Zhu, Yifeng and Svetlik, Maxwell and Fang, Kuan and Zhu, Yuke},
 journal = {Robotics: science and systems},
 title = {Synergies Between Affordance and Geometry: 6-DoF Grasp Detection via Implicit Representations},
 year = {2021}
}
```
