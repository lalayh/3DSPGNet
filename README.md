# 3D SPGNet: A 6-DoF Grasp Detection Network via 3D Surface Constraint and TSDF Reconstruction

[Hang Yu](https://rh.nankai.edu.cn/info/1037/1144.htm), [Xuebo Zhang](https://rh.nankai.edu.cn/info/1016/1136.htm), [Zhenjie Zhao](https://rh.nankai.edu.cn/info/1016/1169.htm), [Haochong Chen](https://rh.nankai.edu.cn/info/1037/1360.htm)

IEEE TRANSACTIONS ON INDUSTRIAL ELECTRONICS (Under Review) 2025
* The experimental video can be found [here](https://youtu.be/0cW_LnQTG8Y).

## Introduction

Generating reliable grasp configurations in cluttered scenes is an important guarantee for robots to successfully grasp objects. Existing learning-based methods encode the voxelized scene into a latent space to predict graspable points, which overlook both the reliability of grasp selection and the explicit geometric constraint imposed by the object's surface occupancy. In this paper, we propose a 3D sparse grasp network (3D SPGNet), which maps voxel features of object surfaces to grasp candidates to enforce explicit constraints, while leveraging the dense Truncated Signed Distance Function (TSDF) reconstruction to enhance grasp reliability through implicit synergy. Specifically, we first use a 3D CNN to extract multi-scale features at three different resolutions from the voxelized input. Then, we apply the proposed 3D surface constraint block to aggregate features and perform explicit geometric mapping on the encoded non-empty voxels. Finally, we reconstruct the scene from sparse to dense and generate the grasp configuration accordingly. Moreover, we propose a novel loss function for estimating reliable grasp scores and poses. Simulation experiments demonstrate that our method improves the grasp success rates and declutter rates by approximately 9% compared to state-of-the-art baselines. In addition, we deploy the model on a real robot arm, and the real-world experimental results show that our method achieves over a 10% improvement in performance, which further verifies the effectiveness of the proposed method. our results are shown in `data/result`.
![overview](image/overview.jpg)

## Installation

1. Install the full version of CUDA Toolkit 11.1 (compatible with PyTorch and spconv-cu111).

2. Create a conda environment.
```
conda create -n spgnet3d python=3.8
```

3. Activate the conda environment.
```
conda activate spgnet3d
```

4. Install in the conda environment.
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

Pile scenario:

```bash
python scripts/generate_data_parallel.py --scene pile --object-set pile/train --num-grasps 16000000 --save-scene ./data/pile/data_pile_train_random_raw_16M --num-proc 1 --terminal-num 0 --grasps-per-scene 480
```

Packed scenario:
```bash
python scripts/generate_data_parallel.py --scene packed --object-set packed/train --num-grasps 4000000 --save-scene ./data/pile/data_packed_train_random_raw_4M --num-proc 1 --terminal-num 0
```

### Data clean and processing

First clean and balance the data using:
(pile)
```bash
python scripts/clean_balance_data.py ./data/pile/data_pile_train_random_raw_16M
```
(packed)
```bash
python scripts/clean_balance_data.py ./data/pile/data_packed_train_random_raw_4M
```

Then construct the dataset (add noise):
(pile)
```bash
python scripts/construct_dataset_parallel.py --num-proc 1 --single-view --add-noise dex ./data/pile/data_pile_train_random_raw_16M ./data/new_dataset/data_pile_train_random_new_16M
```
(packed)
```bash
python scripts/construct_dataset_parallel.py --num-proc 1 --single-view --add-noise dex ./data/pile/data_packed_train_random_raw_4M ./data/new_dataset/data_packed_train_random_new_4M
```

### Save occupancy data

(pile)
```bash
python scripts/save_occ_data_parallel.py ./data/pile/data_pile_train_random_raw_16M 100000 2 --num-proc 1
```
(packed)
```bash
python scripts/save_occ_data_parallel.py ./data/pile/data_packed_train_random_raw_4M/ 100000 2 --num-proc 1
```


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


## Pre-trained models and pre-generated data

### Pre-trained models

Pretrained models are in the [data.zip](https://huggingface.co/lalayh/3DSPGNet/resolve/main/data.zip). They are in `data/models`.

### Pre-generated data

Data generation is very costly, so we upload the generated data . However, the occupancy data of [GIGA](https://github.com/UT-Austin-RPL/GIGA) exceeds 100 GB; therefore, we only uploaded the processed dataset. The new dataset alone is still sufficient to train our model and [VGN](https://github.com/ethz-asl/vgn). After downloading, extract it to the data folder under the repo's root.

| Raw data | Processed data |
| ----------- | ----------- |
| pile | [new dataset](https://huggingface.co/datasets/lalayh/Processed_data/resolve/main/new_dataset.zip) |

## Related Repositories

1. Our code is largely based on [GIGA](https://github.com/UT-Austin-RPL/GIGA) and [anygrasp](https://github.com/graspnet/anygrasp_sdk)

## Citing

```
@article{TBD,
 author = {Hang Yu, Xuebo Zhang, Zhenjie Zhao, Haochong Chen},
 journal = {IEEE TRANSACTIONS ON INDUSTRIAL ELECTRONICS (Under Review)},
 title = {3D SPGNet: A 6-DoF Grasp Detection Network via 3D Surface Constraint and TSDF Reconstruction},
 year = {2025}
}
```
