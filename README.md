# LLA-FLOW
This repository contains the source code for the paper:

[LLA-FLOW: A Lightweight Local Aggregation on Cost Volume for Optical Flow Estimation](https://arxiv.org/pdf/2304.08101.pdf)<br/>ICIP 2023 <br/>
**Jiawei Xu**, Zongqing Lu and Qingmin Liao<br/>

## Requirements
Install the environment, and the code runs well under PyTorch 1.10.
```Shell
conda env create -f requirements.yaml
```

## Pretrained Models

Pretrained models can be downloaded from the [Releases Page](https://github.com/mansang127/LLA-Flow/releases/tag/v1.0.0).

## Demos

Place a sequence of frames in the `./demo_imgs`, run the script and you can view the results in `./demo_imgs/result`:
```Shell
bash demo.sh
```

## Datasets Preparation

To train and evaluate the optical flow methods, you will need to download the required datasets. 
* [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs)
* [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [Sintel](http://sintel.is.tue.mpg.de/)
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)
* [HD1K](http://hci-benchmark.iwr.uni-heidelberg.de/)


The datasets folder should be placed in the root directory of the project.

```Shell
├── datasets
    ├── FlyingChairs_release
        ├── data
    ├── FlyingThings3D
        ├── frames_cleanpass
        ├── frames_finalpass
        ├── optical_flow
    ├── Sintel
        ├── test
        ├── training
        ├── bundler
    ├── KITTI
        ├── testing
        ├── training
    ├── HD1K
    	├── hd1k_input
    	├── hd1k_flow_gt
```

## Training
Training requires two 12GB VRAM GPUs. If you include the GMA module, it might require two 16GB VRAM GPUs.

Training based on RAFT:

```
bash train_raft.sh
```

Training based on GMA:

```
bash train_gma.sh
```

## Evaluation

Select the command from `evaluate.sh`  for model evaluation. For example:

```
python evaluate.py --model checkpoints/lla-raft-sintel.pth --mixed_precision --dataset sintel
```

## Acknowledgement

The overall code framework is adapted from [RAFT](https://github.com/princeton-vl/RAFT) and [GMA](https://github.com/zacjiang/GMA). We thank the authors for the contribution.

