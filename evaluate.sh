#!/bin/bash
# lla-raft
python evaluate.py --model checkpoints/lla-raft-sintel.pth --mixed_precision --dataset sintel
python evaluate.py --model checkpoints/lla-raft-kitti.pth --mixed_precision --dataset kitti

python evaluate.py --model checkpoints/lla-raft-sintel.pth --mixed_precision --dataset sintel_test
python evaluate.py --model checkpoints/lla-raft-kitti.pth --mixed_precision --dataset kitti_test


# lla-gma
python evaluate.py --model checkpoints/lla-gma-sintel.pth --gma --mixed_precision --dataset sintel
python evaluate.py --model checkpoints/lla-gma-kitti.pth --gma --mixed_precision --dataset kitti

python evaluate.py --model checkpoints/lla-gma-sintel.pth --gma --mixed_precision --dataset sintel_test
python evaluate.py --model checkpoints/lla-gma-kitti.pth --gma --mixed_precision --dataset kitti_test