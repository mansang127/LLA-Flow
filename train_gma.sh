#!/bin/bash
mkdir -p checkpoints
python -u train.py --name lla-gma-chairs --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision --gma |tee -a output.txt
python -u train.py --name lla-gma-things --stage things --validation sintel --restore_ckpt checkpoints/lla-gma-chairs.pth --gpus 0 1 --num_steps 180000 --batch_size 6 --lr 0.000125 --image_size 400 720 --wdecay 0.0001 --mixed_precision --gma |tee -a output.txt
python -u train.py --name lla-gma-sintel --stage sintel --validation sintel --restore_ckpt checkpoints/lla-gma-things.pth --gpus 0 1 --num_steps 180000 --batch_size 6 --lr 0.000125 --image_size 368 768 --wdecay 0.00001 --gamma=0.85 --mixed_precision --gma |tee -a output.txt
python -u train.py --name lla-gma-kitti  --stage kitti --validation kitti --restore_ckpt checkpoints/lla-gma-sintel.pth --gpus 0 1 --num_steps 60000 --batch_size 6 --lr 0.000125 --image_size 288 960 --wdecay 0.00001 --gamma=0.85 --mixed_precision --gma |tee -a output.txt
