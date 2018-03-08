#!/usr/bin/env bash
source activate LiangNiu3.6
CUDA_VISIBLE_DEVICES=3 `which python` train_aff.py train --dataset /home/niu/Liang_Niu/IIT_Affordances_2017 --model=coco