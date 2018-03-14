#!/usr/bin/env bash
source deactivate
source activate LiangNiu3.6
CUDA_VISIBLE_DEVICES=3 `which python` train_aff.py train --dataset /media/MMVCNYLOCAL_3/MMVC_NY/Liang_Niu/IIT_Affordances_2017 --model=coco
CUDA_VISIBLE_DEVICES=3 `which python` train_aff.py evaluate --dataset /media/MMVCNYLOCAL_3/MMVC_NY/Liang_Niu/IIT_Affordances_2017 --model=last
