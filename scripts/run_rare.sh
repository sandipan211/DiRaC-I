#!/bin/bash
# cd /home/gdata/sandipan/BTP2021


cd /workspace/arijit_pg/BTP2021

# ############################# commands for split 1 #################################
# python3 rare_effects.py --dataset CUB --sn 1 --al_lr 0.01 --r_ratio 0.05 --c_ratio 0.5
python3 rare_effects.py --dataset SUN --sn 1 --al_lr 0.001 --r_ratio 0.05 --c_ratio 0.5
# # #####################################################################################

# ############################# commands for split 2 #################################
# python3 rare_effects.py --dataset CUB --sn 2 --al_lr 0.01 --r_ratio 0.05 --c_ratio 0.5
python3 rare_effects.py --dataset SUN --sn 2 --al_lr 0.001 --r_ratio 0.05 --c_ratio 0.5
# #####################################################################################

# ############################# commands for split 3 #################################
# python3 rare_effects.py --dataset CUB --sn 3 --al_lr 0.01 --r_ratio 0.05 --c_ratio 0.5
python3 rare_effects.py --dataset SUN --sn 3 --al_lr 0.001 --r_ratio 0.05 --c_ratio 0.5
# #####################################################################################

