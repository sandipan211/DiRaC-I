#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: shiming chen
"""
import os
#original
# os.system('''OMP_NUM_THREADS=8 python train_fear_inductive.py --gammaD 1 --gammaG 1 --gzsl \
# --manualSeed 4115 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding att --nepoch 601 --ngh 4096 \
# --a1 0.1 --a2 0.01 --loop 2 --feed_lr 0.0001 \
# --ndh 4096 --lambda1 10 --critic_iter 1 --dataset SUN \
# --batch_size 512 --nz 102 --latent_size 102 --attSize 102 --resSize 2048 --lr 0.0002 \
# --classifier_lr 0.0005 --nclass_seen 645 --nclass_all 717 --dataroot data \
# --syn_num 300 --center_margin 120 --incenter_weight 0.8 --center_weight 0.5 --recons_weight 0.1''')



# new change - my runner
os.system('''OMP_NUM_THREADS=8 python train_free.py --fname 'free' --sn 1 --al_lr 0.001 --gammaD 1 --gammaG 1 --gzsl \
--manualSeed 4115 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding att --nepoch 601 --ngh 4096 \
--a1 0.1 --a2 0.01 --loop 2 --feed_lr 0.0001 \
--ndh 4096 --lambda1 10 --critic_iter 1 --dataset SUN \
--batch_size 512 --nz 102 --latent_size 102 --attSize 102 --resSize 2048 --lr 0.0002 \
--classifier_lr 0.0005 --nclass_seen 645 --nclass_all 717 --dataroot '/workspace/arijit_pg/BTP2021/xlsa17_final/data' \
--syn_num 300 --center_margin 120 --incenter_weight 0.8 --center_weight 0.5 --recons_weight 0.1 --work_dir '/workspace/arijit_pg/BTP2021/' --al_seed 'new_seed_final'''')


OMP_NUM_THREADS=8 python train_free.py --fname 'free' --sn 1 --al_lr 0.001 --gammaD 1 --gammaG 1 --gzsl --manualSeed 4115 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding att --nepoch 601 --ngh 4096 \
--a1 0.1 --a2 0.01 --loop 2 --feed_lr 0.0001 --ndh 4096 --lambda1 10 --critic_iter 1 --dataset SUN --batch_size 512 --nz 102 --latent_size 102 --attSize 102 --resSize 2048 --lr 0.0002 \
--classifier_lr 0.0005 --nclass_seen 645 --nclass_all 717 --dataroot '/workspace/arijit_pg/BTP2021/xlsa17_final/data' --syn_num 300 --center_margin 120 --incenter_weight 0.8 --center_weight 0.5 --recons_weight 0.1 --work_dir '/workspace/arijit_pg/BTP2021/' --al_seed 'new_seed_final'

OMP_NUM_THREADS=8 python train_free.py --fname 'free' --sn 2 --al_lr 0.001 --gammaD 1 --gammaG 1 --gzsl --manualSeed 4115 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding att --nepoch 601 --ngh 4096 \
--a1 0.1 --a2 0.01 --loop 2 --feed_lr 0.0001 --ndh 4096 --lambda1 10 --critic_iter 1 --dataset SUN --batch_size 512 --nz 102 --latent_size 102 --attSize 102 --resSize 2048 --lr 0.0002 \
--classifier_lr 0.0005 --nclass_seen 645 --nclass_all 717 --dataroot '/workspace/arijit_pg/BTP2021/xlsa17_final/data' --syn_num 300 --center_margin 120 --incenter_weight 0.8 --center_weight 0.5 --recons_weight 0.1 --work_dir '/workspace/arijit_pg/BTP2021/' --al_seed 'original'
