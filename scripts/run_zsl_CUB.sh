#!/bin/bash



########################### commands for split 1 #################################


# # For ALE
cd /home/gdata/sandipan/BTP2021/new_zsl_models/ALE

python3 ale.py -data CUB -sn 1 -al_lr 0.01 -e 50 -es 100 -norm 'L2' -lr 0.3 -al_seed 'new_seed_final'
python3 ale.py -data CUB -sn 1 -al_lr 0.01 -e 50 -es 100 -norm 'L2' -lr 0.3 -al_seed 'original'
python3 ale_gzsl.py -data CUB -sn 1 -al_lr 0.01 -e 50 -es 100 -norm 'L2' -lr 0.3 -al_seed 'new_seed_final'
python3 ale_gzsl.py -data CUB -sn 1 -al_lr 0.01 -e 50 -es 100 -norm 'L2' -lr 0.3 -al_seed 'original'



# # For ESZSL
# cd /home/gdata/sandipan/BTP2021/new_zsl_models/ESZSL

# python3 eszsl.py -data CUB -sn 1 -al_lr 0.01 -mode train -al_seed 'new_seed_final'
# python3 eszsl.py -data CUB -sn 1 -al_lr 0.01 -mode train -al_seed 'original'
# python3 eszsl_gzsl.py -data CUB -sn 1 -al_lr 0.01 -mode train -al_seed 'original'
# python3 eszsl_gzsl.py -data CUB -sn 1 -al_lr 0.01 -mode train -al_seed 'new_seed_final'


# # For DEVISE
# # Carefully select params - different for different datasets

# cd /home/gdata/sandipan/BTP2021/new_zsl_models/DEVISE
# python devise.py -data CUB -sn 1 -al_lr 0.01 -e 50 -es 100 -norm 'L2' -lr 1.0 -mr 1.0 -al_seed 'original'
# python devise.py -data CUB -sn 1 -al_lr 0.01 -e 50 -es 100 -norm 'L2' -lr 1.0 -mr 1.0 -al_seed 'new_seed_final'
# python devise_gzsl.py -data CUB -sn 1 -al_lr 0.01 -e 50 -es 100 -norm 'L2' -lr 1.0 -mr 1.0 -al_seed 'original'
# python devise_gzsl.py -data CUB -sn 1 -al_lr 0.01 -e 50 -es 100 -norm 'L2' -lr 1.0 -mr 1.0 -al_seed 'new_seed_final'


# # For SAE
# # Carefully select ld1 and ld2 - different for different datasets

# cd /home/gdata/sandipan/BTP2021/new_zsl_models/SAE
# python3 sae.py -data CUB -sn 1 -al_lr 0.01 -mode train -ld1 0.1 -ld2 200 -al_seed 'original'
# python3 sae.py -data CUB -sn 1 -al_lr 0.01 -mode train -ld1 0.1 -ld2 200 -al_seed 'new_seed_final'
# python3 sae_gzsl.py -data CUB -sn 1 -al_lr 0.01 -mode train -ld1 0.1 -ld2 200 -al_seed 'original'    
# python3 sae_gzsl.py -data CUB -sn 1 -al_lr 0.01 -mode train -ld1 0.1 -ld2 200 -al_seed 'new_seed_final'  


# # For SJE

# cd /home/gdata/sandipan/BTP2021/new_zsl_models/SJE

# python3 sje.py -data CUB -sn 1 -al_lr 0.01 -e 50 -es 100 -norm 'std' -lr 0.1 -mr 4.0 -al_seed 'original'
# python3 sje.py -data CUB -sn 1 -al_lr 0.01 -e 50 -es 100 -norm 'std' -lr 0.1 -mr 4.0 -al_seed 'new_seed_final'
# python3 sje_gzsl.py -data CUB -sn 1 -al_lr 0.01 -e 50 -es 100 -norm 'std' -lr 0.1 -mr 4.0 -al_seed 'original'
# python3 sje_gzsl.py -data CUB -sn 1 -al_lr 0.01 -e 50 -es 100 -norm 'std' -lr 0.1 -mr 4.0 -al_seed 'new_seed_final'


# # For FGN - faulty - dont run

# # cd /home/gdata/sandipan/BTP2021/new_zsl_models/FGN

# # CUDA_VISIBLE_DEVICES=0 python3 clswgan.py --manualSeed 3483 --val_every 1 --cls_weight 0.01 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 70 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset CUB --batch_size 64 --nz 312 --attSize 312 --resSize 2048 --syn_num 300 --outname cub --al_seed 'original'
# # python3 clswgan.py --manualSeed 3483 --val_every 1 --cls_weight 0.01 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 70 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset CUB --batch_size 64 --nz 312 --attSize 312 --resSize 2048 --syn_num 300 --outname cub --al_seed 'original'
# # python3 clswgan.py --manualSeed 3483 --val_every 1 --cls_weight 0.01 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 70 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset CUB --batch_size 64 --nz 312 --attSize 312 --resSize 2048 --syn_num 300 --outname cub --al_seed 'new_seed_final'
# # python3 clswgan.py --manualSeed 3483 --val_every 1 --cls_weight 0.01 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 70 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset CUB --batch_size 64 --nz 312 --attSize 312 --resSize 2048 --syn_num 300 --outname cub --al_seed 'original' --gzsl
# # python3 clswgan.py --manualSeed 3483 --val_every 1 --cls_weight 0.01 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 70 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset CUB --batch_size 64 --nz 312 --attSize 312 --resSize 2048 --syn_num 300 --outname cub --al_seed 'new_seed_final' --gzsl


# # For LSRGAN
# cd /home/gdata/sandipan/BTP2021/new_zsl_models/LSRGAN

# python3 clswgan.py --sn 1 --al_lr 0.01 --manualSeed 3483 --val_every 1 --cls_weight 0.01 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 70 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset CUB --batch_size 64 --nz 312 --attSize 312 --resSize 2048 --syn_num 300 --outname cub --upper_epsilon 0.15 --epsilon 0.15 --correlation_penalty 0.15 --no_classifier True --unseen_cls_weight 0.01 --nclass_all 200 --al_seed 'original'
# python3 clswgan.py --sn 1 --al_lr 0.01 --manualSeed 3483 --val_every 1 --cls_weight 0.01 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 70 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset CUB --batch_size 64 --nz 312 --attSize 312 --resSize 2048 --syn_num 300 --outname cub --upper_epsilon 0.15 --epsilon 0.15 --correlation_penalty 0.15 --no_classifier True --unseen_cls_weight 0.01 --nclass_all 200 --al_seed 'new_seed_final'

# # for gzsl, keerti had taken a few params changed which gave best results using hit and trial -  but here i have taken from original repo

# python3 clswgan.py --sn 1 --al_lr 0.01 --manualSeed 3483 --val_every 1 --cls_weight 0.01 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 40 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset CUB --nclass_all 200 --batch_size 64 --nz 312 --attSize 312 --resSize 2048 --syn_num 300 --outname cub --no_classifier True --epsilon 0.1 --upper_epsilon 0.1 --nclass_all 200 --correlation_penalty 20 --unseen_cls_weight 0.03 --al_seed 'original' --gzsl
# python3 clswgan.py --sn 1 --al_lr 0.01 --manualSeed 3483 --val_every 1 --cls_weight 0.01 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 40 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset CUB --nclass_all 200 --batch_size 64 --nz 312 --attSize 312 --resSize 2048 --syn_num 300 --outname cub --no_classifier True --epsilon 0.1 --upper_epsilon 0.1 --nclass_all 200 --correlation_penalty 20 --unseen_cls_weight 0.03 --al_seed 'new_seed_final' --gzsl


# # For TFVAEGAN
# # carefully check all new params with what keerti ran

# cd /home/gdata/sandipan/BTP2021/new_zsl_models/TFVAEGAN/tfvaegan-master

# CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4  python3 train_images.py --sn 1 --al_lr 0.01 --gammaD 10 --gammaG 10 \
# --gzsl --manualSeed 3483 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding att \
# --nepoch 300 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset CUB \
# --nclass_all 200 --batch_size 64 --nz 312 --latent_size 312 --attSize 312 --resSize 2048 --syn_num 300 \
# --recons_weight 0.01 --a1 1 --a2 1 --feed_lr 0.00001 --dec_lr 0.0001 --feedback_loop 2 --al_seed 'original'

# CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4  python3 train_images.py --sn 1 --al_lr 0.01 --cq 6 --gammaD 10 --gammaG 10 \
# --gzsl --manualSeed 3483 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding att \
# --nepoch 300 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset CUB \
# --nclass_all 200 --batch_size 64 --nz 312 --latent_size 312 --attSize 312 --resSize 2048 --syn_num 300 \
# --recons_weight 0.01 --a1 1 --a2 1 --feed_lr 0.00001 --dec_lr 0.0001 --feedback_loop 2 --al_seed 'new_seed_final'





# For SAGAN
# cd /home/gdata/sandipan/BTP2021/new_zsl_models/SAGAN

# python3 sagan.py --dataset CUB --sn 1 --al_lr 0.01 --al_seed 'original' --d_lr 0.05 --g_lr 0.05 --hl 512 --t 0.05 --paper_gamma 0.001 



# For CNZSL
# cd /workspace/arijit_pg/BTP2021/new_zsl_models/CNZSL
# python3 cnzsl.py --sn 1 --al_lr 0.01 --dataset CUB --image_embedding res101 --class_embedding att --al_seed 'new_seed_final'
# python3 cnzsl.py --sn 1 --al_lr 0.01 --dataset CUB --image_embedding res101 --class_embedding att --al_seed 'original'


# For FREE
# cd /workspace/arijit_pg/BTP2021/new_zsl_models/FREE
# OMP_NUM_THREADS=8 python train_free.py --fname 'free' --sn 1 --al_lr 0.01 --gammaD 10 --gammaG 10 \
# --gzsl --manualSeed 3483 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding att \
# --nepoch 501 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 1 --dataroot '/workspace/arijit_pg/BTP2021/xlsa17_final/data' --dataset CUB \
#  --a1 1 --a2 1 --feed_lr 0.00001 --dec_lr 0.0001 --loop 2 \
# --nclass_all 200 --nclass_seen 150 --batch_size 64 --nz 312 --latent_size 312 --attSize 312 --resSize 2048  \
# --syn_num 700 --center_margin 200 --center_weight 0.5 --recons_weight 0.001 --incenter_weight 0.8 --work_dir '/workspace/arijit_pg/BTP2021/' \
# --al_seed 'original'


# OMP_NUM_THREADS=8 python train_free.py --fname 'free' --sn 1 --al_lr 0.01 --gammaD 10 --gammaG 10 \
# --gzsl --manualSeed 3483 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding att \
# --nepoch 501 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 1 --dataroot '/workspace/arijit_pg/BTP2021/xlsa17_final/data' --dataset CUB \
#  --a1 1 --a2 1 --feed_lr 0.00001 --dec_lr 0.0001 --loop 2 \
# --nclass_all 200 --nclass_seen 150 --batch_size 64 --nz 312 --latent_size 312 --attSize 312 --resSize 2048  \
# --syn_num 700 --center_margin 200 --center_weight 0.5 --recons_weight 0.001 --incenter_weight 0.8 --work_dir '/workspace/arijit_pg/BTP2021/' \
# --al_seed 'new_seed_final'

# For TransZero - in CICPS DGX
# cd /workspace/sandipan/dirac-i/BTP2021/new_zsl_models/TransZero

# first do preprocessing
# python3 preprocessing.py --dataset CUB --gzsl --compression --sn 1 --al_lr 0.01 --fname 'TransZero' --al_seed 'original'
# python3 preprocessing.py --dataset CUB --gzsl --compression --sn 1 --al_lr 0.01 --fname 'TransZero' --al_seed 'new_seed_final'


# then training+eval
# python3 train_cub.py --dataset CUB --sn 1 --al_lr 0.01 --fname 'TransZero' --gzsl --workers 4 --al_seed 'original'
# python3 train_cub.py --dataset CUB --sn 1 --al_lr 0.01 --fname 'TransZero' --gzsl --workers 4 --al_seed 'new_seed_final'

# For MSDN - in CSE DGX
cd /workspace/arijit_pg/BTP2021/new_zsl_models/MSDN

# first do preprocessing - done already for TransZero in CICPS DGX and results saved in MSDN folder too
# python3 preprocessing.py --dataset CUB --gzsl --compression --sn 1 --al_lr 0.01 --fname 'TransZero' --al_seed 'original'
# python3 preprocessing.py --dataset CUB --gzsl --compression --sn 1 --al_lr 0.01 --fname 'TransZero' --al_seed 'new_seed_final'


# then training+eval
# python3 MSDN_cub.py --dataset CUB --sn 1 --al_lr 0.01 --fname 'MSDN' --gzsl --workers 4 --al_seed 'original'
# python3 MSDN_cub.py --dataset CUB --sn 1 --al_lr 0.01 --fname 'MSDN' --gzsl --workers 4 --al_seed 'new_seed_final'

############################################################################################



# ########################### commands for split 2 #################################


# For ALE
# cd /home/gdata/sandipan/BTP2021/new_zsl_models/ALE

# python3 ale.py -data CUB -sn 2 -al_lr 0.01 -e 50 -es 100 -norm 'L2' -lr 0.3 -al_seed 'new_seed_final'
# python3 ale.py -data CUB -sn 2 -al_lr 0.01 -e 50 -es 100 -norm 'L2' -lr 0.3 -al_seed 'original'
# python3 ale_gzsl.py -data CUB -sn 2 -al_lr 0.01 -e 50 -es 100 -norm 'L2' -lr 0.3 -al_seed 'new_seed_final'
# python3 ale_gzsl.py -data CUB -sn 2 -al_lr 0.01 -e 50 -es 100 -norm 'L2' -lr 0.3 -al_seed 'original'



# # For ESZSL
# cd /home/gdata/sandipan/BTP2021/new_zsl_models/ESZSL

# # python3 eszsl.py -data CUB -sn 2 -al_lr 0.01 -mode train -al_seed 'new_seed_final'
# # python3 eszsl.py -data CUB -sn 2 -al_lr 0.01 -mode train -al_seed 'original'
# python3 eszsl_gzsl.py -data CUB -sn 2 -al_lr 0.01 -mode train -al_seed 'original'
# python3 eszsl_gzsl.py -data CUB -sn 2 -al_lr 0.01 -mode train -al_seed 'new_seed_final'


# # For DEVISE
# # Carefully select params - different for different datasets

# cd /home/gdata/sandipan/BTP2021/new_zsl_models/DEVISE
# python devise.py -data CUB -sn 2 -al_lr 0.01 -e 50 -es 100 -norm 'L2' -lr 1.0 -mr 1.0 -al_seed 'original'
# python devise.py -data CUB -sn 2 -al_lr 0.01 -e 50 -es 100 -norm 'L2' -lr 1.0 -mr 1.0 -al_seed 'new_seed_final'
# python devise_gzsl.py -data CUB -sn 2 -al_lr 0.01 -e 50 -es 100 -norm 'L2' -lr 1.0 -mr 1.0 -al_seed 'original'
# python devise_gzsl.py -data CUB -sn 2 -al_lr 0.01 -e 50 -es 100 -norm 'L2' -lr 1.0 -mr 1.0 -al_seed 'new_seed_final'


# # For SAE
# # Carefully select ld1 and ld2 - different for different datasets

# cd /home/gdata/sandipan/BTP2021/new_zsl_models/SAE
# python3 sae.py -data CUB -sn 2 -al_lr 0.01 -mode train -ld1 0.1 -ld2 200 -al_seed 'original'
# python3 sae.py -data CUB -sn 2 -al_lr 0.01 -mode train -ld1 0.1 -ld2 200 -al_seed 'new_seed_final'
# python3 sae_gzsl.py -data CUB -sn 2 -al_lr 0.01 -mode train -ld1 0.1 -ld2 200 -al_seed 'original'    
# python3 sae_gzsl.py -data CUB -sn 2 -al_lr 0.01 -mode train -ld1 0.1 -ld2 200 -al_seed 'new_seed_final'  


# # For SJE

# cd /home/gdata/sandipan/BTP2021/new_zsl_models/SJE

# python3 sje.py -data CUB -sn 2 -al_lr 0.01 -e 50 -es 100 -norm 'std' -lr 0.1 -mr 4.0 -al_seed 'original'
# python3 sje.py -data CUB -sn 2 -al_lr 0.01 -e 50 -es 100 -norm 'std' -lr 0.1 -mr 4.0 -al_seed 'new_seed_final'
# python3 sje_gzsl.py -data CUB -sn 2 -al_lr 0.01 -e 50 -es 100 -norm 'std' -lr 0.1 -mr 4.0 -al_seed 'original'
# python3 sje_gzsl.py -data CUB -sn 2 -al_lr 0.01 -e 50 -es 100 -norm 'std' -lr 0.1 -mr 4.0 -al_seed 'new_seed_final'




# # For LSRGAN
# cd /home/gdata/sandipan/BTP2021/new_zsl_models/LSRGAN

# python3 clswgan.py --sn 2 --al_lr 0.01 --manualSeed 3483 --val_every 1 --cls_weight 0.01 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 70 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset CUB --batch_size 64 --nz 312 --attSize 312 --resSize 2048 --syn_num 300 --outname cub --upper_epsilon 0.15 --epsilon 0.15 --correlation_penalty 0.15 --no_classifier True --unseen_cls_weight 0.01 --nclass_all 200 --al_seed 'original'
# python3 clswgan.py --sn 2 --al_lr 0.01 --manualSeed 3483 --val_every 1 --cls_weight 0.01 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 70 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset CUB --batch_size 64 --nz 312 --attSize 312 --resSize 2048 --syn_num 300 --outname cub --upper_epsilon 0.15 --epsilon 0.15 --correlation_penalty 0.15 --no_classifier True --unseen_cls_weight 0.01 --nclass_all 200 --al_seed 'new_seed_final'

# # for gzsl, keerti had taken a few params changed which gave best results using hit and trial -  but here i have taken from original repo

# python3 clswgan.py --sn 2 --al_lr 0.01 --manualSeed 3483 --val_every 1 --cls_weight 0.01 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 40 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset CUB --nclass_all 200 --batch_size 64 --nz 312 --attSize 312 --resSize 2048 --syn_num 300 --outname cub --no_classifier True --epsilon 0.1 --upper_epsilon 0.1 --nclass_all 200 --correlation_penalty 20 --unseen_cls_weight 0.03 --al_seed 'original' --gzsl
# python3 clswgan.py --sn 2 --al_lr 0.01 --manualSeed 3483 --val_every 1 --cls_weight 0.01 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 40 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset CUB --nclass_all 200 --batch_size 64 --nz 312 --attSize 312 --resSize 2048 --syn_num 300 --outname cub --no_classifier True --epsilon 0.1 --upper_epsilon 0.1 --nclass_all 200 --correlation_penalty 20 --unseen_cls_weight 0.03 --al_seed 'new_seed_final' --gzsl


# # For TFVAEGAN
# # carefully check all new params with what keerti ran

# cd /home/gdata/sandipan/BTP2021/new_zsl_models/TFVAEGAN/tfvaegan-master

# CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4  python3 train_images.py --sn 2 --al_lr 0.01 --gammaD 10 --gammaG 10 \
# --gzsl --manualSeed 3483 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding att \
# --nepoch 300 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset CUB \
# --nclass_all 200 --batch_size 64 --nz 312 --latent_size 312 --attSize 312 --resSize 2048 --syn_num 300 \
# --recons_weight 0.01 --a1 1 --a2 1 --feed_lr 0.00001 --dec_lr 0.0001 --feedback_loop 2 --al_seed 'original'

# CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4  python3 train_images.py --sn 2 --al_lr 0.01 --gammaD 10 --gammaG 10 \
# --gzsl --manualSeed 3483 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding att \
# --nepoch 300 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset CUB \
# --nclass_all 200 --batch_size 64 --nz 312 --latent_size 312 --attSize 312 --resSize 2048 --syn_num 300 \
# --recons_weight 0.01 --a1 1 --a2 1 --feed_lr 0.00001 --dec_lr 0.0001 --feedback_loop 2 --al_seed 'new_seed_final'


# For CNZSL
# cd /workspace/arijit_pg/BTP2021/new_zsl_models/CNZSL
# python3 cnzsl.py --sn 2 --al_lr 0.01 --dataset CUB --image_embedding res101 --class_embedding att --al_seed 'new_seed_final'
# python3 cnzsl.py --sn 2 --al_lr 0.01 --dataset CUB --image_embedding res101 --class_embedding att --al_seed 'original'


# For MSDN - in CSE DGX
# cd /workspace/arijit_pg/BTP2021/new_zsl_models/MSDN

# first do preprocessing - done already for TransZero in CICPS DGX and results saved in MSDN folder too
# python3 preprocessing.py --dataset CUB --gzsl --compression --sn 2 --al_lr 0.01 --fname 'TransZero' --al_seed 'original'
# python3 preprocessing.py --dataset CUB --gzsl --compression --sn 2 --al_lr 0.01 --fname 'TransZero' --al_seed 'new_seed_final'


# then training+eval
# python3 MSDN_cub.py --dataset CUB --sn 2 --al_lr 0.01 --fname 'MSDN' --gzsl --workers 4 --al_seed 'original'
# python3 MSDN_cub.py --dataset CUB --sn 2 --al_lr 0.01 --fname 'MSDN' --gzsl --workers 4 --al_seed 'new_seed_final'


# ###########################################################################################




########################### commands for split 3 #################################


# # For ALE
# cd /home/gdata/sandipan/BTP2021/new_zsl_models/ALE

# python3 ale.py -data CUB -sn 3 -al_lr 0.01 -e 50 -es 100 -norm 'L2' -lr 0.3 -al_seed 'new_seed_final'
# python3 ale.py -data CUB -sn 3 -al_lr 0.01 -e 50 -es 100 -norm 'L2' -lr 0.3 -al_seed 'original'
# python3 ale_gzsl.py -data CUB -sn 3 -al_lr 0.01 -e 50 -es 100 -norm 'L2' -lr 0.3 -al_seed 'new_seed_final'
# python3 ale_gzsl.py -data CUB -sn 3 -al_lr 0.01 -e 50 -es 100 -norm 'L2' -lr 0.3 -al_seed 'original'



# # For ESZSL
# cd /home/gdata/sandipan/BTP2021/new_zsl_models/ESZSL

# python3 eszsl.py -data CUB -sn 3 -al_lr 0.01 -mode train -al_seed 'new_seed_final'
# python3 eszsl.py -data CUB -sn 3 -al_lr 0.01 -mode train -al_seed 'original'
# python3 eszsl_gzsl.py -data CUB -sn 3 -al_lr 0.01 -mode train -al_seed 'original'
# python3 eszsl_gzsl.py -data CUB -sn 3 -al_lr 0.01 -mode train -al_seed 'new_seed_final'


# # For DEVISE
# # Carefully select params - different for different datasets

# cd /home/gdata/sandipan/BTP2021/new_zsl_models/DEVISE
# python devise.py -data CUB -sn 3 -al_lr 0.01 -e 50 -es 100 -norm 'L2' -lr 1.0 -mr 1.0 -al_seed 'original'
# python devise.py -data CUB -sn 3 -al_lr 0.01 -e 50 -es 100 -norm 'L2' -lr 1.0 -mr 1.0 -al_seed 'new_seed_final'
# python devise_gzsl.py -data CUB -sn 3 -al_lr 0.01 -e 50 -es 100 -norm 'L2' -lr 1.0 -mr 1.0 -al_seed 'original'
# python devise_gzsl.py -data CUB -sn 3 -al_lr 0.01 -e 50 -es 100 -norm 'L2' -lr 1.0 -mr 1.0 -al_seed 'new_seed_final'


# # For SAE
# # Carefully select ld1 and ld2 - different for different datasets

# cd /home/gdata/sandipan/BTP2021/new_zsl_models/SAE
# python3 sae.py -data CUB -sn 3 -al_lr 0.01 -mode train -ld1 0.1 -ld2 200 -al_seed 'original'
# python3 sae.py -data CUB -sn 3 -al_lr 0.01 -mode train -ld1 0.1 -ld2 200 -al_seed 'new_seed_final'
# python3 sae_gzsl.py -data CUB -sn 3 -al_lr 0.01 -mode train -ld1 0.1 -ld2 200 -al_seed 'original'    
# python3 sae_gzsl.py -data CUB -sn 3 -al_lr 0.01 -mode train -ld1 0.1 -ld2 200 -al_seed 'new_seed_final'  


# # For SJE

# cd /home/gdata/sandipan/BTP2021/new_zsl_models/SJE

# python3 sje.py -data CUB -sn 3 -al_lr 0.01 -e 50 -es 100 -norm 'std' -lr 0.1 -mr 4.0 -al_seed 'original'
# python3 sje.py -data CUB -sn 3 -al_lr 0.01 -e 50 -es 100 -norm 'std' -lr 0.1 -mr 4.0 -al_seed 'new_seed_final'
# python3 sje_gzsl.py -data CUB -sn 3 -al_lr 0.01 -e 50 -es 100 -norm 'std' -lr 0.1 -mr 4.0 -al_seed 'original'
# python3 sje_gzsl.py -data CUB -sn 3 -al_lr 0.01 -e 50 -es 100 -norm 'std' -lr 0.1 -mr 4.0 -al_seed 'new_seed_final'




# # For LSRGAN
# cd /home/gdata/sandipan/BTP2021/new_zsl_models/LSRGAN

# python3 clswgan.py --sn 3 --al_lr 0.01 --manualSeed 3483 --val_every 1 --cls_weight 0.01 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 70 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset CUB --batch_size 64 --nz 312 --attSize 312 --resSize 2048 --syn_num 300 --outname cub --upper_epsilon 0.15 --epsilon 0.15 --correlation_penalty 0.15 --no_classifier True --unseen_cls_weight 0.01 --nclass_all 200 --al_seed 'original'
# python3 clswgan.py --sn 3 --al_lr 0.01 --manualSeed 3483 --val_every 1 --cls_weight 0.01 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 70 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset CUB --batch_size 64 --nz 312 --attSize 312 --resSize 2048 --syn_num 300 --outname cub --upper_epsilon 0.15 --epsilon 0.15 --correlation_penalty 0.15 --no_classifier True --unseen_cls_weight 0.01 --nclass_all 200 --al_seed 'new_seed_final'

# # for gzsl, keerti had taken a few params changed which gave best results using hit and trial -  but here i have taken from original repo

# python3 clswgan.py --sn 3 --al_lr 0.01 --manualSeed 3483 --val_every 1 --cls_weight 0.01 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 40 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset CUB --nclass_all 200 --batch_size 64 --nz 312 --attSize 312 --resSize 2048 --syn_num 300 --outname cub --no_classifier True --epsilon 0.1 --upper_epsilon 0.1 --nclass_all 200 --correlation_penalty 20 --unseen_cls_weight 0.03 --al_seed 'original' --gzsl
# python3 clswgan.py --sn 3 --al_lr 0.01 --manualSeed 3483 --val_every 1 --cls_weight 0.01 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 40 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset CUB --nclass_all 200 --batch_size 64 --nz 312 --attSize 312 --resSize 2048 --syn_num 300 --outname cub --no_classifier True --epsilon 0.1 --upper_epsilon 0.1 --nclass_all 200 --correlation_penalty 20 --unseen_cls_weight 0.03 --al_seed 'new_seed_final' --gzsl


# # For TFVAEGAN
# # carefully check all new params with what keerti ran

# cd /home/gdata/sandipan/BTP2021/new_zsl_models/TFVAEGAN/tfvaegan-master

# CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4  python3 train_images.py --sn 3 --al_lr 0.01 --gammaD 10 --gammaG 10 \
# --gzsl --manualSeed 3483 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding att \
# --nepoch 300 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset CUB \
# --nclass_all 200 --batch_size 64 --nz 312 --latent_size 312 --attSize 312 --resSize 2048 --syn_num 300 \
# --recons_weight 0.01 --a1 1 --a2 1 --feed_lr 0.00001 --dec_lr 0.0001 --feedback_loop 2 --al_seed 'original'

# CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4  python3 train_images.py --sn 3 --al_lr 0.01 --gammaD 10 --gammaG 10 \
# --gzsl --manualSeed 3483 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding att \
# --nepoch 300 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset CUB \
# --nclass_all 200 --batch_size 64 --nz 312 --latent_size 312 --attSize 312 --resSize 2048 --syn_num 300 \
# --recons_weight 0.01 --a1 1 --a2 1 --feed_lr 0.00001 --dec_lr 0.0001 --feedback_loop 2 --al_seed 'new_seed_final'


# For CNZSL
# cd /workspace/arijit_pg/BTP2021/new_zsl_models/CNZSL
# python3 cnzsl.py --sn 3 --al_lr 0.01 --dataset CUB --image_embedding res101 --class_embedding att --al_seed 'new_seed_final'
# python3 cnzsl.py --sn 3 --al_lr 0.01 --dataset CUB --image_embedding res101 --class_embedding att --al_seed 'original'

# For MSDN - in CSE DGX
# cd /workspace/arijit_pg/BTP2021/new_zsl_models/MSDN

# first do preprocessing - done already for TransZero in CICPS DGX and results saved in MSDN folder too
# python3 preprocessing.py --dataset CUB --gzsl --compression --sn 3 --al_lr 0.01 --fname 'TransZero' --al_seed 'original'
# python3 preprocessing.py --dataset CUB --gzsl --compression --sn 3 --al_lr 0.01 --fname 'TransZero' --al_seed 'new_seed_final'


# then training+eval
# python3 MSDN_cub.py --dataset CUB --sn 3 --al_lr 0.01 --fname 'MSDN' --gzsl --workers 4 --al_seed 'original'
# python3 MSDN_cub.py --dataset CUB --sn 3 --al_lr 0.01 --fname 'MSDN' --gzsl --workers 4 --al_seed 'new_seed_final'

# ###########################################################################################
