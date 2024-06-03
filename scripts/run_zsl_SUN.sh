#!/bin/bash

# ########################### commands for split 1 #################################

# # For ALE
# cd /home/gdata/sandipan/BTP2021/new_zsl_models/ALE
# python3 ale.py -data SUN -sn 1 -al_lr 0.001 -e 50 -es 100 -norm 'L2' -lr 0.1 -al_seed 'original'
# python3 ale.py -data SUN -sn 1 -al_lr 0.001 -e 50 -es 100 -norm 'L2' -lr 0.1 -al_seed 'new_seed_final'
# python3 ale_gzsl.py -data SUN -sn 1 -al_lr 0.001 -e 50 -es 100 -norm 'L2' -lr 0.1 -al_seed 'original'
# python3 ale_gzsl.py -data SUN -sn 1 -al_lr 0.001 -e 50 -es 100 -norm 'L2' -lr 0.1 -al_seed 'new_seed_final'



# # For ESZSL
# cd /home/gdata/sandipan/BTP2021/new_zsl_models/ESZSL
# python3 eszsl.py -data SUN -sn 1 -al_lr 0.001 -mode train -al_seed 'original'
# python3 eszsl.py -data SUN -sn 1 -al_lr 0.001 -mode train -al_seed 'new_seed_final'
# python eszsl_gzsl.py -data SUN -sn 1 -al_lr 0.001 -mode train -al_seed 'original'
# python eszsl_gzsl.py -data SUN -sn 1 -al_lr 0.001 -mode train -al_seed 'new_seed_final'


# # For DEVISE
# cd /home/gdata/sandipan/BTP2021/new_zsl_models/DEVISE
# python3 devise.py -data SUN -sn 1 -al_lr 0.001 -e 50 -es 100 -norm 'None' -lr 0.01 -mr 3.0 -al_seed 'original'
# python3 devise.py -data SUN -sn 1 -al_lr 0.001 -e 50 -es 100 -norm 'None' -lr 0.01 -mr 3.0 -al_seed 'new_seed_final'
# python3 devise_gzsl.py -data SUN -sn 1 -al_lr 0.001 -e 50 -es 100 -norm 'None' -lr 0.01 -mr 3.0 -al_seed 'original'
# python3 devise_gzsl.py -data SUN -sn 1 -al_lr 0.001 -e 50 -es 100 -norm 'None' -lr 0.01 -mr 3.0 -al_seed 'new_seed_final'


# # For SAE
# # Carefully select ld1 and ld2 - different for different datasets

# cd /home/gdata/sandipan/BTP2021/new_zsl_models/SAE
# python3 sae.py -data SUN -sn 1 -al_lr 0.001 -mode train -ld1 0.1 -ld2 5 -al_seed 'original' 
# python3 sae.py -data SUN -sn 1 -al_lr 0.001 -mode train -ld1 0.1 -ld2 5 -al_seed 'new_seed_final' 
# python3 sae_gzsl.py -data SUN -sn 1 -al_lr 0.001 -mode train -ld1 0.1 -ld2 5 -al_seed 'original'
# python3 sae_gzsl.py -data SUN -sn 1 -al_lr 0.001 -mode train -ld1 0.1 -ld2 5 -al_seed 'new_seed_final'


# # For SJE

# cd /home/gdata/sandipan/BTP2021/new_zsl_models/SJE

# python3 sje.py -data SUN -sn 1 -al_lr 0.001 -e 50 -es 100 -norm 'std' -lr 1.0 -mr 2.0 -al_seed 'original'
# python3 sje.py -data SUN -sn 1 -al_lr 0.001 -e 50 -es 100 -norm 'std' -lr 1.0 -mr 2.0 -al_seed 'new_seed_final'
# python3 sje_gzsl.py -data SUN -sn 1 -al_lr 0.001 -e 50 -es 100 -norm 'std' -lr 1.0 -mr 2.0 -al_seed 'original'
# python3 sje_gzsl.py -data SUN -sn 1 -al_lr 0.001 -e 50 -es 100 -norm 'std' -lr 1.0 -mr 2.0 -al_seed 'new_seed_final'


# #For LSRGAN
# cd /home/gdata/sandipan/BTP2021/new_zsl_models/LSRGAN

# python clswgan.py --sn 1 --al_lr 0.001 --manualSeed 4115 --cls_weight 0.03 --val_every 1 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 35 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 64 --nz 102 --attSize 102 --resSize 2048 --lr 0.0002 --classifier_lr 0.0005 --syn_num 100 --outname sun --upper_epsilon 0.1 --epsilon 0.1 --correlation_penalty 20 --no_classifier True --unseen_cls_weight 0.01 --nclass_all 717 --al_seed 'original'

# python clswgan.py --sn 1 --al_lr 0.001 --manualSeed 4115 --cls_weight 0.03 --val_every 1 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 35 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 64 --nz 102 --attSize 102 --resSize 2048 --lr 0.0002 --classifier_lr 0.0005 --syn_num 100 --outname sun --upper_epsilon 0.1 --epsilon 0.1 --correlation_penalty 20 --no_classifier True --unseen_cls_weight 0.01 --nclass_all 717 --al_seed 'new_seed_final'

# # # for gzsl, keerti had taken a few params changed which gave best results using hit and trial -  but here i have taken from original repo
# python3 clswgan.py --gzsl --sn 1 --al_lr 0.001 --manualSeed 4115 --cls_weight 0.03 --val_every 1 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 20 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 64 --nz 102 --attSize 102 --resSize 2048 --lr 0.0002 --syn_num 400 --classifier_lr 0.001 --nclass_all 717 --outname sun --upper_epsilon 0.1 --epsilon 0.1 --correlation_penalty 10 --no_classifier True --unseen_cls_weight 0.01 --al_seed 'original'

# python3 clswgan.py --gzsl --sn 1 --al_lr 0.001 --manualSeed 4115 --cls_weight 0.03 --val_every 1 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 20 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 64 --nz 102 --attSize 102 --resSize 2048 --lr 0.0002 --syn_num 400 --classifier_lr 0.001 --nclass_all 717 --outname sun --upper_epsilon 0.1 --epsilon 0.1 --correlation_penalty 10 --no_classifier True --unseen_cls_weight 0.01 --al_seed 'new_seed_final'


# # For TFVAEGAN
# # carefully check all new params with what keerti ran

# cd /home/gdata/sandipan/BTP2021/new_zsl_models/TFVAEGAN/tfvaegan-master
# CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python3 train_images.py --sn 1 --al_lr 0.001 --gammaD 1 --gammaG 1 --gzsl --manualSeed 4115 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding att --nepoch 400 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 64 --nz 102 --latent_size 102 --attSize 102 --resSize 2048 --lr 0.001 --classifier_lr 0.0005 --syn_num 400 --nclass_all 717 --recons_weight 0.01 --a1 0.1 --a2 0.01 --feedback_loop 2 --feed_lr 0.0001 --al_seed 'original'


# for hyperparam studies - keep changing --cq as 5,6,7...

# CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python3 train_images.py --sn 1 --al_lr 0.001 --cq 8 --gammaD 1 --gammaG 1 --manualSeed 4115 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding att --nepoch 400 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 64 --nz 102 --latent_size 102 --attSize 102 --resSize 2048 --lr 0.001 --classifier_lr 0.0005 --syn_num 400 --nclass_all 717 --recons_weight 0.01 --a1 0.1 --a2 0.01 --feedback_loop 2 --feed_lr 0.0001 --al_seed 'new_seed_final'


# python3 train_images.py --sn 1 --al_lr 0.001 --cq 8 --gammaD 1 --gammaG 1 --manualSeed 4115 --encoded_noise --preprocessing --image_embedding res101 --class_embedding att --nepoch 400 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 64 --nz 102 --latent_size 102 --attSize 102 --resSize 2048 --lr 0.001 --classifier_lr 0.0005 --syn_num 400 --nclass_all 717 --recons_weight 0.01 --a1 0.1 --a2 0.01 --feedback_loop 2 --feed_lr 0.0001 --al_seed 'new_seed_final'


# For SAGAN
# cd /home/gdata/sandipan/BTP2021/new_zsl_models/SAGAN

# python3 sagan.py --dataset SUN --sn 1 --al_lr 0.001 --al_seed 'original' --d_lr 0.005 --g_lr 0.005 --hl 1024 --t 0.05 --paper_gamma 0.00001 


# For CNZSL
# cd /workspace/arijit_pg/BTP2021/new_zsl_models/CNZSL
# python3 cnzsl.py --sn 1 --al_lr 0.001 --dataset SUN --image_embedding res101 --class_embedding att --al_seed 'original'
# python3 cnzsl.py --sn 1 --al_lr 0.001 --dataset SUN --image_embedding res101 --class_embedding att --al_seed 'new_seed_final'


# For MSDN - in CSE DGX
cd /workspace/arijit_pg/BTP2021/new_zsl_models/MSDN

# first do preprocessing - done already for TransZero in CICPS DGX and results saved in MSDN folder too
# python3 preprocessing.py --dataset CUB --gzsl --compression --sn 1 --al_lr 0.01 --fname 'TransZero' --al_seed 'original'
# python3 preprocessing.py --dataset CUB --gzsl --compression --sn 1 --al_lr 0.01 --fname 'TransZero' --al_seed 'new_seed_final'


# then training+eval
# python3 MSDN_sun.py --dataset SUN --sn 1 --al_lr 0.001 --fname 'MSDN' --gzsl --workers 4 --al_seed 'original'
# python3 MSDN_sun.py --dataset SUN --sn 1 --al_lr 0.001 --fname 'MSDN' --gzsl --workers 4 --al_seed 'new_seed_final'

# #################################################################################


# ########################### commands for split 2 #################################

# # For ALE
# cd /home/gdata/sandipan/BTP2021/new_zsl_models/ALE
# python3 ale.py -data SUN -sn 2 -al_lr 0.001 -e 50 -es 100 -norm 'L2' -lr 0.1 -al_seed 'original'
# python3 ale.py -data SUN -sn 2 -al_lr 0.001 -e 50 -es 100 -norm 'L2' -lr 0.1 -al_seed 'new_seed_final'
# python3 ale_gzsl.py -data SUN -sn 2 -al_lr 0.001 -e 50 -es 100 -norm 'L2' -lr 0.1 -al_seed 'original'
# python3 ale_gzsl.py -data SUN -sn 2 -al_lr 0.001 -e 50 -es 100 -norm 'L2' -lr 0.1 -al_seed 'new_seed_final'



# # For ESZSL
# cd /home/gdata/sandipan/BTP2021/new_zsl_models/ESZSL
# python3 eszsl.py -data SUN -sn 2 -al_lr 0.001 -mode train -al_seed 'original'
# python3 eszsl.py -data SUN -sn 2 -al_lr 0.001 -mode train -al_seed 'new_seed_final'
# python eszsl_gzsl.py -data SUN -sn 2 -al_lr 0.001 -mode train -al_seed 'original'
# python eszsl_gzsl.py -data SUN -sn 2 -al_lr 0.001 -mode train -al_seed 'new_seed_final'


# # For DEVISE
# cd /home/gdata/sandipan/BTP2021/new_zsl_models/DEVISE
# python3 devise.py -data SUN -sn 2 -al_lr 0.001 -e 50 -es 100 -norm 'None' -lr 0.01 -mr 3.0 -al_seed 'original'
# python3 devise.py -data SUN -sn 2 -al_lr 0.001 -e 50 -es 100 -norm 'None' -lr 0.01 -mr 3.0 -al_seed 'new_seed_final'
# python3 devise_gzsl.py -data SUN -sn 2 -al_lr 0.001 -e 50 -es 100 -norm 'None' -lr 0.01 -mr 3.0 -al_seed 'original'
# python3 devise_gzsl.py -data SUN -sn 2 -al_lr 0.001 -e 50 -es 100 -norm 'None' -lr 0.01 -mr 3.0 -al_seed 'new_seed_final'


# # For SAE
# # Carefully select ld1 and ld2 - different for different datasets

# cd /home/gdata/sandipan/BTP2021/new_zsl_models/SAE
# python3 sae.py -data SUN -sn 2 -al_lr 0.001 -mode train -ld1 0.1 -ld2 5 -al_seed 'original' 
# python3 sae.py -data SUN -sn 2 -al_lr 0.001 -mode train -ld1 0.1 -ld2 5 -al_seed 'new_seed_final' 
# python3 sae_gzsl.py -data SUN -sn 2 -al_lr 0.001 -mode train -ld1 0.1 -ld2 5 -al_seed 'original'
# python3 sae_gzsl.py -data SUN -sn 2 -al_lr 0.001 -mode train -ld1 0.1 -ld2 5 -al_seed 'new_seed_final'


# # For SJE

# cd /home/gdata/sandipan/BTP2021/new_zsl_models/SJE

# python3 sje.py -data SUN -sn 2 -al_lr 0.001 -e 50 -es 100 -norm 'std' -lr 1.0 -mr 2.0 -al_seed 'original'
# python3 sje.py -data SUN -sn 2 -al_lr 0.001 -e 50 -es 100 -norm 'std' -lr 1.0 -mr 2.0 -al_seed 'new_seed_final'
# python3 sje_gzsl.py -data SUN -sn 2 -al_lr 0.001 -e 50 -es 100 -norm 'std' -lr 1.0 -mr 2.0 -al_seed 'original'
# python3 sje_gzsl.py -data SUN -sn 2 -al_lr 0.001 -e 50 -es 100 -norm 'std' -lr 1.0 -mr 2.0 -al_seed 'new_seed_final'


# #For LSRGAN
# cd /home/gdata/sandipan/BTP2021/new_zsl_models/LSRGAN

# python clswgan.py --sn 2 --al_lr 0.001 --manualSeed 4115 --cls_weight 0.03 --val_every 1 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 35 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 64 --nz 102 --attSize 102 --resSize 2048 --lr 0.0002 --classifier_lr 0.0005 --syn_num 100 --outname sun --upper_epsilon 0.1 --epsilon 0.1 --correlation_penalty 20 --no_classifier True --unseen_cls_weight 0.01 --nclass_all 717 --al_seed 'original'

# python clswgan.py --sn 2 --al_lr 0.001 --manualSeed 4115 --cls_weight 0.03 --val_every 1 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 35 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 64 --nz 102 --attSize 102 --resSize 2048 --lr 0.0002 --classifier_lr 0.0005 --syn_num 100 --outname sun --upper_epsilon 0.1 --epsilon 0.1 --correlation_penalty 20 --no_classifier True --unseen_cls_weight 0.01 --nclass_all 717 --al_seed 'new_seed_final'

# # # for gzsl, keerti had taken a few params changed which gave best results using hit and trial -  but here i have taken from original repo
# python3 clswgan.py --gzsl --sn 2 --al_lr 0.001 --manualSeed 4115 --cls_weight 0.03 --val_every 1 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 20 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 64 --nz 102 --attSize 102 --resSize 2048 --lr 0.0002 --syn_num 400 --classifier_lr 0.001 --nclass_all 717 --outname sun --upper_epsilon 0.1 --epsilon 0.1 --correlation_penalty 10 --no_classifier True --unseen_cls_weight 0.01 --al_seed 'original'

# python3 clswgan.py --gzsl --sn 2 --al_lr 0.001 --manualSeed 4115 --cls_weight 0.03 --val_every 1 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 20 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 64 --nz 102 --attSize 102 --resSize 2048 --lr 0.0002 --syn_num 400 --classifier_lr 0.001 --nclass_all 717 --outname sun --upper_epsilon 0.1 --epsilon 0.1 --correlation_penalty 10 --no_classifier True --unseen_cls_weight 0.01 --al_seed 'new_seed_final'


# # For TFVAEGAN
# # carefully check all new params with what keerti ran

# cd /home/gdata/sandipan/BTP2021/new_zsl_models/TFVAEGAN/tfvaegan-master
# CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python3 train_images.py --sn 2 --al_lr 0.001 --gammaD 1 --gammaG 1 --gzsl --manualSeed 4115 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding att --nepoch 400 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 64 --nz 102 --latent_size 102 --attSize 102 --resSize 2048 --lr 0.001 --classifier_lr 0.0005 --syn_num 400 --nclass_all 717 --recons_weight 0.01 --a1 0.1 --a2 0.01 --feedback_loop 2 --feed_lr 0.0001 --al_seed 'original'

# CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python3 train_images.py --sn 2 --al_lr 0.001 --gammaD 1 --gammaG 1 --gzsl --manualSeed 4115 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding att --nepoch 400 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 64 --nz 102 --latent_size 102 --attSize 102 --resSize 2048 --lr 0.001 --classifier_lr 0.0005 --syn_num 400 --nclass_all 717 --recons_weight 0.01 --a1 0.1 --a2 0.01 --feedback_loop 2 --feed_lr 0.0001 --al_seed 'new_seed_final'


# For MSDN - in CSE DGX
# cd /workspace/arijit_pg/BTP2021/new_zsl_models/MSDN

# first do preprocessing - done already for TransZero in CICPS DGX and results saved in MSDN folder too
# python3 preprocessing.py --dataset SUN --gzsl --compression --sn 2 --al_lr 0.001 --fname 'MSDN' --al_seed 'original'
# python3 preprocessing.py --dataset SUN --gzsl --compression --sn 2 --al_lr 0.001 --fname 'MSDN' --al_seed 'new_seed_final'


# then training+eval
python3 MSDN_sun.py --dataset SUN --sn 2 --al_lr 0.001 --fname 'MSDN' --gzsl --workers 4 --al_seed 'original'
python3 MSDN_sun.py --dataset SUN --sn 2 --al_lr 0.001 --fname 'MSDN' --gzsl --workers 4 --al_seed 'new_seed_final'


# #################################################################################

# ########################### commands for split 3 #################################

# # For ALE
# cd /home/gdata/sandipan/BTP2021/new_zsl_models/ALE
# python3 ale.py -data SUN -sn 3 -al_lr 0.001 -e 50 -es 100 -norm 'L2' -lr 0.1 -al_seed 'original'
# python3 ale.py -data SUN -sn 3 -al_lr 0.001 -e 50 -es 100 -norm 'L2' -lr 0.1 -al_seed 'new_seed_final'
# python3 ale_gzsl.py -data SUN -sn 3 -al_lr 0.001 -e 50 -es 100 -norm 'L2' -lr 0.1 -al_seed 'original'
# python3 ale_gzsl.py -data SUN -sn 3 -al_lr 0.001 -e 50 -es 100 -norm 'L2' -lr 0.1 -al_seed 'new_seed_final'



# # For ESZSL
# cd /home/gdata/sandipan/BTP2021/new_zsl_models/ESZSL
# python3 eszsl.py -data SUN -sn 3 -al_lr 0.001 -mode train -al_seed 'original'
# python3 eszsl.py -data SUN -sn 3 -al_lr 0.001 -mode train -al_seed 'new_seed_final'
# python eszsl_gzsl.py -data SUN -sn 3 -al_lr 0.001 -mode train -al_seed 'original'
# python eszsl_gzsl.py -data SUN -sn 3 -al_lr 0.001 -mode train -al_seed 'new_seed_final'


# # For DEVISE
# cd /home/gdata/sandipan/BTP2021/new_zsl_models/DEVISE
# python3 devise.py -data SUN -sn 3 -al_lr 0.001 -e 50 -es 100 -norm 'None' -lr 0.01 -mr 3.0 -al_seed 'original'
# python3 devise.py -data SUN -sn 3 -al_lr 0.001 -e 50 -es 100 -norm 'None' -lr 0.01 -mr 3.0 -al_seed 'new_seed_final'
# python3 devise_gzsl.py -data SUN -sn 3 -al_lr 0.001 -e 50 -es 100 -norm 'None' -lr 0.01 -mr 3.0 -al_seed 'original'
# python3 devise_gzsl.py -data SUN -sn 3 -al_lr 0.001 -e 50 -es 100 -norm 'None' -lr 0.01 -mr 3.0 -al_seed 'new_seed_final'


# # For SAE
# # Carefully select ld1 and ld2 - different for different datasets

# cd /home/gdata/sandipan/BTP2021/new_zsl_models/SAE
# python3 sae.py -data SUN -sn 3 -al_lr 0.001 -mode train -ld1 0.1 -ld2 5 -al_seed 'original' 
# python3 sae.py -data SUN -sn 3 -al_lr 0.001 -mode train -ld1 0.1 -ld2 5 -al_seed 'new_seed_final' 
# python3 sae_gzsl.py -data SUN -sn 3 -al_lr 0.001 -mode train -ld1 0.1 -ld2 5 -al_seed 'original'
# python3 sae_gzsl.py -data SUN -sn 3 -al_lr 0.001 -mode train -ld1 0.1 -ld2 5 -al_seed 'new_seed_final'


# # For SJE

# cd /home/gdata/sandipan/BTP2021/new_zsl_models/SJE

# python3 sje.py -data SUN -sn 3 -al_lr 0.001 -e 50 -es 100 -norm 'std' -lr 1.0 -mr 2.0 -al_seed 'original'
# python3 sje.py -data SUN -sn 3 -al_lr 0.001 -e 50 -es 100 -norm 'std' -lr 1.0 -mr 2.0 -al_seed 'new_seed_final'
# python3 sje_gzsl.py -data SUN -sn 3 -al_lr 0.001 -e 50 -es 100 -norm 'std' -lr 1.0 -mr 2.0 -al_seed 'original'
# python3 sje_gzsl.py -data SUN -sn 3 -al_lr 0.001 -e 50 -es 100 -norm 'std' -lr 1.0 -mr 2.0 -al_seed 'new_seed_final'


# #For LSRGAN
# cd /home/gdata/sandipan/BTP2021/new_zsl_models/LSRGAN

# python clswgan.py --sn 3 --al_lr 0.001 --manualSeed 4115 --cls_weight 0.03 --val_every 1 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 35 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 64 --nz 102 --attSize 102 --resSize 2048 --lr 0.0002 --classifier_lr 0.0005 --syn_num 100 --outname sun --upper_epsilon 0.1 --epsilon 0.1 --correlation_penalty 20 --no_classifier True --unseen_cls_weight 0.01 --nclass_all 717 --al_seed 'original'

# python clswgan.py --sn 3 --al_lr 0.001 --manualSeed 4115 --cls_weight 0.03 --val_every 1 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 35 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 64 --nz 102 --attSize 102 --resSize 2048 --lr 0.0002 --classifier_lr 0.0005 --syn_num 100 --outname sun --upper_epsilon 0.1 --epsilon 0.1 --correlation_penalty 20 --no_classifier True --unseen_cls_weight 0.01 --nclass_all 717 --al_seed 'new_seed_final'

# # # for gzsl, keerti had taken a few params changed which gave best results using hit and trial -  but here i have taken from original repo
# python3 clswgan.py --gzsl --sn 3 --al_lr 0.001 --manualSeed 4115 --cls_weight 0.03 --val_every 1 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 20 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 64 --nz 102 --attSize 102 --resSize 2048 --lr 0.0002 --syn_num 400 --classifier_lr 0.001 --nclass_all 717 --outname sun --upper_epsilon 0.1 --epsilon 0.1 --correlation_penalty 10 --no_classifier True --unseen_cls_weight 0.01 --al_seed 'original'

# python3 clswgan.py --gzsl --sn 3 --al_lr 0.001 --manualSeed 4115 --cls_weight 0.03 --val_every 1 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 20 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 64 --nz 102 --attSize 102 --resSize 2048 --lr 0.0002 --syn_num 400 --classifier_lr 0.001 --nclass_all 717 --outname sun --upper_epsilon 0.1 --epsilon 0.1 --correlation_penalty 10 --no_classifier True --unseen_cls_weight 0.01 --al_seed 'new_seed_final'


# # For TFVAEGAN
# # carefully check all new params with what keerti ran

# cd /home/gdata/sandipan/BTP2021/new_zsl_models/TFVAEGAN/tfvaegan-master
# CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python3 train_images.py --sn 3 --al_lr 0.001 --gammaD 1 --gammaG 1 --gzsl --manualSeed 4115 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding att --nepoch 400 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 64 --nz 102 --latent_size 102 --attSize 102 --resSize 2048 --lr 0.001 --classifier_lr 0.0005 --syn_num 400 --nclass_all 717 --recons_weight 0.01 --a1 0.1 --a2 0.01 --feedback_loop 2 --feed_lr 0.0001 --al_seed 'original'

# CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python3 train_images.py --sn 3 --al_lr 0.001 --gammaD 1 --gammaG 1 --gzsl --manualSeed 4115 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding att --nepoch 400 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 64 --nz 102 --latent_size 102 --attSize 102 --resSize 2048 --lr 0.001 --classifier_lr 0.0005 --syn_num 400 --nclass_all 717 --recons_weight 0.01 --a1 0.1 --a2 0.01 --feedback_loop 2 --feed_lr 0.0001 --al_seed 'new_seed_final'


# For MSDN - in CSE DGX
# cd /workspace/arijit_pg/BTP2021/new_zsl_models/MSDN

# first do preprocessing - done already for TransZero in CICPS DGX and results saved in MSDN folder too
# python3 preprocessing.py --dataset SUN --gzsl --compression --sn 2 --al_lr 0.001 --fname 'MSDN' --al_seed 'original'
# python3 preprocessing.py --dataset SUN --gzsl --compression --sn 2 --al_lr 0.001 --fname 'MSDN' --al_seed 'new_seed_final'


# then training+eval
python3 MSDN_sun.py --dataset SUN --sn 3 --al_lr 0.001 --fname 'MSDN' --gzsl --workers 4 --al_seed 'original'
python3 MSDN_sun.py --dataset SUN --sn 3 --al_lr 0.001 --fname 'MSDN' --gzsl --workers 4 --al_seed 'new_seed_final'

# #################################################################################