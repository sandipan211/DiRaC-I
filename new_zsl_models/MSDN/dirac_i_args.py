#######################
#author: Shiming Chen
#FREE
#######################
import argparse

# CSE DGX
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CUB', help='CUB')
parser.add_argument('--dataroot', default='/workspace/arijit_pg/BTP2021/xlsa17_final/data', help='path to dataset')
parser.add_argument('--gzsl', action='store_true', default=False, help='enable generalized zero-shot learning')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)


#############################################################

#new change
parser.add_argument('--work_dir', default='/workspace/arijit_pg/BTP2021/', help='path to dataset')
parser.add_argument('--al_seed', default = 'new_seed_final', type =str)
parser.add_argument('--sn', default=1, type=int, help='split number')
parser.add_argument('--al_lr', default=0.01, type=float, help='learning rate used during active learning')
# parser.add_argument('--imgdata', default='/workspace/sandipan/', help='path to dataset')

#added for hyperparameter studies
parser.add_argument('--cq', type=int, action='store', help='classes queried each time during DiRaC-I')
parser.add_argument('--fname', default='model_name', help='model_name')

#################################################################
opt = parser.parse_args()

