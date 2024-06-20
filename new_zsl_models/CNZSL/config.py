#author: sandipan
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='AWA2', help='AWA2')
#new change - edited dataroot
parser.add_argument('--dataroot', default='/workspace/arijit_pg/BTP2021/xlsa17_final/data', help='path to dataset')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')

#new change
parser.add_argument('--al_seed', default = 'new_seed_final', type =str)
parser.add_argument('--sn', default=1, type=int, help='split number')
parser.add_argument('--al_lr', default=0.01, type=float, help='learning rate used during active learning')
#added for hyperparameter studies
parser.add_argument('--cq', type=int, action='store', help='classes queried each time during DiRaC-I')


opt = parser.parse_args()