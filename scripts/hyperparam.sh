#!/bin/bash
cd /home/gdata/sandipan/BTP2021

# hyperparameter sensitivity study
python3 dirac_i.py -d 'CUB' -sn 1 -cq 3 -es 25 -c -l 'ward' -lr 0.01 -m 'latest_trained_model' -t 100 -o 'new_seed_final' -r 0.8 -n 150 -s 5 -w 0.5
# python3 dirac_i.py -d 'CUB' -sn 1 -cq 4 -es 25 -c -l 'ward' -lr 0.01 -m 'latest_trained_model' -t 100 -o 'new_seed_final' -r 0.8 -n 150 -s 5 -w 0.5
# python3 dirac_i.py -d 'SUN' -sn 1 -cq 5 -es 25 -c -l 'ward' -lr 0.001 -m 'latest_trained_model' -t 580 -o 'new_seed_final' -r 0.8 -n 645 -s 5 -w 0.5
# python3 dirac_i.py -d 'SUN' -sn 1 -cq 7 -es 25 -c -l 'ward' -lr 0.001 -m 'latest_trained_model' -t 580 -o 'new_seed_final' -r 0.8 -n 645 -s 5 -w 0.5


# final dirac_i with different arguments written: changed arg - -prev_snum or -snum instead of -sn as before
# python3 dirac_i.py -d 'SUN' -prev_snum 1 -cq 8 -es 25 -c -l 'ward' -lr 0.001 -m 'latest_trained_model' -t 580 -o 'new_seed_final' -r 0.8 -n 645 -s 5 -w 0.5
