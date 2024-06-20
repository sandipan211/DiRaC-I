#!/bin/bash
cd /home/gdata/sandipan/BTP2021

python3 dirac_i.py -d 'CUB' -snum 1 -cq 2 -es 25 -c -l 'ward' -lr 0.01 -m 'latest_trained_model' -t 100 -o 'new_seed_final' -r 0.8 -n 150 -s 5 -w 0.5
