#!/bin/bash
cd /home/gdata/sandipan/BTP2021

python3 dirac_i.py -d 'SUN' -snum 1 -cq 4 -es 25 -c -l 'ward' -lr 0.001 -m 'latest_trained_model' -t 580 -o 'new_seed_final' -r 0.8 -n 645 -s 5 -w 0.5
