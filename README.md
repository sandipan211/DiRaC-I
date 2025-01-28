# DiRaC-I: Identifying Diverse and Rare Training Classes for Zero-Shot Learning (ACM TOMM 2023)

## 👓 At a glance
This repository contains the official PyTorch implementation of our ACM TOMM 2023 paper : [DiRaC-I: Identifying Diverse and Rare Training Classes for Zero-Shot Learning](https://dl.acm.org/doi/10.1145/3603147), a work done by Sandipan Sarma and Arijit Sur at [Indian Institute of Technology Guwahati](https://www.iitg.ac.in/cseweb/mmlab/index2.html).


- Most existing ZSL works in image classification use a predetermined, disjoint set of seen-unseen classes [[1]](#1) to evaluate their methods. These seen (training) classes might be sub-optimal for ZSL methods to appreciate the diversity and rarity of an object domain.
- In this work, we propose a framework called Diverse and Rare Class Identifier (DiRaC-I) which, given an attribute-based dataset, can intelligently yield the most suitable “seen classes” for training ZSL models. DiRaC-I has two main goals:
  - constructing a diversified set of seed classes
  - using them to initialize a visual-semantic mining algorithm for acquiring the classes capturing both diversity and rarity in the object domain adequately. These classes can then be used as “seen classes” to train ZSL models, boosting their performance for image classification.
- We simulate a real-world scenario where visual samples of novel object classes in the wild are available to neither DiRaC-I nor the ZSL models during training and conducted extensive experiments on two benchmark data sets for zero-shot image classification — CUB and SUN.

  ![Screenshot from 2025-01-28 12-38-50](https://github.com/user-attachments/assets/c5ef47cc-af60-4aea-923a-7719260ff0d6)

## 🏢 Creating the work environment
Our code is based on PyTorch and has been implemented using an NVIDIA A100 80 GB. Install Anaconda/Miniconda on your system and create a conda environment using the following command:

```bash
conda env create -f dirac-i-env.yml
```
## ⌛ Dataset download and preprocessing
We evaluate our performance on two benchmark ZSL datasets: [CUB](https://www.vision.caltech.edu/datasets/cub_200_2011/) and [SUN](https://cs.brown.edu/~gmpatter/sunattributes.html), with ResNet-101 visual features for each image taken from [xlsa17](http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip). The data directory should have the following structure:

```
datasets
│   Label.mat
│   Split.mat    
│
└───ActivityNet_v_1_3
│   │   activity_net.v1-3.min.json
│   │   anet_classwise_videos.npy
│   |   anet_splits.npy
│   └───Anet_videos_15fps_short256
│       │   v___c8enCfzqw.mp4
│       │   v___dXUJsj3yo.mp4
│       |   ...
│
└───hmdb
│   └───hmdb51_org
│       └───brush_hair
│       └───cartwheel
│       └───...
│   
└───kinetics-400
│   └───train_256
│   │   └───abseiling
│   │   └───air_drumming
│   │   └───...
│   │
│   └───val_256
│   │   └───abseiling
│   │   └───air_drumming
│   │   └───...
│   └───zsar_kinetics_400
│   
└───kinetics-600
│   │   download.py
│   │   test.csv  
│   │   validate.csv
│   │ 
│   └───test
│   │   └───abseiling
│   │   └───acting in play
│   │   └───...
│   │
│   └───validate
│   │   └───abseiling
│   │   └───acting in play
│   │   └───...
│   └───zsar_kinetics_600
│   
└───ucf
│   └───UCF101
│       └───ApplyEyeMakeup
│       └───ApplyLipstick
│       └───...
```


## :scroll: References
<a id="1">[1]</a> 
Y. Xian, C. H. Lampert, B. Schiele and Z. Akata, "Zero-Shot Learning—A Comprehensive Evaluation of the Good, the Bad and the Ugly," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 41, no. 9, pp. 2251-2265, 1 Sept. 2019, doi: 10.1109/TPAMI.2018.2857768.


## More updates - coming soon!
