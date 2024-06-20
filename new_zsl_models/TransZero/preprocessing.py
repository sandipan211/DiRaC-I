import pickle
import h5py
import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import torchvision.models.resnet as models
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

#new change
import pickle
import copy
import sys


class CustomedDataset(Dataset):
    def __init__(self, dataset, img_dir, file_paths, transform=None):
        self.dataset = dataset
        self.matcontent = sio.loadmat(file_paths)
        self.image_files = np.squeeze(self.matcontent['image_files'])
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx][0]
        if self.dataset == 'CUB':
            split_idx = 6 
        elif self.dataset == 'SUN':
            split_idx = 7
        elif self.dataset == 'AWA2':
            split_idx = 5
        image_file = os.path.join(self.img_dir,
                                    '/'.join(image_file.split('/')[split_idx:]))
        image = Image.open(image_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def extract_features(config):

    #new change
    czsl_folder = None
    gzsl_folder = None
    pklfile_czsl = None
    pklfile_gzsl = None
    result_filename_czsl = None
    result_filename_gzsl = None
    gzsl_save_path = None
    czsl_save_path = None

    if config.gzsl:
        #new change - made separate report folders
        gzsl_folder = config.work_dir + 'new_zsl_models/' + config.fname + '/GZSL_al_lr' + str(config.al_lr) + '/' 
        gzsl_report_folder = gzsl_folder +'u_split' + str(config.sn) + '/'
        if not os.path.exists(gzsl_folder):
            os.mkdir(gzsl_folder)
        if not os.path.exists(gzsl_report_folder):
            os.mkdir(gzsl_report_folder) 
        # if config.cq:
        #     fstr = gzsl_report_folder + config.fname + '_gzsl_' + config.dataset  + '_' + config.al_seed + '_cq' + str(config.cq)
        # else:
        #     fstr = gzsl_report_folder + config.fname + '_gzsl_' + config.dataset  + '_' + config.al_seed
        # result_filename_gzsl = fstr + '_prep_reports.txt'
        # pklfile_gzsl = fstr + '_results.pickle'
        # gzsl_res_file = open(result_filename_gzsl, 'w')



    #new change - made separate report folders
    czsl_folder = config.work_dir + 'new_zsl_models/' + config.fname + '/CZSL_al_lr' + str(config.al_lr) + '/' 
    czsl_report_folder = czsl_folder +'u_split' + str(config.sn) + '/'
    if not os.path.exists(czsl_folder):
        os.mkdir(czsl_folder)
    if not os.path.exists(czsl_report_folder):
        os.mkdir(czsl_report_folder)

    # if config.cq:
    #     fstr = czsl_report_folder + config.fname + '_czsl_' + config.dataset  + '_' + config.al_seed + '_cq' +str(config.cq)
    # else:
    #     fstr = czsl_report_folder + config.fname + '_czsl_' + config.dataset  + '_' + config.al_seed 
    # result_filename_czsl = fstr + '_prep_reports.txt'
    # pklfile_czsl = fstr + '_results.pickle'
    # czsl_res_file = open(result_filename_czsl, 'w')


    # all in-training results will be written in result_filename_czsl
    # sys.stdout = czsl_res_file


    #new change
    img_dir = f'{config.imgdata}/data/{config.dataset}'
    file_paths = f'{config.dataroot}/{config.dataset}/res101.mat'
    if config.gzsl:
        gzsl_save_path = f'{gzsl_report_folder}/feature_map_ResNet_101_{config.dataset}_{config.al_seed}.hdf5'
        
    czsl_save_path = f'{czsl_report_folder}/feature_map_ResNet_101_{config.dataset}_{config.al_seed}.hdf5'
        
    attribute_path = f'{config.work_dir}/new_zsl_models/{config.fname}/w2v/{config.dataset}_attribute.pkl'




    # regionfeature extractor
    resnet101 = models.resnet101(pretrained=True).to(config.device)
    resnet101 = nn.Sequential(*list(resnet101.children())[:-2]).eval()

    data_transforms = transforms.Compose([
        transforms.Resize(448),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    Dataset = CustomedDataset(config.dataset, img_dir, file_paths, data_transforms)
    dataset_loader = torch.utils.data.DataLoader(Dataset,
                                                 batch_size=config.batch_size,
                                                 shuffle=False,
                                                 num_workers=config.nun_workers)

    with torch.no_grad():
        all_features = []
        batch_no = 0
        for _, imgs in enumerate(dataset_loader):
            imgs = imgs.to(config.device)
            features = resnet101(imgs)
            all_features.append(features.cpu().numpy())
            batch_no += 1
            print('Load batch %d' % (batch_no))
        all_features = np.concatenate(all_features, axis=0)

    # get remaining metadata
    matcontent = Dataset.matcontent
    labels = matcontent['labels'].astype(int).squeeze() - 1

    # split_path = os.path.join(f'data/xlsa17/data/{config.dataset}/att_splits.mat')
    # matcontent = sio.loadmat(split_path)
    
    #new change
    if config.al_seed == 'original':
        split_path = config.dataroot + "/" + config.dataset + "/"  + "att_splits.mat"
        matcontent = sio.loadmat(split_path)
    else:
        #new change - file name change
        if config.cq:
            split_name = 's' + str(config.sn) + '_cq' + str(config.cq)
        else:
            split_name = 's' + str(config.sn)
        matcontent = sio.loadmat(config.dataroot + '/' + config.dataset + '/' + 'att_splits_' + config.dataset + '_al_' + config.al_seed + '_' + split_name + '.mat')
        print('Split path from util.py: ')
        print(config.dataroot + '/' + config.dataset + '/' + 'att_splits_' + config.dataset + '_al_' + config.al_seed + '_' + split_name + '.mat')

    trainval_loc = matcontent['trainval_loc'].squeeze() - 1
    # train_loc = matcontent['train_loc'].squeeze() - 1
    # val_unseen_loc = matcontent['val_loc'].squeeze() - 1
    test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
    test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
    att = matcontent['att'].T
    original_att = matcontent['original_att'].T

    # construct attribute w2v
    with open(attribute_path,'rb') as f:
        w2v_att = pickle.load(f)
    if config.dataset == 'CUB':
        assert w2v_att.shape == (312,300)
    elif config.dataset == 'SUN':
        assert w2v_att.shape == (102,300)
    elif config.dataset == 'AWA2':
        assert w2v_att.shape == (85,300)

    compression = 'gzip' if config.compression else None 
    # f = h5py.File(save_path, 'w')

    # new change
    if config.gzsl:
        f = h5py.File(gzsl_save_path, 'w')
        f.create_dataset('feature_map', data=all_features,compression=compression)
        f.create_dataset('labels', data=labels,compression=compression)
        f.create_dataset('trainval_loc', data=trainval_loc,compression=compression)
        # f.create_dataset('train_loc', data=train_loc,compression=compression)
        # f.create_dataset('val_unseen_loc', data=val_unseen_loc,compression=compression)
        f.create_dataset('test_seen_loc', data=test_seen_loc,compression=compression)
        f.create_dataset('test_unseen_loc', data=test_unseen_loc,compression=compression)
        f.create_dataset('att', data=att,compression=compression)
        f.create_dataset('original_att', data=original_att,compression=compression)
        f.create_dataset('w2v_att', data=w2v_att,compression=compression)
        f.close()

    f = h5py.File(czsl_save_path, 'w')
    f.create_dataset('feature_map', data=all_features,compression=compression)
    f.create_dataset('labels', data=labels,compression=compression)
    f.create_dataset('trainval_loc', data=trainval_loc,compression=compression)
    # f.create_dataset('train_loc', data=train_loc,compression=compression)
    # f.create_dataset('val_unseen_loc', data=val_unseen_loc,compression=compression)
    f.create_dataset('test_seen_loc', data=test_seen_loc,compression=compression)
    f.create_dataset('test_unseen_loc', data=test_unseen_loc,compression=compression)
    f.create_dataset('att', data=att,compression=compression)
    f.create_dataset('original_att', data=original_att,compression=compression)
    f.create_dataset('w2v_att', data=w2v_att,compression=compression)
    f.close()





if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='AWA2')
    parser.add_argument('--compression', '-c', action='store_true', default=False)
    parser.add_argument('--batch_size', '-b', type=int, default=16)
    parser.add_argument('--device', '-g', type=str, default='cuda:0')
    parser.add_argument('--nun_workers', '-n', type=int, default='8')

    #new change
    parser.add_argument('--dataroot', default='/workspace/sandipan/dirac-i/BTP2021/xlsa17_final/data', help='path to dataset')
    parser.add_argument('--work_dir', default='/workspace/sandipan/dirac-i/BTP2021/', help='path to dataset')
    parser.add_argument('--sn', default=1, type=int, help='split number')
    parser.add_argument('--al_lr', default=0.01, type=float, help='learning rate used during active learning')
    parser.add_argument('--imgdata', default='/workspace/sandipan/dirac-i', help='path to dataset')
    parser.add_argument('--fname', default='model_name', help='model_name')
    parser.add_argument('--al_seed', default = 'original', type =str)
    parser.add_argument('--gzsl', action='store_true', default=False, help='enable generalized zero-shot learning')



    #added for hyperparameter studies
    parser.add_argument('--cq', type=int, action='store', help='classes queried each time during DiRaC-I')
    config = parser.parse_args()
    print('\n\nStarting preprocessing.....')
    extract_features(config)
    print('Done!\n\n')
