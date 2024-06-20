# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 11:53:09 2019

@author: badat
"""
import os,sys
#import scipy.io as sio
import torch
import numpy as np
import h5py
import time
import pickle
import pdb
from sklearn import preprocessing
from global_setting import NFS_path
#%%
import scipy.io as sio
import pandas as pd
#%%
#import pdb
#%%

img_dir = os.path.join(NFS_path,'data/CUB/')

class CUBDataLoader():
    def __init__(self, data_path, device, is_scale = False,is_unsupervised_attr = False,is_balance=True, opt=None):

        print(data_path)
        sys.path.append(data_path)

        self.data_path = data_path
        self.device = device
        self.dataset = 'CUB'
        print('$'*30)
        print(self.dataset)
        print('$'*30)
        self.datadir = self.data_path + 'data/{}/'.format(self.dataset)
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.is_scale = is_scale
        self.is_balance = is_balance
        if self.is_balance:
            print('Balance dataloader')
        self.is_unsupervised_attr = is_unsupervised_attr
        
        #new change
        self.opt = opt
        self.read_matdataset()
        self.get_idx_classes()        
        
    def next_batch_img(self, batch_size,class_id,is_trainset = False):
        features = None
        labels = None
        img_files = None
        if class_id in self.seenclasses:
            if is_trainset:
                features = self.data['train_seen']['resnet_features']
                labels = self.data['train_seen']['labels']
                img_files = self.data['train_seen']['img_path']
            else:
                features = self.data['test_seen']['resnet_features']
                labels = self.data['test_seen']['labels']
                img_files = self.data['test_seen']['img_path']
        elif class_id in self.unseenclasses:
            features = self.data['test_unseen']['resnet_features']
            labels = self.data['test_unseen']['labels']
            img_files = self.data['test_unseen']['img_path']
        else:
            raise Exception("Cannot find this class {}".format(class_id))
        
        #note that img_files is numpy type !!!!!
        
        idx_c = torch.squeeze(torch.nonzero(labels == class_id))
        
        features = features[idx_c]
        labels = labels[idx_c]
        img_files = img_files[idx_c.cpu().numpy()]
        
        batch_label = labels[:batch_size].to(self.device)
        batch_feature = features[:batch_size].to(self.device)
        batch_files = img_files[:batch_size]
        batch_att = self.att[batch_label].to(self.device)
        
        return batch_label, batch_feature,batch_files, batch_att
    

    def next_batch(self, batch_size):
        if self.is_balance:
            idx = []
            n_samples_class = max(batch_size //self.ntrain_class,1)
            sampled_idx_c = np.random.choice(np.arange(self.ntrain_class),min(self.ntrain_class,batch_size),replace=False).tolist()
            for i_c in sampled_idx_c:
                idxs = self.idxs_list[i_c]
                idx.append(np.random.choice(idxs,n_samples_class))
            idx = np.concatenate(idx)
            idx = torch.from_numpy(idx)
        else:
            idx = torch.randperm(self.ntrain)[0:batch_size]
    
        batch_feature = self.data['train_seen']['resnet_features'][idx].to(self.device)
        batch_label =  self.data['train_seen']['labels'][idx].to(self.device)
        batch_att = self.att[batch_label].to(self.device)
        return batch_label, batch_feature, batch_att
    
    def get_idx_classes(self):
        n_classes = self.seenclasses.size(0)
        self.idxs_list = []
        train_label = self.data['train_seen']['labels']
        for i in range(n_classes):
            idx_c = torch.nonzero(train_label == self.seenclasses[i].cpu()).cpu().numpy()
            idx_c = np.squeeze(idx_c)
            self.idxs_list.append(idx_c)
        return self.idxs_list

    #new change
    def get_names(self, config=None):
        if config.al_seed == 'original':
            split_path = config.dataroot + "/" + config.dataset + "/"  + "att_splits.mat"
            matcontent = sio.loadmat(split_path)
        else:
            if config.cq:
                split_name = 's' + str(config.sn) + '_cq' + str(config.cq)
            else:
                split_name = 's' + str(config.sn)
            matcontent = sio.loadmat(config.dataroot + '/' + config.dataset + '/' + 'att_splits_' + config.dataset + '_al_' + config.al_seed + '_' + split_name + '.mat')
            print('Split path from dirac-i: ')
            print(config.dataroot + '/' + config.dataset + '/' + 'att_splits_' + config.dataset + '_al_' + config.al_seed + '_' + split_name + '.mat')

        allclasses = matcontent['allclasses_names']
        return allclasses

    def read_matdataset(self):

        # path= self.datadir + 'feature_map_ResNet_101_{}.hdf5'.format(self.dataset)
        if self.opt.gzsl:
            folder = '/GZSL_al_lr'
        else:
            folder = '/CZSL_al_lr'
        zsl_folder = self.opt.work_dir + 'new_zsl_models/' + self.opt.fname + folder + str(self.opt.al_lr) + '/' 
        zsl_report_folder = zsl_folder +'u_split' + str(self.opt.sn) + '/'
        path = zsl_report_folder + 'feature_map_ResNet_101_{}_{}.hdf5'.format(self.opt.dataset, self.opt.al_seed)



        print('_____')
        print(path)
        # tic = time.time()
        hf = h5py.File(path, 'r')
        features = np.array(hf.get('feature_map'))
#        shape = features.shape
#        features = features.reshape(shape[0],shape[1],shape[2]*shape[3])
#        pdb.set_trace()
        labels = np.array(hf.get('labels'))
        trainval_loc = np.array(hf.get('trainval_loc'))
#        train_loc = np.array(hf.get('train_loc')) #--> train_feature = TRAIN SEEN
#        val_unseen_loc = np.array(hf.get('val_unseen_loc')) #--> test_unseen_feature = TEST UNSEEN
        test_seen_loc = np.array(hf.get('test_seen_loc'))
        test_unseen_loc = np.array(hf.get('test_unseen_loc'))
        
        if self.is_unsupervised_attr:
            print('Unsupervised Attr')
            class_path = './w2v/{}_class.pkl'.format(self.dataset)
            with open(class_path,'rb') as f:
                w2v_class = pickle.load(f)
            temp = np.array(hf.get('att'))
            print(w2v_class.shape,temp.shape)
#            assert w2v_class.shape == temp.shape 
            w2v_class = torch.tensor(w2v_class).float()
            
            U, s, V = torch.svd(w2v_class)
            reconstruct = torch.mm(torch.mm(U,torch.diag(s)),torch.transpose(V,1,0))
            print('sanity check: {}'.format(torch.norm(reconstruct-w2v_class).item()))
            
            print('shape U:{} V:{}'.format(U.size(),V.size()))
            print('s: {}'.format(s))
            
            self.w2v_att = torch.transpose(V,1,0).to(self.device)
            self.att = torch.mm(U,torch.diag(s)).to(self.device)
            self.normalize_att = torch.mm(U,torch.diag(s)).to(self.device)
            
        else:
            print('Expert Attr')
            att = np.array(hf.get('att'))
            self.att = torch.from_numpy(att).float().to(self.device)
            
            original_att = np.array(hf.get('original_att'))
            self.original_att = torch.from_numpy(original_att).float().to(self.device)
            
            w2v_att = np.array(hf.get('w2v_att'))
            self.w2v_att = torch.from_numpy(w2v_att).float().to(self.device)
            
            self.normalize_att = self.original_att/100
        
        # print('Finish loading data in ',time.time()-tic)
        
        train_feature = features[trainval_loc]
        test_seen_feature = features[test_seen_loc]
        test_unseen_feature = features[test_unseen_loc]
        if self.is_scale:
            scaler = preprocessing.MinMaxScaler()
    
            train_feature = scaler.fit_transform(train_feature)
            test_seen_feature = scaler.fit_transform(test_seen_feature)
            test_unseen_feature = scaler.fit_transform(test_unseen_feature)

        train_feature = torch.from_numpy(train_feature).float() #.to(self.device)
        test_seen_feature = torch.from_numpy(test_seen_feature) #.float().to(self.device)
        test_unseen_feature = torch.from_numpy(test_unseen_feature) #.float().to(self.device)

        train_label = torch.from_numpy(labels[trainval_loc]).long() #.to(self.device)
        test_unseen_label = torch.from_numpy(labels[test_unseen_loc]) #.long().to(self.device)
        test_seen_label = torch.from_numpy(labels[test_seen_loc]) #.long().to(self.device)

        self.seenclasses = torch.from_numpy(np.unique(train_label.cpu().numpy())).to(self.device)
        self.unseenclasses = torch.from_numpy(np.unique(test_unseen_label.cpu().numpy())).to(self.device)
        self.ntrain = train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()

#        self.train_mapped_label = map_label(train_label, self.seenclasses)

        #new change
        self.all_names = self.get_names(self.opt)
                #new change - get the split_info in final results
        if self.opt.cq:
            splits_folder = self.opt.work_dir + self.opt.dataset + '/' + 'split_info_lr' + str(self.opt.al_lr) + '_cq' + str(self.opt.cq) + '/'
        else:
            splits_folder = self.opt.work_dir + self.opt.dataset + '/' + 'split_info_lr' + str(self.opt.al_lr) + '/'
        pklfile = splits_folder + 'u_split' + str(self.opt.sn) + '_' + 'split_info_' + self.opt.dataset + '.pickle'
        res = open(pklfile, 'rb')
        final_results = pickle.load(res)
        # print('Split info: \n')
        # print(final_results)        
        self.common_unseen = final_results['common_unseen']
        print('Common unseen test classes ({}): {}'.format(len(self.common_unseen), self.common_unseen))
        self.test_res = {}
        self.trainval_class_names, self.testnames_unseen, self.testnames_seen = [], [], []
        trainval_labels_seen = np.unique(train_label.numpy())
        test_labels_seen = np.unique(test_seen_label.numpy())
        test_labels_unseen = np.unique(test_unseen_label.numpy())

        for i in trainval_labels_seen:
            self.trainval_class_names.append(self.all_names[i][0][0])
        for i in test_labels_seen:
            self.testnames_seen.append(self.all_names[i][0][0])
        for i in test_labels_unseen:
            self.testnames_unseen.append(self.all_names[i][0][0])

        print('\n\nTrainval classes ({}): {}'.format(len(self.trainval_class_names), self.trainval_class_names))
        print('\n\nTest seen classes ({}): {}'.format(len(self.testnames_seen), self.testnames_seen))
        print('\n\nTest unseen classes ({}): {}'.format(len(self.testnames_unseen), self.testnames_unseen))

        self.test_res['zsl_trainval'] = self.trainval_class_names
        self.test_res['zsl_test_seen'] = self.testnames_seen
        self.test_res['zsl_test_unseen'] = self.testnames_unseen
        self.test_res['zsl_common_unseen'] = self.common_unseen





        self.data = {}
        self.data['train_seen'] = {}
        self.data['train_seen']['resnet_features'] = train_feature
        self.data['train_seen']['labels']= train_label


        self.data['train_unseen'] = {}
        self.data['train_unseen']['resnet_features'] = None
        self.data['train_unseen']['labels'] = None

        self.data['test_seen'] = {}
        self.data['test_seen']['resnet_features'] = test_seen_feature
        self.data['test_seen']['labels'] = test_seen_label

        self.data['test_unseen'] = {}
        self.data['test_unseen']['resnet_features'] = test_unseen_feature
        self.data['test_unseen']['labels'] = test_unseen_label
