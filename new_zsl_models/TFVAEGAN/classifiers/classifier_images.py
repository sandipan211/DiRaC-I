#author: akshitac8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import datasets.image_util as util
from sklearn.preprocessing import MinMaxScaler 
import sys
import copy
import pdb
#new change
import pickle
import scipy.io as sio

class CLASSIFIER:
    # train_Y is interger 
    def __init__(self, _train_X, _train_Y, data_loader, _nclass, _cuda, _lr=0.001, _beta1=0.5, _nepoch=20, _batch_size=100, generalized=True, netDec=None, dec_size=4096, dec_hidden_size=4096, testing = False, opt = None):
        self.train_X =  _train_X.clone() 
        self.train_Y = _train_Y.clone() 
        self.test_seen_feature = data_loader.test_seen_feature.clone()
        self.test_seen_label = data_loader.test_seen_label 
        self.test_unseen_feature = data_loader.test_unseen_feature.clone()
        self.test_unseen_label = data_loader.test_unseen_label 
        self.seenclasses = data_loader.seenclasses
        self.unseenclasses = data_loader.unseenclasses
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.nclass = _nclass
        self.input_dim = _train_X.size(1)
        self.cuda = _cuda
        self.model =  LINEAR_LOGSOFTMAX_CLASSIFIER(self.input_dim, self.nclass)
        self.netDec = netDec

        #new change
        self.all_names = None
        self.opt = opt
        self.testing = testing

        #new change - get the split_info in final results

        if opt.cq:
            splits_folder = '/home/gdata/sandipan/BTP2021/' + opt.dataset + '/' + 'split_info_lr' + str(opt.al_lr) + '_cq' + str(opt.cq) + '/'  
        else:
            splits_folder = '/home/gdata/sandipan/BTP2021/' + opt.dataset + '/' + 'split_info_lr' + str(opt.al_lr) + '/' 
        pklfile = splits_folder + 'u_split' + str(opt.sn) + '_' + 'split_info_' + opt.dataset + '.pickle'
        res = open(pklfile, 'rb')
        final_results = pickle.load(res)
        # print('Split info: \n')
        # print(final_results)        

        # at actual run, take a parameter to get common_unssen list
        # self.common_unseen = ['sheep', 'dolphin', 'bat', 'seal', 'blue+whale']
        #new change
        self.common_unseen = final_results['common_unseen']
        # print('Common unseen test classes ({}): {}'.format(len(self.common_unseen), self.common_unseen))


        if self.opt.al_seed == 'original':
          matcontent = sio.loadmat(self.opt.dataroot + "/" + self.opt.dataset + "/" + self.opt.class_embedding + "_splits.mat")
        else:
            #new change - file name change
            if opt.cq:
                split_name = 's' + str(opt.sn) + '_cq' + str(opt.cq)
            else:
                split_name = 's' + str(opt.sn)
            matcontent = sio.loadmat(self.opt.dataroot + '/' + self.opt.dataset + '/' + self.opt.class_embedding + '_splits_' + self.opt.dataset + '_al_' + self.opt.al_seed + '_' + split_name + '.mat')

            print(self.opt.dataroot + '/' + self.opt.dataset + '/' + self.opt.class_embedding + '_splits_' + self.opt.dataset + '_al_' + self.opt.al_seed + '_' + split_name + '.mat')
        
        # numpy array index starts from 0, matlab starts from 1
        self.all_names = matcontent['allclasses_names']


        if self.netDec:
            self.netDec.eval()
            self.input_dim = self.input_dim + dec_size
            self.input_dim += dec_hidden_size
            self.model =  LINEAR_LOGSOFTMAX_CLASSIFIER(self.input_dim, self.nclass)
            self.train_X = self.compute_dec_out(self.train_X, self.input_dim)
            self.test_unseen_feature = self.compute_dec_out(self.test_unseen_feature, self.input_dim)
            self.test_seen_feature = self.compute_dec_out(self.test_seen_feature, self.input_dim)
        self.model.apply(util.weights_init)
        self.criterion = nn.NLLLoss()
        self.input = torch.FloatTensor(_batch_size, self.input_dim) 
        self.label = torch.LongTensor(_batch_size) 
        self.lr = _lr
        self.beta1 = _beta1
        self.optimizer = optim.Adam(self.model.parameters(), lr=_lr, betas=(_beta1, 0.999))
        if self.cuda:
            self.model.cuda()
            self.criterion.cuda()
            self.input = self.input.cuda()
            self.label = self.label.cuda()
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size()[0]
        if generalized:
            self.acc_seen, self.acc_unseen, self.H, self.epoch, self.H_new, self.acc_common_unseen_gzsl, self.classwise_accs_gzsl, self.classwise_accs_common_unseen_gzsl= self.fit()
            # print('Final: acc_seen=%.4f, acc_unseen=%.4f, h=%.4f, con h=%.4f, acc common unseen=%.4f ' % (self.acc_seen, self.acc_unseen, self.H, self.H_new, self.acc_common_unseen_gzsl))
            # print(self.classwise_accs_gzsl)
            # print(self.classwise_accs_common_unseen_gzsl)
        else:
            self.acc,self.best_model, self.acc_common_unseen, self.classwise_accs, self.classwise_accs_common_unseen = self.fit_zsl() 
            #print('acc=%.4f' % (self.acc))



    def fit_zsl(self):
        best_acc = 0
        mean_loss = 0
        last_loss_epoch = 1e8 
        best_model = copy.deepcopy(self.model.state_dict())

        #new change
        best_acc_common_unseen = 0
        best_classwise_accs = {}
        best_classwise_accs_common_unseen = {}

        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):      
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size) 
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)
                   
                inputv = Variable(self.input)
                labelv = Variable(self.label)
                output = self.model(inputv)
                loss = self.criterion(output, labelv)
                mean_loss += loss.data.item()
                loss.backward()
                self.optimizer.step()
                #print('Training classifier loss= ', loss.data[0])
            acc, acc_common_unseen, classwise_accs, classwise_accs_common_unseen = self.val(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses, unseen = True)
            #print('acc %.4f' % (acc))
            if acc_common_unseen > best_acc_common_unseen:
                # best on the basis of only common unseen classes
                best_acc = acc
                best_acc_common_unseen = acc_common_unseen
                best_classwise_accs = classwise_accs
                best_classwise_accs_common_unseen = classwise_accs_common_unseen
                best_model = copy.deepcopy(self.model.state_dict())

        return best_acc, best_model, best_acc_common_unseen, best_classwise_accs, best_classwise_accs_common_unseen
        
    def fit(self):
        best_H = 0
        best_seen = 0
        best_unseen = 0

        #new change
        best_H_new = 0
        best_acc_common_unseen_gzsl = 0
        best_classwise_accs_gzsl = {}
        best_classwise_accs_common_unseen_gzsl = {}


        out = []
        best_model = copy.deepcopy(self.model.state_dict())
        # early_stopping = EarlyStopping(patience=20, verbose=True)
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):      
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size) 
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)
                   
                inputv = Variable(self.input)
                labelv = Variable(self.label)
                output = self.model(inputv)
                loss = self.criterion(output, labelv)
                loss.backward()
                self.optimizer.step()
            acc_seen = 0
            acc_unseen = 0

            #new change
            acc_common_unseen = 0 
            H_new = 0
            acc_seen = self.val_gzsl(self.test_seen_feature, self.test_seen_label, self.seenclasses)
            acc_unseen, acc_common_unseen, classwise_accs, classwise_accs_common_unseen = self.val_gzsl(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses, unseen = True)
            H = 2*acc_seen*acc_unseen / (acc_seen+acc_unseen)
            H_new = 2*acc_seen*acc_common_unseen / (acc_seen+acc_common_unseen)

            if H_new > best_H_new:
                # seeing results only depending on common unseen classes - hence using H_new
                best_seen = acc_seen
                best_unseen = acc_unseen
                best_H = H
                best_H_new = H_new
                best_acc_common_unseen_gzsl = acc_common_unseen
                best_classwise_accs_gzsl = classwise_accs
                best_classwise_accs_common_unseen_gzsl = classwise_accs_common_unseen

        return best_seen, best_unseen, best_H, epoch, best_H_new, best_acc_common_unseen_gzsl, best_classwise_accs_gzsl, best_classwise_accs_common_unseen_gzsl
                     
    def next_batch(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
            # shuffle the data
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            #print(start, end)
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0) , torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            #print(start, end)
            # from index start to index end-1
            return self.train_X[start:end], self.train_Y[start:end]

    #new change - added param
    def val_gzsl(self, test_X, test_label, target_classes, unseen = False): 
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                # new change - replace volatile = true with torch.no_grad()
                with torch.no_grad():
                    inputX = Variable(test_X[start:end].cuda())
            else:
                # new change - replace volatile = true with torch.no_grad()
                with torch.no_grad():
                    inputX = Variable(test_X[start:end])

            output = self.model(inputX)  
            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end


        #new change
        if self.testing == True and unseen == True:
            testclass_names = []
            numpy_unseen = target_classes.numpy()
            # target_classes have already 0-indexed labels
            for i in numpy_unseen:
                testclass_names.append(self.all_names[i][0][0])
            acc, acc_common_unseen, classwise_accs, classwise_accs_common_unseen = self.compute_per_class_acc_gzsl(test_label, predicted_label, target_classes, testclass_names, unseen)

            return acc, acc_common_unseen, classwise_accs, classwise_accs_common_unseen

        acc = self.compute_per_class_acc_gzsl(test_label, predicted_label, target_classes)
        return acc


    #new change - added param
    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes, testclass_names = None, unseen = False):

        #new change
        orderedLabel_to_class = {}

        if self.testing == True and unseen == True:
            all_labels = np.unique(test_label.numpy())
            # print('ZSL unique unseen labels: ', all_labels)
            # print(type(all_labels))
            # print(torch.is_tensor(all_labels))
            for i in all_labels:
                orderedLabel_to_class[i] = self.all_names[i][0][0]
            # print('Test classes ordered: ', orderedLabel_to_class)

        #new change
        nclass = target_classes.size(0)
        acc_per_class = torch.FloatTensor(nclass).fill_(0)
        classwise_accs = {}
        count = 0 
        for i in target_classes:
            idx = (test_label == i)
            acc_per_class[count] = float(torch.sum(test_label[idx]==predicted_label[idx]))
            acc_per_class[count] = acc_per_class[count] / torch.sum(idx)
            if self.testing == True and unseen == True:
                classwise_accs[orderedLabel_to_class[i.item()]] = acc_per_class[count].item()  #new change
            count = count + 1

        #new change
        if self.testing == True and unseen == True:
            acc_common_unseen = torch.FloatTensor(len(self.common_unseen)).fill_(0)
            classwise_accs_common_unseen = {k:v for k, v in classwise_accs.items() if k in self.common_unseen}
            common_unseen_idxs = [i for i,(k,v) in enumerate(classwise_accs.items()) if k in self.common_unseen]
            # print('common unseen: ',len(classwise_accs_common_unseen))
            acc_common_unseen = acc_per_class[common_unseen_idxs]
            # print("zsl per class: ", classwise_accs)
            # print("zsl per common unseen class: ", classwise_accs_common_unseen)           
            # new change - added last two items
            return acc_per_class.mean(), acc_common_unseen.mean(), classwise_accs, classwise_accs_common_unseen     

        #new change
        # print("zsl per class: ", acc_per_class)
        return acc_per_class.mean()    

        #old (repo's) implementation
        # acc_per_class = 0
        # for i in target_classes:
        #     idx = (test_label == i)
        #     acc_per_class += torch.sum(test_label[idx]==predicted_label[idx]) / torch.sum(idx)
        # acc_per_class /= target_classes.size(0)
        # return acc_per_class 

    # test_label is integer 
    def val(self, test_X, test_label, target_classes, unseen = False): 
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                # inputX = Variable(test_X[start:end].cuda(), volatile=True)
                #new change
                with torch.no_grad():
                    inputX = Variable(test_X[start:end].cuda())
            else:
                # inputX = Variable(test_X[start:end], volatile=True)
                #new change
                with torch.no_grad():
                    inputX = Variable(test_X[start:end])

            output = self.model(inputX) 
            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end

        #new change 
        acc, acc_common_unseen, classwise_accs, classwise_accs_common_unseen = self.compute_per_class_acc(util.map_label(test_label, target_classes), predicted_label, target_classes.size(0), target_classes, unseen)
        return acc, acc_common_unseen, classwise_accs, classwise_accs_common_unseen

    def compute_per_class_acc(self, test_label, predicted_label, nclass, target_classes, unseen = False):

        ############################### NOTE#################################
        # test_label does not contain true labels, but mapped labels in range(0, target_classes.size(0)). Hence, we need to construct the orderedLabel_to_class with the help of target_classes and not test_labels
        #####################################################################

        #new change
        orderedLabel_to_class = {}
        if self.testing == True and unseen == True:
            # we can do this because in image_util.map_label, target_classes have classes in a unique order. And test_label with get labels in accordance with the order of classes in target_classes. So first class in target_classes will have labels as 0 in test_label, second class in target_classes will have labels as 1 in test_label, and so on.

            #new change
            all_labels = target_classes.numpy()
            # print('ZSL unique unseen labels: ', all_labels)

            for idx, l in enumerate(all_labels):
                # at 0-indexed index, the apt class name is stored
                orderedLabel_to_class[idx] = self.all_names[l][0][0]
            # print('Test classes ordered: ', orderedLabel_to_class)

        acc_per_class = torch.FloatTensor(nclass).fill_(0)
        classwise_accs = {}   #new change

        for i in range(nclass):
            # can run a loop on nclass because test_label is already mapped in range(0, target_classes.size(0))
            idx = (test_label == i)
            # acc_per_class[i] = torch.sum(test_label[idx]==predicted_label[idx]) / torch.sum(idx)
            acc_per_class[i] = float(torch.sum(test_label[idx]==predicted_label[idx]))
            acc_per_class[i] = acc_per_class[i] / torch.sum(idx)
            if self.testing == True and unseen == True:
                classwise_accs[orderedLabel_to_class[i]] = acc_per_class[i].item()  #new change
                # i will get values 0, 1, 2,....nclass - 1, and in the same order we have got apt labels in orderedLabel_to_class, which was also 0-indexed - so values should be correct

                #new change
        if self.testing == True and unseen == True:
            acc_common_unseen = torch.FloatTensor(len(self.common_unseen)).fill_(0)
            classwise_accs_common_unseen = {k:v for k, v in classwise_accs.items() if k in self.common_unseen}
            common_unseen_idxs = [i for i,(k,v) in enumerate(classwise_accs.items()) if k in self.common_unseen]
            # print('common unseen: ',len(classwise_accs_common_unseen))
            acc_common_unseen = acc_per_class[common_unseen_idxs]
            # print("zsl per class: ", classwise_accs)
            # print("zsl per common unseen class: ", classwise_accs_common_unseen)
          
            # new change - added last two items
            return acc_per_class.mean(), acc_common_unseen.mean(), classwise_accs, classwise_accs_common_unseen

        return acc_per_class.mean() 


    def compute_dec_out(self, test_X, new_size):
        start = 0
        ntest = test_X.size()[0]
        new_test_X = torch.zeros(ntest,new_size)
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                # inputX = Variable(test_X[start:end].cuda(), volatile=True)
                with torch.no_grad():
                    inputX = Variable(test_X[start:end].cuda())
            else:
                # inputX = Variable(test_X[start:end], volatile=True)
                with torch.no_grad():
                    inputX = Variable(test_X[start:end])
            feat1 = self.netDec(inputX)
            feat2 = self.netDec.getLayersOutDet()
            new_test_X[start:end] = torch.cat([inputX,feat1,feat2],dim=1).data.cpu()
            start = end
        return new_test_X


class LINEAR_LOGSOFTMAX_CLASSIFIER(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX_CLASSIFIER, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)
    def forward(self, x): 
        o = self.logic(self.fc(x))
        return o
