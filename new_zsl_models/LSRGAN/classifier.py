import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import util
from sklearn.preprocessing import MinMaxScaler 
import sys
import torch.nn.functional as F
import scipy.io as sio
#new change
import pickle

class CLASSIFIER:
    # train_Y is interger 
    def __init__(self, _train_X, _train_Y,  _nclass, _input_dim, _cuda, _lr=0.001, _beta1=0.5, _nepoch=20, _batch_size=100, pretrain_classifer='', opt = None):
        self.train_X =  _train_X 
        self.train_Y = _train_Y 
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.nclass = _nclass
        self.input_dim = _input_dim
        self.cuda = _cuda
        self.model =  LINEAR_LOGSOFTMAX(self.input_dim, self.nclass)
        self.model.apply(util.weights_init)
        self.criterion = nn.NLLLoss()
        
        self.input = torch.FloatTensor(_batch_size, self.input_dim) 
        self.label = torch.LongTensor(_batch_size) 
        
        self.lr = _lr
        self.beta1 = _beta1
        # setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=_lr, betas=(_beta1, 0.999))

        if self.cuda:
            self.model.cuda()
            self.criterion.cuda()
            self.input = self.input.cuda()
            self.label = self.label.cuda()

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size()[0]

        #new change
        self.all_names = None
        self.opt = opt

        #new change - get the split_info in final results
        splits_folder = '/home/gdata/sandipan/BTP2021/' + opt.dataset + '/' + 'split_info_lr' + str(opt.al_lr) + '/'
        pklfile = splits_folder + 'u_split' + str(opt.sn) + '_' + 'split_info_' + opt.dataset + '.pickle'
        res = open(pklfile, 'rb')
        final_results = pickle.load(res)
        print('Split info: \n')
        print(final_results)        

        # at actual run, take a parameter to get common_unssen list
        # self.common_unseen = ['sheep', 'dolphin', 'bat', 'seal', 'blue+whale']
        #new change
        self.common_unseen = final_results['common_unseen']
        print('Common unseen test classes ({}): {}'.format(len(self.common_unseen), self.common_unseen))


        if self.opt.al_seed == 'original':
          matcontent = sio.loadmat(self.opt.dataroot + "/" + self.opt.dataset + "/" + self.opt.class_embedding + "_splits.mat")
        else:
            #new change - file name change
            split_name = 's' + str(opt.sn)
            matcontent = sio.loadmat(self.opt.dataroot + '/' + self.opt.dataset + '/' + self.opt.class_embedding + '_splits_' + self.opt.dataset + '_al_' + self.opt.al_seed + '_' + split_name + '.mat')

            print(self.opt.dataroot + '/' + self.opt.dataset + '/' + self.opt.class_embedding + '_splits_' + self.opt.dataset + '_al_' + self.opt.al_seed + '_' + split_name + '.mat')
        
        # numpy array index starts from 0, matlab starts from 1
        self.all_names = matcontent['allclasses_names']



        if pretrain_classifer == '':
            # as per given parameters to script file, always this branch will tun, not else part
            self.fit()
        else:
            self.model.load_state_dict(torch.load(pretrain_classifier))
    

    def fit(self):
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
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0) , torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            # from index start to index end-1
            return self.train_X[start:end], self.train_Y[start:end]

    # test_label is integer 
    def val(self, test_X, test_label, target_classes): 
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                with torch.no_grad():
                    output = self.model(Variable(test_X[start:end].cuda())) 
            else:
                with torch.no_grad():
                    output = self.model(Variable(test_X[start:end])) 
            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end

        acc = self.compute_per_class_acc(util.map_label(test_label, target_classes), predicted_label, target_classes.size(0))
        return acc

    def confidence_cal (self, seen_vis, seen_label , unseen_vis, unseen_label, seenclasses, unseenclasses):
        unseen_dic = {}
        seen_dic = {}

        for i in unseenclasses:
            idx = (i == unseen_label)
            with torch.no_grad():
                output = self.model(Variable(unseen_vis[idx].cuda()))
            unseen_dic[str(i.item())] = F.softmax(output.mean(axis=0), dim=0).data.cpu().numpy()
        
        for i in seenclasses:
            idx = (i == seen_label)
            with torch.no_grad():
                output = self.model(Variable(seen_vis[idx].cuda()))
            seen_dic[str(i.item())] = F.softmax(output.mean(axis=0), dim= 0).data.cpu().numpy()
        
        return unseen_dic, seen_dic

    def output_max_unseen(self, output_data, test_label):
        dic = {}
        for i in test_label:
            value = i.item()
            if value not in dic:
                dic[value] = True

        class_reference = torch.ones([output_data.shape[0], output_data.shape[1]])*float('-inf')
        for i in dic:
            class_reference[:, i] = output_data[:, i]

        return class_reference

    def val_gzsl(self, test_X, test_label, target_classes, testing = False, given_opt = None, log_dir = None, unseen = False): 
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                with torch.no_grad():
                    output = self.model(Variable(test_X[start:end].cuda())) 
            else:
                with torch.no_grad():
                    output = self.model(Variable(test_X[start:end])) 

            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end

        #new change
        if testing == True:
            testclass_names = []
            numpy_unseen = target_classes.numpy()
            # target_classes have already 0-indexed labels
            for i in numpy_unseen:
                testclass_names.append(self.all_names[i][0][0])
            acc, acc_common_unseen, classwise_accs, classwise_accs_common_unseen = self.compute_per_class_acc_unseen(test_label, predicted_label, target_classes.size(0), testing, given_opt, testclass_names, log_dir, unseen)
            return acc, acc_common_unseen, classwise_accs, classwise_accs_common_unseen

        else:
            acc = self.compute_per_class_acc_unseen(test_label, predicted_label, target_classes.size(0), testing, given_opt, log_dir, unseen)
            return acc


        
    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):
        acc_per_class = torch.FloatTensor(target_classes.size(0)).fill_(0)
        counter = 0
        for i in target_classes:
            idx = (test_label == i)
            acc_per_class[counter] = float(torch.sum(test_label[idx]==predicted_label[idx]))
            acc_per_class[counter] = acc_per_class[counter] /  torch.sum(idx)
            counter = counter + 1 

        return acc_per_class.mean()

    def unesen_val(self, test_X, test_label, target_classes, testing = False, given_opt = None, log_dir = None, unseen = False): 
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                with torch.no_grad():
                    output = self.model(Variable(test_X[start:end].cuda())) 
            else:
                with torch.no_grad():
                    output = self.model(Variable(test_X[start:end])) 

            _, predicted_label[start:end] = torch.max(self.output_max_unseen(output.data, test_label), 1)
            start = end

        #new change
        if testing == True:
            testclass_names = []
            numpy_unseen = target_classes.numpy()
            # target_classes have already 0-indexed labels
            for i in numpy_unseen:
                testclass_names.append(self.all_names[i][0][0])
            acc, acc_common_unseen, classwise_accs, classwise_accs_common_unseen = self.compute_per_class_acc_unseen(test_label, predicted_label, target_classes.size(0), testing, given_opt, testclass_names, log_dir, unseen)
            return acc, acc_common_unseen, classwise_accs, classwise_accs_common_unseen

        else:
            acc = self.compute_per_class_acc_unseen(test_label, predicted_label, target_classes.size(0), testing, given_opt, log_dir, unseen)
            return acc

    def compute_per_class_acc_unseen(self, test_label, predicted_label, nclass, testing = False, opt = None, testclass_names = None, log_dir = None, unseen = False):
        # Written by me
        dic = {}
        for i in test_label:
            value = i.item()
            if value not in dic:
                dic[value] = True   #gets all the unique test_label as keys

        #new change
        orderedLabel_to_class = {}
        if testing == True and unseen == True:
            all_labels = np.unique(test_label.numpy())

            print('ZSL unique unseen labels: ', all_labels)
            for i in all_labels:
                orderedLabel_to_class[i] = self.all_names[i][0][0]
            print('Test classes ordered: ', orderedLabel_to_class)
            with open(log_dir, 'a') as f:
                f.write('ZSL unique unseen labels: ' + str(all_labels.tolist())+'\n')
                f.write('Test classes ordered: ' + str(orderedLabel_to_class)+'\n\n')


        acc_per_class = torch.FloatTensor(nclass).fill_(0)
        classwise_accs = {}   #new change

        count = 0 
        for i in dic:
            idx = (test_label == i)
            acc_per_class[count] = float(torch.sum(test_label[idx]==predicted_label[idx]))
            acc_per_class[count] = acc_per_class[count] / torch.sum(idx)
            if testing == True and unseen == True:
                classwise_accs[orderedLabel_to_class[i]] = acc_per_class[count].item()  #new change


            count = count + 1

        #new change
        if testing == True and unseen == True:
            acc_common_unseen = torch.FloatTensor(len(self.common_unseen)).fill_(0)
            classwise_accs_common_unseen = {k:v for k, v in classwise_accs.items() if k in self.common_unseen}
            common_unseen_idxs = [i for i,(k,v) in enumerate(classwise_accs.items()) if k in self.common_unseen]
            print('common unseen: ',len(classwise_accs_common_unseen))
            acc_common_unseen = acc_per_class[common_unseen_idxs]
            print("zsl per class: ", classwise_accs)
            print("zsl per common unseen class: ", classwise_accs_common_unseen)
            with open(log_dir, 'a') as f:
                f.write('common_unseen: '+str(len(classwise_accs_common_unseen))+'\n')
                f.write("zsl per class: "+ str(classwise_accs)+'\n')
                f.write("zsl per common unseen class: "+ str(classwise_accs_common_unseen)+'\n\n')            
            # new change - added last two items
            return acc_per_class.mean(), acc_common_unseen.mean(), classwise_accs, classwise_accs_common_unseen


        print("zsl per class: ", acc_per_class);return acc_per_class.mean() 



    def compute_per_class_acc(self, test_label, predicted_label, nclass):
        acc_per_class = torch.FloatTensor(nclass).fill_(0)
        for i in range(nclass):
            idx = (test_label == i)
            acc_per_class[i] = float(torch.sum(test_label[idx]==predicted_label[idx]))
            acc_per_class[i] = acc_per_class[i] / torch.sum(idx)
        return acc_per_class.mean() 

class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)
    def forward(self, x): 
        o = self.logic(self.fc(x))
        return o  
