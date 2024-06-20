import torch
import torch.nn.functional as F

#new change
import pickle
import scipy.io as sio
import numpy as np
import wandb


def val_gzsl(test_X, test_label, target_classes,in_package, all_names=None, common_unseen=None, testing = False, bias = 0):

    # testing = False indicates evaluation only for non-unseen classes

    batch_size = in_package['batch_size']
    model = in_package['model']
    device = in_package['device']
    with torch.no_grad():
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, batch_size):

            end = min(ntest, start+batch_size)

            input = test_X[start:end].to(device)
            
            out_package = model(input)
            
            output = out_package['S_pp']
            output[:,target_classes] = output[:,target_classes]+bias
            predicted_label[start:end] = torch.argmax(output.data, 1)

            start = end

        acc = compute_per_class_acc_gzsl(test_label, predicted_label, target_classes, in_package, testing, all_names, common_unseen)
        return acc


def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size()).fill_(-1)
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i    

    return mapped_label


def val_zs_gzsl(test_X, test_label, unseen_classes,in_package, all_names=None, common_unseen=None, testing = False, bias = 0):
    batch_size = in_package['batch_size']
    model = in_package['model']
    device = in_package['device']
    with torch.no_grad():
        start = 0
        ntest = test_X.size()[0]
        predicted_label_gzsl = torch.LongTensor(test_label.size())
        predicted_label_zsl = torch.LongTensor(test_label.size())
        predicted_label_zsl_t = torch.LongTensor(test_label.size())
        for i in range(0, ntest, batch_size):

            end = min(ntest, start+batch_size)

            input = test_X[start:end].to(device)
            
            out_package = model(input)
            output = out_package['S_pp']
            
            output_t = output.clone()
            output_t[:,unseen_classes] = output_t[:,unseen_classes]+torch.max(output)+1
            predicted_label_zsl[start:end] = torch.argmax(output_t.data, 1)
            predicted_label_zsl_t[start:end] = torch.argmax(output.data[:,unseen_classes], 1) 
            
            output[:,unseen_classes] = output[:,unseen_classes]+bias
            predicted_label_gzsl[start:end] = torch.argmax(output.data, 1)
            
            
            start = end

        # acc_gzsl = compute_per_class_acc_gzsl(test_label, predicted_label_gzsl, unseen_classes, in_package, all_names, common_unseen)
        # acc_zs = compute_per_class_acc_gzsl(test_label, predicted_label_zsl, unseen_classes, in_package)
        # acc_zs_t = compute_per_class_acc(map_label(test_label, unseen_classes), predicted_label_zsl_t, unseen_classes.size(0), all_names, common_unseen)
        
        # return acc_gzsl,acc_zs_t


        #new change
        if testing:
            testclass_names = []
            # wandb.log({'\n\nunseen classes': unseen_classes})
            numpy_unseen = unseen_classes.cpu().numpy()
            # target_classes have already 0-indexed labels
            for i in numpy_unseen:
                testclass_names.append(all_names[i][0][0])
            acc_gzsl, acc_common_unseen, classwise_accs, classwise_accs_common_unseen = compute_per_class_acc_gzsl(test_label, predicted_label_gzsl, unseen_classes, in_package, testing, all_names, common_unseen)
            acc_zs_t, acc_common_unseen_zs_t, classwise_accs_zs_t, classwise_accs_common_unseen_zs_t = compute_per_class_acc(map_label(test_label, unseen_classes), predicted_label_zsl_t, unseen_classes, unseen_classes.size(0), all_names, common_unseen)

            return acc_gzsl, acc_common_unseen, classwise_accs, classwise_accs_common_unseen, acc_zs_t, acc_common_unseen_zs_t, classwise_accs_zs_t, classwise_accs_common_unseen_zs_t 


            # acc_zs = compute_per_class_acc_gzsl(test_label, predicted_label_zsl, unseen_classes, in_package)
        

def compute_per_class_acc(test_label, predicted_label, target_classes, nclass, all_names, common_unseen):
    ############################### NOTE#################################
    # test_label does not contain true labels, but mapped labels in range(0, target_classes.size(0)). Hence, we need to construct the orderedLabel_to_class with the help of target_classes and not test_labels
    #####################################################################

    #new change
    orderedLabel_to_class = {}
    # we can do this because in image_util.map_label, target_classes have classes in a unique order. And test_label with get labels in accordance with the order of classes in target_classes. So first class in target_classes will have labels as 0 in test_label, second class in target_classes will have labels as 1 in test_label, and so on.

    #new change
    all_labels = target_classes.cpu().numpy()
    # print('ZSL unique unseen labels: ', all_labels)

    for idx, l in enumerate(all_labels):
        # at 0-indexed index, the apt class name is stored
        orderedLabel_to_class[idx] = all_names[l][0][0]
    # print('Test classes ordered: ', orderedLabel_to_class)

    acc_per_class = torch.FloatTensor(nclass).fill_(0)
    classwise_accs = {}   #new change

    # for i in range(nclass):
    #     idx = (test_label == i)
    #     acc_per_class[i] = torch.sum(test_label[idx]==predicted_label[idx]).float() / torch.sum(idx).float()
    # return acc_per_class.mean().item()


    #new change
    for i in range(nclass):
            # can run a loop on nclass because test_label is already mapped in range(0, target_classes.size(0))
            idx = (test_label == i)
            # acc_per_class[i] = torch.sum(test_label[idx]==predicted_label[idx]) / torch.sum(idx)
            acc_per_class[i] = float(torch.sum(test_label[idx]==predicted_label[idx]))
            acc_per_class[i] = acc_per_class[i] / torch.sum(idx)
            classwise_accs[orderedLabel_to_class[i]] = acc_per_class[i].item()  #new change
                # i will get values 0, 1, 2,....nclass - 1, and in the same order we have got apt labels in orderedLabel_to_class, which was also 0-indexed - so values should be correct

    #new change
    acc_common_unseen = torch.FloatTensor(len(common_unseen)).fill_(0)
    classwise_accs_common_unseen = {k:v for k, v in classwise_accs.items() if k in common_unseen}
    common_unseen_idxs = [i for i,(k,v) in enumerate(classwise_accs.items()) if k in common_unseen]
    # print('common unseen: ',len(classwise_accs_common_unseen))
    acc_common_unseen = acc_per_class[common_unseen_idxs]
    # print("zsl per class: ", classwise_accs)
    # print("zsl per common unseen class: ", classwise_accs_common_unseen)
          
    return acc_per_class.mean(), acc_common_unseen.mean(), classwise_accs, classwise_accs_common_unseen
    

def compute_per_class_acc_gzsl(test_label, predicted_label, target_classes, in_package, testing=False, all_names=None, common_unseen=None):

    #new change
    device = in_package['device']
    per_class_accuracies = torch.zeros(target_classes.size()[0]).float().to(device).detach()
    predicted_label = predicted_label.to(device)
    if testing == False:
        for i in range(target_classes.size()[0]):

            is_class = test_label == target_classes[i]

            per_class_accuracies[i] = torch.div((predicted_label[is_class]==test_label[is_class]).sum().float(),is_class.sum().float())
        return per_class_accuracies.mean().item()

    orderedLabel_to_class = {}
    all_labels = np.unique(test_label.cpu().numpy())
    # print('ZSL unique unseen labels: ', all_labels)
    # print(type(all_labels))
    # print(torch.is_tensor(all_labels))
    for i in all_labels:
        orderedLabel_to_class[i] = all_names[i][0][0]
    # print('Test classes ordered: ', orderedLabel_to_class)

    #new change
    nclass = target_classes.size()[0]
    acc_per_class = torch.FloatTensor(nclass).fill_(0)
    classwise_accs = {}
    count = 0 

    for i in target_classes:
        idx = (test_label == i)
        acc_per_class[count] = float(torch.sum(test_label[idx]==predicted_label[idx]))
        acc_per_class[count] = acc_per_class[count] / torch.sum(idx)
        classwise_accs[orderedLabel_to_class[i.item()]] = acc_per_class[count].item()  #new change
        count = count + 1




    #new change
    acc_common_unseen = torch.FloatTensor(len(common_unseen)).fill_(0)
    classwise_accs_common_unseen = {k:v for k, v in classwise_accs.items() if k in common_unseen}
    common_unseen_idxs = [i for i,(k,v) in enumerate(classwise_accs.items()) if k in common_unseen]
    # print('common unseen: ',len(classwise_accs_common_unseen))
    acc_common_unseen = acc_per_class[common_unseen_idxs]
    # print("zsl per class: ", classwise_accs)
    # print("zsl per common unseen class: ", classwise_accs_common_unseen)           
    # new change - added last two items
    return acc_per_class.mean(), acc_common_unseen.mean(), classwise_accs, classwise_accs_common_unseen     


#new change
def get_names(config=None):
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

def eval_zs_gzsl(dataloader,model,device,bias_seen=0, bias_unseen=0, batch_size=50, testing = False, opt = None):
    model.eval()
    # print('bias_seen {} bias_unseen {}'.format(bias_seen,bias_unseen))
    
    #new change - get the split_info in final results

    if opt.cq:
        splits_folder = opt.work_dir + opt.dataset + '/' + 'split_info_lr' + str(opt.al_lr) + '_cq' + str(opt.cq) + '/'  
    else:
        splits_folder = opt.work_dir + opt.dataset + '/' + 'split_info_lr' + str(opt.al_lr) + '/' 
    pklfile = splits_folder + 'u_split' + str(opt.sn) + '_' + 'split_info_' + opt.dataset + '.pickle'
    res = open(pklfile, 'rb')
    final_results = pickle.load(res)
    # print('Split info: \n')
    # print(final_results)        

    # at actual run, take a parameter to get common_unssen list
    # self.common_unseen = ['sheep', 'dolphin', 'bat', 'seal', 'blue+whale']
    #new change
    common_unseen = final_results['common_unseen']
    # print('Common unseen test classes ({}): {}'.format(len(self.common_unseen), self.common_unseen))
    all_names = get_names(opt)
    
    
    test_seen_feature = dataloader.data['test_seen']['resnet_features']
    test_seen_label = dataloader.data['test_seen']['labels'].to(device)
    
    test_unseen_feature = dataloader.data['test_unseen']['resnet_features']
    test_unseen_label = dataloader.data['test_unseen']['labels'].to(device)
    
    seenclasses = dataloader.seenclasses
    unseenclasses = dataloader.unseenclasses
    
    batch_size = batch_size

    H = 0
    H_new = 0
    
    in_package = {'model':model,'device':device, 'batch_size':batch_size}
    
    with torch.no_grad():
        #new change - adding last 2 params
        acc_seen = val_gzsl(test_seen_feature, test_seen_label, seenclasses, in_package, all_names, common_unseen, testing=False, bias=bias_seen)
        # acc_novel,acc_zs = val_zs_gzsl(test_unseen_feature, test_unseen_label, unseenclasses, in_package, all_names, common_unseen, testing, bias = bias_unseen)
        acc_unseen, acc_common_unseen_gzsl, classwise_accs_gzsl, classwise_accs_common_unseen_gzsl, acc_zs_t, acc_common_unseen_zs_t, classwise_accs_zs_t, classwise_accs_common_unseen_zs_t = val_zs_gzsl(test_unseen_feature, test_unseen_label, unseenclasses, in_package, all_names, common_unseen, testing, bias = bias_unseen)

    if (acc_seen+acc_unseen)>0:
        H = (2*acc_seen*acc_unseen) / (acc_seen+acc_unseen)
    if (acc_seen+acc_common_unseen_gzsl)>0:
        H_new = (2*acc_seen*acc_common_unseen_gzsl) / (acc_seen+acc_common_unseen_gzsl)

        
    # return acc_seen, acc_novel, H, acc_zs
    return acc_seen, acc_unseen, H, H_new, acc_common_unseen_gzsl, classwise_accs_gzsl, classwise_accs_common_unseen_gzsl, acc_zs_t, acc_common_unseen_zs_t, classwise_accs_zs_t, classwise_accs_common_unseen_zs_t
    

def val_gzsl_k(k,test_X, test_label, target_classes,in_package,bias = 0,is_detect=False):
    batch_size = in_package['batch_size']
    model = in_package['model']
    device = in_package['device']
    n_classes = in_package["num_class"]
    
    with torch.no_grad():
        start = 0
        ntest = test_X.size()[0]
        test_label = F.one_hot(test_label, num_classes=n_classes)
        predicted_label = torch.LongTensor(test_label.size()).fill_(0).to(test_label.device)
        for i in range(0, ntest, batch_size):

            end = min(ntest, start+batch_size)

            input = test_X[start:end].to(device)
            
            out_package = model(input)
            
            output = out_package['S_pp']
            output[:,target_classes] = output[:,target_classes]+bias
            _,idx_k = torch.topk(output,k,dim=1)
            if is_detect:
                assert k == 1
                detection_mask=in_package["detection_mask"]
                predicted_label[start:end] = detection_mask[torch.argmax(output.data, 1)]
            else:
                predicted_label[start:end] = predicted_label[start:end].scatter_(1,idx_k,1)
            start = end
        
        acc = compute_per_class_acc_gzsl_k(test_label, predicted_label, target_classes, in_package)
        return acc


def val_zs_gzsl_k(k,test_X, test_label, unseen_classes,in_package,bias = 0,is_detect=False):
    batch_size = in_package['batch_size']
    model = in_package['model']
    device = in_package['device']
    n_classes = in_package["num_class"]
    with torch.no_grad():
        start = 0
        ntest = test_X.size()[0]
        
        test_label_gzsl = F.one_hot(test_label, num_classes=n_classes)
        predicted_label_gzsl = torch.LongTensor(test_label_gzsl.size()).fill_(0).to(test_label.device)
        
        predicted_label_zsl = torch.LongTensor(test_label.size())
        predicted_label_zsl_t = torch.LongTensor(test_label.size())
        for i in range(0, ntest, batch_size):

            end = min(ntest, start+batch_size)

            input = test_X[start:end].to(device)
            
            out_package = model(input)
            output = out_package['S_pp']
            
            output_t = output.clone()
            output_t[:,unseen_classes] = output_t[:,unseen_classes]+torch.max(output)+1
            predicted_label_zsl[start:end] = torch.argmax(output_t.data, 1)
            predicted_label_zsl_t[start:end] = torch.argmax(output.data[:,unseen_classes], 1) 
            
            output[:,unseen_classes] = output[:,unseen_classes]+bias
            _,idx_k = torch.topk(output,k,dim=1)
            if is_detect:
                assert k == 1
                detection_mask=in_package["detection_mask"]
                predicted_label_gzsl[start:end] = detection_mask[torch.argmax(output.data, 1)]
            else:
                predicted_label_gzsl[start:end] = predicted_label_gzsl[start:end].scatter_(1,idx_k,1)
            
            start = end
        
        acc_gzsl = compute_per_class_acc_gzsl_k(test_label_gzsl, predicted_label_gzsl, unseen_classes, in_package)
        #print('acc_zs: {} acc_zs_t: {}'.format(acc_zs,acc_zs_t))
        return acc_gzsl,-1


def compute_per_class_acc_k(test_label, predicted_label, nclass):
    acc_per_class = torch.FloatTensor(nclass).fill_(0)
    for i in range(nclass):
        idx = (test_label == i)
        acc_per_class[i] = torch.sum(test_label[idx]==predicted_label[idx]).float() / torch.sum(idx).float()
    return acc_per_class.mean().item()
    

def compute_per_class_acc_gzsl_k(test_label, predicted_label, target_classes, in_package):
    device = in_package['device']
    per_class_accuracies = torch.zeros(target_classes.size()[0]).float().to(device).detach()

    predicted_label = predicted_label.to(device)
    
    hit = test_label*predicted_label
    for i in range(target_classes.size()[0]):

        target = target_classes[i]
        n_pos = torch.sum(hit[:,target])
        n_gt = torch.sum(test_label[:,target])
        per_class_accuracies[i] = torch.div(n_pos.float(),n_gt.float())
        #pdb.set_trace()
    return per_class_accuracies.mean().item()


def eval_zs_gzsl_k(k,dataloader,model,device,bias_seen,bias_unseen,is_detect=False):
    model.eval()
    print('bias_seen {} bias_unseen {}'.format(bias_seen,bias_unseen))
    test_seen_feature = dataloader.data['test_seen']['resnet_features']
    test_seen_label = dataloader.data['test_seen']['labels'].to(device)
    
    test_unseen_feature = dataloader.data['test_unseen']['resnet_features']
    test_unseen_label = dataloader.data['test_unseen']['labels'].to(device)
    
    seenclasses = dataloader.seenclasses
    unseenclasses = dataloader.unseenclasses
    
    batch_size = 100
    n_classes = dataloader.ntrain_class+dataloader.ntest_class
    in_package = {'model':model,'device':device, 'batch_size':batch_size,'num_class':n_classes}
    
    if is_detect:
        print("Measure novelty detection k: {}".format(k))
        
        detection_mask = torch.zeros((n_classes,n_classes)).long().to(dataloader.device)
        detect_label = torch.zeros(n_classes).long().to(dataloader.device)
        detect_label[seenclasses]=1
        detection_mask[seenclasses,:] = detect_label
        
        detect_label = torch.zeros(n_classes).long().to(dataloader.device)
        detect_label[unseenclasses]=1
        detection_mask[unseenclasses,:]=detect_label
        in_package["detection_mask"]=detection_mask
    
    with torch.no_grad():
        acc_seen = val_gzsl_k(k,test_seen_feature, test_seen_label, seenclasses, in_package,bias=bias_seen,is_detect=is_detect)
        acc_novel,acc_zs = val_zs_gzsl_k(k,test_unseen_feature, test_unseen_label, unseenclasses, in_package,bias = bias_unseen,is_detect=is_detect)

    if (acc_seen+acc_novel)>0:
        H = (2*acc_seen*acc_novel) / (acc_seen+acc_novel)
    else:
        H = 0
        
    return acc_seen, acc_novel, H, acc_zs
