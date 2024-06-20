# DATASET = 'AWA1' # One of ["AWA1", "AWA2", "APY", "CUB", "SUN"]
USE_CLASS_STANDARTIZATION = True # i.e. equation (9) from the paper
USE_PROPER_INIT = True # i.e. equation (10) from the paper
HOME = '/workspace/arijit_pg/BTP2021/'

import numpy as np; np.random.seed(1)
import torch; torch.manual_seed(1)
import torch.nn as nn
import torch.nn.functional as F
from time import time
from tqdm import tqdm
from scipy import io
from torch.utils.data import DataLoader

#new change
import os
import sys
import pickle
from config import opt

#new change
czsl_folder = None
gzsl_folder = None
pklfile_czsl = None
pklfile_gzsl = None
result_filename_czsl = None
result_filename_gzsl = None
fname = 'cnzsl'

#new change - made separate report folders
gzsl_folder = HOME + 'new_zsl_models/CNZSL/GZSL_al_lr' + str(opt.al_lr) + '/' 
report_folder_gzsl = gzsl_folder +'u_split' + str(opt.sn) + '/'
if not os.path.exists(gzsl_folder):
    os.mkdir(gzsl_folder)
if not os.path.exists(report_folder_gzsl):
    os.mkdir(report_folder_gzsl) 
if opt.cq:
    fstr = report_folder_gzsl + fname + '_gzsl_' + opt.dataset  + '_' + opt.al_seed + '_cq' + str(opt.cq)
else:
    fstr = report_folder_gzsl + fname + '_gzsl_' + opt.dataset  + '_' + opt.al_seed
result_filename_gzsl = fstr + '_reports.txt'
pklfile_gzsl = fstr + '_results.pickle'
gzsl_res_file = open(result_filename_gzsl, 'w')



#new change - made separate report folders
# czsl_folder = '/home/gdata/sandipan/BTP2021/new_zsl_models/TFVAEGAN/tfvaegan-master/CZSL_al_lr' + str(opt.al_lr) + '/' 
czsl_folder = HOME + 'new_zsl_models/CNZSL/CZSL_al_lr' + str(opt.al_lr) + '/' 
report_folder_czsl = czsl_folder +'u_split' + str(opt.sn) + '/'
if not os.path.exists(czsl_folder):
    os.mkdir(czsl_folder)
if not os.path.exists(report_folder_czsl):
    os.mkdir(report_folder_czsl)

if opt.cq:
    fstr = report_folder_czsl + fname + '_czsl_' + opt.dataset  + '_' + opt.al_seed + '_cq' +str(opt.cq)
else:
    fstr = report_folder_czsl + fname + '_czsl_' + opt.dataset  + '_' + opt.al_seed 
result_filename_czsl = fstr + '_reports.txt'
pklfile_czsl = fstr + '_results.pickle'
czsl_res_file = open(result_filename_czsl, 'w')


# all in-training results will be written in result_filename_czsl
sys.stdout = czsl_res_file

#new change
DATASET = opt.dataset
print(f'<=============== Loading data for {DATASET} ===============>')
DEVICE = 'cuda' # Set to 'cpu' if a GPU is not available
DATA_DIR = opt.dataroot + "/" + opt.dataset
# data = io.loadmat(f'{DATA_DIR}/res101.mat')
# attrs_mat = io.loadmat(f'{DATA_DIR}/att_splits.mat')

#new change
data = io.loadmat(DATA_DIR + "/" + opt.image_embedding + ".mat")
feats = data['features'].T.astype(np.float32)
labels = data['labels'].squeeze() - 1 # Using "-1" here and for idx to normalize to 0-index
if opt.al_seed == 'original':
    attrs_mat = io.loadmat(DATA_DIR + "/" + opt.class_embedding + "_splits.mat")
else:
    #new change - file name change
    if opt.cq:
        split_name = 's' + str(opt.sn) + '_cq' + str(opt.cq)
    else:
        split_name = 's' + str(opt.sn)
    attrs_mat = io.loadmat(opt.dataroot + '/' + opt.dataset + '/' + opt.class_embedding + '_splits_' + opt.dataset + '_al_' + opt.al_seed + '_' + split_name + '.mat')
    print('Split path: ')
    print(DATA_DIR + '/' + opt.class_embedding + '_splits_' + opt.dataset + '_al_' + opt.al_seed + '_' + split_name + '.mat')


train_idx = attrs_mat['trainval_loc'].squeeze() - 1
test_seen_idx = attrs_mat['test_seen_loc'].squeeze() - 1
test_unseen_idx = attrs_mat['test_unseen_loc'].squeeze() - 1  
train_loc = attrs_mat['train_loc'].squeeze() - 1
val_unseen_loc = attrs_mat['val_loc'].squeeze() - 1

print(len(train_loc.tolist()))
print(len(val_unseen_loc.tolist()))
print(len(train_idx.tolist()))
print(len(test_seen_idx.tolist()))
print(len(test_unseen_idx.tolist()))


print(len(np.unique(labels[test_seen_idx])))



attrs = attrs_mat['att'].T
attrs = torch.from_numpy(attrs).to(DEVICE).float()
attrs = attrs / attrs.norm(dim=1, keepdim=True) * np.sqrt(attrs.shape[1])

test_idx = np.array(test_seen_idx.tolist() + test_unseen_idx.tolist())
seen_classes = sorted(np.unique(labels[test_seen_idx]))
unseen_classes = sorted(np.unique(labels[test_unseen_idx])) 
# In case of PS, these unseen labels contain all the common unseen classes + those considered neither in AL selected seen classes nor in common unseen

#new change - get the split_info in final results
splits_folder = None
if opt.cq:
    splits_folder = HOME + opt.dataset + '/' + 'split_info_lr' + str(opt.al_lr) + '_cq' + str(opt.cq) + '/'
else:
    splits_folder = HOME + opt.dataset + '/' + 'split_info_lr' + str(opt.al_lr) + '/'
pklfile = splits_folder + 'u_split' + str(opt.sn) + '_' + 'split_info_' + opt.dataset + '.pickle'
res = open(pklfile, 'rb')
final_results = pickle.load(res)
print('Split info: \n')
print(final_results)        

#new change
common_unseen = final_results['common_unseen'] # this is a list to names (string)
print('Common unseen test classes ({}): {}'.format(len(common_unseen), common_unseen))
all_names = attrs_mat['allclasses_names']



print(f'<=============== Preprocessing ===============>')
num_classes = len(seen_classes) + len(unseen_classes)
print('num classes: {}'.format(num_classes))
seen_mask = np.array([(c in seen_classes) for c in range(num_classes)])
unseen_mask = np.array([(c in unseen_classes) for c in range(num_classes)])


attrs_seen = attrs[seen_mask]
attrs_unseen = attrs[unseen_mask]

train_labels = labels[train_idx]
test_labels = labels[test_idx]     # contains all sample labels for all test samples seen/unseen

#new change
test_seen_labels = labels[test_seen_idx]    # contains samples labels for only seen-class test labels
test_unseen_labels = labels[test_unseen_idx]
test_res = {}
trainval_class_names, testnames_unseen, testnames_seen = [], [], []
# trainval_labels_seen = np.unique(train_labels.numpy())
# test_labels_seen = np.unique(test_seen_labels.numpy())
# test_labels_unseen = np.unique(test_unseen_labels.numpy())

trainval_labels_seen = np.unique(train_labels)
test_labels_seen = np.unique(test_seen_labels)
test_labels_unseen = np.unique(test_unseen_labels)

for i in trainval_labels_seen:
    trainval_class_names.append(all_names[i][0][0])
for i in test_labels_seen:
    testnames_seen.append(all_names[i][0][0])
for i in test_labels_unseen:
    testnames_unseen.append(all_names[i][0][0])   # testnames_unseen[i] contains names of the ith class
    
#new change - all numeric labels of common unseen classes
common_unseen_classes = sorted([i for i in unseen_classes if all_names[i][0][0] in common_unseen])

print('\n\nTrainval classes ({}): {}'.format(len(trainval_class_names), trainval_class_names))
print('\n\nTest seen classes ({}): {}'.format(len(testnames_seen), testnames_seen))
print('\n\nTest unseen classes ({}): {}'.format(len(testnames_unseen), testnames_unseen))

test_res['zsl_trainval'] = trainval_class_names
test_res['zsl_test_seen'] = testnames_seen
test_res['zsl_test_unseen'] = testnames_unseen
test_res['zsl_common_unseen'] = common_unseen
#############################################################


test_seen_idx = [i for i, y in enumerate(test_labels) if y in seen_classes]
test_unseen_idx = [i for i, y in enumerate(test_labels) if y in unseen_classes]
labels_remapped_to_seen = [(seen_classes.index(t) if t in seen_classes else -1) for t in labels]
test_labels_remapped_seen = [(seen_classes.index(t) if t in seen_classes else -1) for t in test_labels]
test_labels_remapped_unseen = [(unseen_classes.index(t) if t in unseen_classes else -1) for t in test_labels]
# In case of PS, these unseen labels contain all the common unseen classes + those considered neither in AL selected seen classes nor in common unseen
ds_train = [(feats[i], labels_remapped_to_seen[i]) for i in train_idx]
ds_test = [(feats[i], int(labels[i])) for i in test_idx]
train_dataloader = DataLoader(ds_train, batch_size=256, shuffle=True)
test_dataloader = DataLoader(ds_test, batch_size=2048)

class_indices_inside_test = {c: [i for i in range(len(test_idx)) if labels[test_idx[i]] == c] for c in range(num_classes)}
#new change
class_indices_inside_common_unseen = {c: [i for i in range(len(test_idx)) if labels[test_idx[i]] == c] for c in range(num_classes)}

class ClassStandardization(nn.Module):
    """
    Class Standardization procedure from the paper.
    Conceptually, it is equivalent to nn.BatchNorm1d with affine=False,
    but for some reason nn.BatchNorm1d performs slightly worse.
    """
    def __init__(self, feat_dim: int):
        super().__init__()
        
        self.running_mean = nn.Parameter(torch.zeros(feat_dim), requires_grad=False)
        self.running_var = nn.Parameter(torch.ones(feat_dim), requires_grad=False)
    
    def forward(self, class_feats):
        """
        Input: class_feats of shape [num_classes, feat_dim]
        Output: class_feats (standardized) of shape [num_classes, feat_dim]
        """
        if self.training:
            batch_mean = class_feats.mean(dim=0)
            batch_var = class_feats.var(dim=0)
            
            # Normalizing the batch
            result = (class_feats - batch_mean.unsqueeze(0)) / (batch_var.unsqueeze(0) + 1e-5)
            
            # Updating the running mean/std
            self.running_mean.data = 0.9 * self.running_mean.data + 0.1 * batch_mean.detach()
            self.running_var.data = 0.9 * self.running_var.data + 0.1 * batch_var.detach()
        else:
            # Using accumulated statistics
            # Attention! For the test inference, we cant use batch-wise statistics,
            # only the accumulated ones. Otherwise, it will be quite transductive
            result = (class_feats - self.running_mean.unsqueeze(0)) / (self.running_var.unsqueeze(0) + 1e-5)
        
        return result


class CNZSLModel(nn.Module):
    def __init__(self, attr_dim: int, hid_dim: int, proto_dim: int):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(attr_dim, hid_dim),
            nn.ReLU(),
            
            nn.Linear(hid_dim, hid_dim),
            ClassStandardization(hid_dim) if USE_CLASS_STANDARTIZATION else nn.Identity(),
            nn.ReLU(),
            
            ClassStandardization(hid_dim) if USE_CLASS_STANDARTIZATION else nn.Identity(),
            nn.Linear(hid_dim, proto_dim),
            nn.ReLU(),
        )
        
        if USE_PROPER_INIT:
            weight_var = 1 / (hid_dim * proto_dim)
            b = np.sqrt(3 * weight_var)
            self.model[-2].weight.data.uniform_(-b, b)
        
    def forward(self, x, attrs):
        protos = self.model(attrs)
        x_ns = 5 * x / x.norm(dim=1, keepdim=True) # [batch_size, x_dim]
        protos_ns = 5 * protos / protos.norm(dim=1, keepdim=True) # [num_classes, x_dim]
        logits = x_ns @ protos_ns.t() # [batch_size, num_classes]
        
        return logits
    

print(f'\n<=============== Starting training ===============>')
start_time = time()
model = CNZSLModel(attrs.shape[1], 1024, feats.shape[1]).to(DEVICE)
optim = torch.optim.Adam(model.model.parameters(), lr=0.0005, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optim, gamma=0.1, step_size=25)



for epoch in tqdm(range(50)):
    model.train()
    
    for i, batch in enumerate(train_dataloader):
        feats = torch.from_numpy(np.array(batch[0])).to(DEVICE)
        targets = torch.from_numpy(np.array(batch[1])).to(DEVICE)
        logits = model(feats, attrs[seen_mask])
        loss = F.cross_entropy(logits, targets)
        optim.zero_grad()
        loss.backward()
        optim.step()
    
    scheduler.step()

print(f'Training is done! Took time: {(time() - start_time): .1f} seconds')

model.eval() # Important! Otherwise we would use unseen batch statistics

print('Testing...\n')
testing = True


logits = [model(x.to(DEVICE), attrs).cpu() for x, _ in test_dataloader]
logits = torch.cat(logits, dim=0)
logits[:, seen_mask] *= (0.95 if DATASET != "CUB" else 1.0) # Trading a bit of gzsl-s for a bit of gzsl-u

preds_gzsl = logits.argmax(dim=1).numpy()
preds_zsl_s = logits[:, seen_mask].argmax(dim=1).numpy()
preds_zsl_u = logits[:, ~seen_mask].argmax(dim=1).numpy()  # this is for all unseen
guessed_zsl_u = (preds_zsl_u == test_labels_remapped_unseen)
guessed_gzsl = (preds_gzsl == test_labels)

#new change
zsl_classwise_all_unseen = [guessed_zsl_u[cls_idx].mean().item() for cls_idx in [class_indices_inside_test[c] for c in unseen_classes]]
# zsl_unseen_acc = np.mean([guessed_zsl_u[cls_idx].mean().item() for cls_idx in [class_indices_inside_test[c] for c in unseen_classes]]) 
zsl_unseen_acc = np.mean(zsl_classwise_all_unseen)
#new change
classwise_accs = {testnames_unseen[i]:a for i, a in enumerate(zsl_classwise_all_unseen)}
classwise_accs_common_unseen = {k:v for k, v in classwise_accs.items() if k in common_unseen}
print(len(classwise_accs_common_unseen))
acc_common_unseen = sum(classwise_accs_common_unseen.values()) / len(classwise_accs_common_unseen)


gzsl_seen_acc = np.mean([guessed_gzsl[cls_idx].mean().item() for cls_idx in [class_indices_inside_test[c] for c in seen_classes]])
gzsl_unseen_acc = np.mean([guessed_gzsl[cls_idx].mean().item() for cls_idx in [class_indices_inside_test[c] for c in unseen_classes]])
gzsl_harmonic = 2 * (gzsl_seen_acc * gzsl_unseen_acc) / (gzsl_seen_acc + gzsl_unseen_acc)


#new change
gzsl_classwise_all_seen = [guessed_gzsl[cls_idx].mean().item() for cls_idx in [class_indices_inside_test[c] for c in seen_classes]]
gzsl_classwise_all_unseen = [guessed_gzsl[cls_idx].mean().item() for cls_idx in [class_indices_inside_test[c] for c in unseen_classes]]
classwise_accs_seen_gzsl = {testnames_seen[i]:a for i, a in enumerate(gzsl_classwise_all_seen)}
classwise_accs_unseen_gzsl = {testnames_unseen[i]:a for i, a in enumerate(gzsl_classwise_all_unseen)}
classwise_accs_common_unseen_gzsl = {k:v for k, v in classwise_accs_unseen_gzsl.items() if k in common_unseen}
print(len(classwise_accs_common_unseen_gzsl))
acc_common_unseen_gzsl = sum(classwise_accs_common_unseen_gzsl.values()) / len(classwise_accs_common_unseen_gzsl)
gzsl_harmonic_new = 2 * (gzsl_seen_acc * acc_common_unseen_gzsl) / (gzsl_seen_acc + acc_common_unseen_gzsl)

print('Test Acc = {:.4f}'.format(zsl_unseen_acc * 100))
print('Class-wise accuracies: ', classwise_accs)
#new change
print('Common unseen Test Acc = {:.4f}'.format(acc_common_unseen * 100))
print('Common unseen Class-wise accuracies: ', classwise_accs_common_unseen)

pkl = open(pklfile_czsl, 'wb')
test_res['total_acc'] = zsl_unseen_acc * 100
test_res['total_classwise'] = classwise_accs
test_res['common_unseen_acc'] = acc_common_unseen * 100
test_res['common_unseen_classwise'] = classwise_accs_common_unseen
pickle.dump(test_res, pkl)
pkl.close()

# print(f'ZSL-U: {zsl_unseen_acc * 100:.02f}')
# print(f'ZSL-U: {zsl_unseen_acc * 100:.02f}')
czsl_res_file.close()
sys.stdout = gzsl_res_file
print('\n\n\nFinal results....................')
print('Dataset', opt.dataset)
print('the best GZSL seen accuracy is', gzsl_seen_acc * 100)
print('the best GZSL unseen accuracy is', gzsl_unseen_acc * 100)
print('the best GZSL H is', gzsl_harmonic * 100)



print('the best GZSL considered H is', gzsl_harmonic_new * 100)
print('acc common unseen: ', acc_common_unseen_gzsl * 100)
print('classwise accs seen: ', classwise_accs_seen_gzsl)
print('classwise accs unseen: ', classwise_accs_unseen_gzsl)
print('classwise_accs_common_unseen: ', classwise_accs_common_unseen_gzsl)


pkl = open(pklfile_gzsl, 'wb')
gzsl_test_res = {}
gzsl_test_res['acc_unseen_classes'] = gzsl_unseen_acc *100
gzsl_test_res['acc_seen_classes'] = gzsl_seen_acc * 100
gzsl_test_res['total_HM'] = gzsl_harmonic * 100
gzsl_test_res['classwise_unseen_accs'] = classwise_accs_unseen_gzsl
gzsl_test_res['acc_common_unseen'] = acc_common_unseen_gzsl * 100
gzsl_test_res['classwise_accs_common_unseen'] = classwise_accs_common_unseen_gzsl
gzsl_test_res['considered_HM'] = gzsl_harmonic_new * 100
pickle.dump(gzsl_test_res, pkl)
pkl.close()



# print(f'GZSL-U: {gzsl_unseen_acc * 100:.02f}')
# print(f'GZSL-S: {gzsl_seen_acc * 100:.02f}')
# print(f'GZSL-H: {gzsl_harmonic * 100:.02f}')