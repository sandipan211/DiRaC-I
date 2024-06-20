import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from core.MSDN import MSDN
from core.CUBDataLoader import CUBDataLoader
from core.helper_MSDN_CUB import eval_zs_gzsl
# from global_setting import NFS_path
import importlib
import pdb
import numpy as np
import matplotlib.pyplot as plt

#new change
import pickle
import os
import sys
from dirac_i_args import opt
import time

#new change
czsl_folder = None
gzsl_folder = None
pklfile_czsl = None
pklfile_gzsl = None
result_filename_czsl = None
result_filename_gzsl = None
gzsl_save_path = None
czsl_save_path = None

if opt.gzsl:
    #new change - made separate report folders
    gzsl_folder = opt.work_dir + 'new_zsl_models/' + opt.fname + '/GZSL_al_lr' + str(opt.al_lr) + '/' 
    gzsl_report_folder = gzsl_folder +'u_split' + str(opt.sn) + '/'
    if not os.path.exists(gzsl_folder):
        os.mkdir(gzsl_folder)
    if not os.path.exists(gzsl_report_folder):
        os.mkdir(gzsl_report_folder) 
    if opt.cq:
        fstr = gzsl_report_folder + opt.fname + '_gzsl_' + opt.dataset  + '_' + opt.al_seed + '_cq' + str(opt.cq)
    else:
        fstr = gzsl_report_folder + opt.fname + '_gzsl_' + opt.dataset  + '_' + opt.al_seed
    result_filename_gzsl = fstr + '_reports.txt'
    pklfile_gzsl = fstr + '_results.pickle'
    gzsl_res_file = open(result_filename_gzsl, 'w')



#new change - made separate report folders
czsl_folder = opt.work_dir + 'new_zsl_models/' + opt.fname + '/CZSL_al_lr' + str(opt.al_lr) + '/' 
czsl_report_folder = czsl_folder +'u_split' + str(opt.sn) + '/'
if not os.path.exists(czsl_folder):
    os.mkdir(czsl_folder)
if not os.path.exists(czsl_report_folder):
    os.mkdir(czsl_report_folder)

if opt.cq:
    fstr = czsl_report_folder + opt.fname + '_czsl_' + opt.dataset  + '_' + opt.al_seed + '_cq' +str(opt.cq)
else:
    fstr = czsl_report_folder + opt.fname + '_czsl_' + opt.dataset  + '_' + opt.al_seed 
result_filename_czsl = fstr + '_reports.txt'
pklfile_czsl = fstr + '_results.pickle'
czsl_res_file = open(result_filename_czsl, 'w')


# all in-training results will be written in result_filename_czsl
sys.stdout = czsl_res_file

NFS_path = './' # change it

idx_GPU = 0
device = torch.device("cuda:{}".format(idx_GPU) if torch.cuda.is_available() else "cpu")
dataloader = CUBDataLoader(NFS_path,device,is_unsupervised_attr=False,is_balance=False, opt=opt)
torch.backends.cudnn.benchmark = True

def get_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr.append(param_group['lr'])
    return lr

seed = 214#215#
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

batch_size = 50
nepoches = 30#22
niters = dataloader.ntrain * nepoches//batch_size
dim_f = 2048
dim_v = 300
init_w2v_att = dataloader.w2v_att
att = dataloader.att
normalize_att = dataloader.normalize_att

trainable_w2v = True
lambda_ = 0.18#0.1 for GZSL, 0.18 for CZSL
bias = 0
prob_prune = 0
uniform_att_1 = False
uniform_att_2 = False

seenclass = dataloader.seenclasses
unseenclass = dataloader.unseenclasses
desired_mass = 1
report_interval = niters//nepoches

model = MSDN(dim_f,dim_v,init_w2v_att,att,normalize_att,
            seenclass,unseenclass,
            lambda_,
            trainable_w2v,normalize_V=False,normalize_F=True,is_conservative=True,
            uniform_att_1=uniform_att_1,uniform_att_2=uniform_att_2,
            prob_prune=prob_prune,desired_mass=desired_mass, is_conv=False,
            is_bias=True)
model.to(device)

setup = {'pmp':{'init_lambda':0.1,'final_lambda':0.1,'phase':0.8},
         'desired_mass':{'init_lambda':-1,'final_lambda':-1,'phase':0.8}}
print(setup)
#scheduler = Scheduler(model,niters,batch_size,report_interval,setup)

params_to_update = []
params_names = []
for name,param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        params_names.append(name)
        print("\t",name)
#%%
lr = 0.0001
weight_decay = 0.0001#0.000#0.#
momentum = 0.9#0.#
#%%
lr_seperator = 1
lr_factor = 1
print('default lr {} {}x lr {}'.format(params_names[:lr_seperator],lr_factor,params_names[lr_seperator:]))
optimizer  = optim.RMSprop( params_to_update ,lr=lr,weight_decay=weight_decay, momentum=momentum)

print('-'*30)
print('learing rate {}'.format(lr))
print('trainable V {}'.format(trainable_w2v))
print('lambda_ {}'.format(lambda_))
print('optimized seen only')
print('optimizer: RMSProp with momentum = {} and weight_decay = {}'.format(momentum,weight_decay))
print('-'*30)

iter_x = []
# best_H = []
# best_ACC =[]

# best_performance = [0,0,0]
best_acc = 0
#new change
best_acc_seen = 0
best_acc_unseen = 0
best_H = 0
best_H_new = 0
best_acc_common_unseen = 0
best_acc_common_unseen_gzsl = 0
best_classwise_accs = {}
best_classwise_accs_common_unseen = {}
best_classwise_accs_gzsl = {}
best_classwise_accs_common_unseen_gzsl = {}


best_performance = [0, 0, 0, 0, 0, 0]


for i in range(0,niters):
    model.train()
    optimizer.zero_grad()
    
    batch_label, batch_feature, batch_att = dataloader.next_batch(batch_size)

    out_package1, out_package2= model(batch_feature)
    
    in_package1 = out_package1
    in_package2 = out_package2
    in_package1['batch_label'] = batch_label
    in_package2['batch_label'] = batch_label
    
    out_package1=model.compute_loss(in_package1)
    out_package2=model.compute_loss(in_package2)
    loss,loss_CE,loss_cal = out_package1['loss']+out_package2['loss'],out_package1['loss_CE']+out_package2['loss_CE'],out_package1['loss_cal']+out_package2['loss_cal']
    constrastive_loss1=model.compute_contrastive_loss(in_package1, in_package2)

    loss=loss + 0.001*constrastive_loss1##0.001

    
    loss.backward()
    optimizer.step()
    # if i%report_interval==0:
    #     print('-'*30)
    #     acc_seen, acc_novel, H, acc_zs = eval_zs_gzsl(dataloader,model,device,bias_seen=-bias,bias_unseen=bias)
        
    #     if H > best_performance[2]:
    #         best_performance = [acc_novel, acc_seen, H]
    #     if acc_zs > best_acc:
    #         best_acc = acc_zs
    #     print('iter=%d, loss=%.3f, loss_CE=%.3f, loss_cal=%.3f, acc_unseen=%.3f, acc_seen=%.3f, H=%.3f, acc_zs=%.3f'%(i,loss.item(),loss_CE.item(),loss_cal.item(),best_performance[0],best_performance[1],best_performance[2],best_acc))

    
    # report result
    if i % report_interval == 0:
        print('-'*30)
        #new change - added last 2 params
        acc_seen, acc_unseen, H, H_new, acc_common_unseen_gzsl, classwise_accs_gzsl, classwise_accs_common_unseen_gzsl, \
                acc_zs_t, acc_common_unseen_zs_t, classwise_accs_zs_t, \
                classwise_accs_common_unseen_zs_t=   eval_zs_gzsl(dataloader, model, device, bias_seen=-bias,bias_unseen=bias, testing=True, opt=opt)


        #new change - doing everything on the basis of H_new
        if H_new > best_performance[4]:
            best_performance = [acc_unseen, acc_seen, H, acc_common_unseen_gzsl, H_new, acc_common_unseen_zs_t]
            best_seen = acc_seen
            best_unseen = acc_unseen
            best_H = H
            best_H_new = H_new
            best_acc_common_unseen_gzsl = acc_common_unseen_gzsl
            best_classwise_accs_gzsl = classwise_accs_gzsl
            best_classwise_accs_common_unseen_gzsl = classwise_accs_common_unseen_gzsl

        if acc_common_unseen_zs_t > best_acc_common_unseen:
            best_acc = acc_zs_t
            best_acc_common_unseen = acc_common_unseen_zs_t
            best_classwise_accs = classwise_accs_zs_t
            best_classwise_accs_common_unseen = classwise_accs_common_unseen_zs_t

        if opt.gzsl:
            # new change - change output file
            sys.stdout = gzsl_res_file
            print('\n\nTesting starts for GZSL.......')
            
            print('iter=%d, loss=%.3f, loss_CE=%.3f, loss_cal=%.3f, acc_unseen=%.4f, acc_seen=%.4f, H=%.4f |' 'acc_common_unseen=%.4f , acc_considered_H=%.4f'%(i,loss.item(),loss_CE.item(),loss_cal.item(),best_performance[0],best_performance[1],best_performance[2],best_performance[3], best_performance[4]))

            #for czsl
            sys.stdout = czsl_res_file
            print('\n\nTesting starts for CZSL.......')
            print('iter=%d, loss=%.3f, loss_CE=%.3f, loss_cal=%.3f, acc_unseen=%.4f , acc_common_unseen=%.4f'%(i,loss.item(),loss_CE.item(),loss_cal.item(),best_acc, best_performance[5]))


#new change - change output file 
sys.stdout = czsl_res_file
print('\n\n\nFinal results....................')
print('Dataset', opt.dataset)
print('the best ZSL unseen accuracy is', best_acc)
print('Class-wise accuracies: ', best_classwise_accs)
print('Common unseen Test Acc = {:.4f}'.format(best_acc_common_unseen))
print('Common unseen Class-wise accuracies: ', best_classwise_accs_common_unseen)
pkl = open(pklfile_czsl, 'wb')
dataloader.test_res['total_acc'] = best_acc
dataloader.test_res['total_classwise'] = best_classwise_accs
dataloader.test_res['common_unseen_acc'] = best_acc_common_unseen
dataloader.test_res['common_unseen_classwise'] = best_classwise_accs_common_unseen
pickle.dump(dataloader.test_res, pkl)
pkl.close()



if opt.gzsl:
    #new change - change output file 
    sys.stdout = gzsl_res_file
    print('\n\n\nFinal results....................')
    print(time.strftime('ending time:%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    print('Dataset', opt.dataset)
    print('the best GZSL seen accuracy is', best_acc_seen)
    print('the best GZSL unseen accuracy is', best_acc_unseen)
    print('the best GZSL H is', best_H)
    print('the best GZSL considered H is', best_H_new)
    print('acc common unseen: ', best_acc_common_unseen_gzsl)
    print('classwise accs: ', best_classwise_accs_gzsl)
    print('classwise_accs_common_unseen: ', best_classwise_accs_common_unseen_gzsl)
    pkl = open(pklfile_gzsl, 'wb')
    gzsl_test_res = {}
    gzsl_test_res['acc_unseen_classes'] = best_acc_unseen
    gzsl_test_res['acc_seen_classes'] = best_acc_seen
    gzsl_test_res['total_HM'] = best_H
    gzsl_test_res['classwise_accs'] = best_classwise_accs_gzsl
    gzsl_test_res['acc_common_unseen'] = best_acc_common_unseen_gzsl
    gzsl_test_res['classwise_accs_common_unseen'] = best_classwise_accs_common_unseen_gzsl
    gzsl_test_res['considered_HM'] = best_H_new
    pickle.dump(gzsl_test_res, pkl)
    pkl.close()

#new change - closing files
czsl_res_file.close()
if opt.gzsl:
    gzsl_res_file.close()