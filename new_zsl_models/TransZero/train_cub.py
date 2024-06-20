import torch
import torch.optim as optim
import torch.nn as nn
from model import TransZero
from dataset import CUBDataLoader
from helper_func import eval_zs_gzsl
import numpy as np
import wandb

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


# new change - init wandb from config file
if opt.al_seed == 'original':
    sname = 'ES'
else:
    sname = 'PS'
run_name = opt.dataset + '_split' + str(opt.sn) + '_' + sname  
wandb.init(project='TransZero', config='wandb_config/cub_gzsl.yaml', name=run_name)
config = wandb.config
print('Config file from wandb:', config)

# load dataset
# dataloader = CUBDataLoader('.', config.device, is_balance=False)

#new change
dataloader = CUBDataLoader(opt.imgdata, config.device, is_balance=False, opt = opt)

# set random seed
seed = config.random_seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

# TransZero model
model = TransZero(config, dataloader.att, dataloader.w2v_att,
                  dataloader.seenclasses, dataloader.unseenclasses).to(config.device)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

# main loop
niters = dataloader.ntrain * config.epochs//config.batch_size
report_interval = niters//config.epochs

#new change
best_acc_seen = 0
best_acc_unseen = 0
best_H_new = 0
best_acc_common_unseen = 0
best_acc_common_unseen_gzsl = 0
best_classwise_accs = {}
best_classwise_accs_common_unseen = {}
best_classwise_accs_gzsl = {}
best_classwise_accs_common_unseen_gzsl = {}


best_performance = [0, 0, 0, 0, 0, 0]
best_performance_zsl = 0
for i in range(0, niters):
    model.train()
    optimizer.zero_grad()

    batch_label, batch_feature, batch_att = dataloader.next_batch(
        config.batch_size)
    out_package = model(batch_feature)

    in_package = out_package
    in_package['batch_label'] = batch_label

    out_package = model.compute_loss(in_package)
    loss, loss_CE, loss_cal, loss_reg = out_package['loss'], out_package[
        'loss_CE'], out_package['loss_cal'], out_package['loss_reg']

    loss.backward()
    optimizer.step()

    # # report result
    # if i % report_interval == 0:
    #     print('-'*30)
    #     acc_seen, acc_novel, H, acc_zs = eval_zs_gzsl(
    #         dataloader, model, config.device, batch_size=config.batch_size)

    #     if H > best_performance[2]:
    #         best_performance = [acc_novel, acc_seen, H, acc_zs]
    #     if acc_zs > best_performance_zsl:
    #         best_performance_zsl = acc_zs

    #     print('iter/epoch=%d/%d | loss=%.3f, loss_CE=%.3f, loss_cal=%.3f, '
    #           'loss_reg=%.3f | acc_unseen=%.3f, acc_seen=%.3f, H=%.3f | '
    #           'acc_zs=%.3f' % (
    #               i, int(i//report_interval),
    #               loss.item(), loss_CE.item(), loss_cal.item(),
    #               loss_reg.item(),
    #               best_performance[0], best_performance[1],
    #               best_performance[2], best_performance_zsl))

    #     wandb.log({
    #         'iter': i,
    #         'loss': loss.item(),
    #         'loss_CE': loss_CE.item(),
    #         'loss_cal': loss_cal.item(),
    #         'loss_reg': loss_reg.item(),
    #         'acc_unseen': acc_novel,
    #         'acc_seen': acc_seen,
    #         'H': H,
    #         'acc_zs': acc_zs,
    #         'best_acc_unseen': best_performance[0],
    #         'best_acc_seen': best_performance[1],
    #         'best_H': best_performance[2],
    #         'best_acc_zs': best_performance_zsl
    #     })


    # report result
    if i % report_interval == 0:
        print('-'*30)
        #new change - added last 2 params
        acc_seen, acc_unseen, H, H_new, acc_common_unseen_gzsl, classwise_accs_gzsl, classwise_accs_common_unseen_gzsl, \
                acc_zs_t, acc_common_unseen_zs_t, classwise_accs_zs_t, \
                classwise_accs_common_unseen_zs_t=   eval_zs_gzsl(dataloader, model, config.device, batch_size=config.batch_size, testing=True, opt=opt)


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
            best_performance_zsl = acc_zs_t
            best_acc_common_unseen = acc_common_unseen_zs_t
            best_classwise_accs = classwise_accs_zs_t
            best_classwise_accs_common_unseen = classwise_accs_common_unseen_zs_t

        if opt.gzsl:
            # new change - change output file
            sys.stdout = gzsl_res_file
            print('\n\nTesting starts for GZSL.......')

            print('iter/epoch=%d/%d | loss=%.3f, loss_CE=%.4f, loss_cal=%.4f, '
                'loss_reg=%.4f | acc_unseen=%.4f, acc_seen=%.3f, H=%.4f | '
                'acc_common_unseen=%.4f, acc_considered_H=%.4f' % (
                    i, int(i//report_interval),
                    loss.item(), loss_CE.item(), loss_cal.item(),
                    loss_reg.item(),
                    best_performance[0], best_performance[1],
                    best_performance[2], best_performance[3], best_performance[4]))
            
            #for czsl
            sys.stdout = czsl_res_file
            print('\n\nTesting starts for CZSL.......')

            print('iter/epoch=%d/%d | loss=%.3f, loss_CE=%.4f, loss_cal=%.4f, '
                'loss_reg=%.4f | acc_unseen=%.4f, acc_common_unseen=%.4f' % (
                    i, int(i//report_interval),
                    loss.item(), loss_CE.item(), loss_cal.item(),
                    loss_reg.item(),
                    best_performance_zsl, best_performance[5]))
        
            #new change
            wandb.log({
                'iter': i,
                'loss': loss.item(),
                'loss_CE': loss_CE.item(),
                'loss_cal': loss_cal.item(),
                'loss_reg': loss_reg.item(),
                'acc_unseen': acc_unseen,
                'acc_seen': acc_seen,
                'H': H,
                'H_new': H_new,
                'acc_zs': acc_zs_t,
                'best_acc_unseen': best_performance[0],
                'best_acc_seen': best_performance[1],
                'best_H': best_performance[2],
                'best_acc_zs': best_performance_zsl,
                'best_H_considered': best_performance[4],
                'best_acc_common_unseen_gzsl': best_acc_common_unseen_gzsl,
                'best_acc_common_unseen': best_acc_common_unseen
            })


#new change - change output file 
sys.stdout = czsl_res_file
print('\n\n\nFinal results....................')
print(time.strftime('ending time:%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
print('Dataset', opt.dataset)
print('the best ZSL unseen accuracy is', best_performance_zsl)
print('Class-wise accuracies: ', best_classwise_accs)
print('Common unseen Test Acc = {:.4f}'.format(best_acc_common_unseen))
print('Common unseen Class-wise accuracies: ', best_classwise_accs_common_unseen)
pkl = open(pklfile_czsl, 'wb')
dataloader.test_res['total_acc'] = best_performance_zsl
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