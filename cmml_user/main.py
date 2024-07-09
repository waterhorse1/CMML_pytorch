import copy
import os
import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import util as utils
from dataset import Metamovie_new
from logger import Logger
from module import meta_mlp_context, meta_soft_module, meta_final_context
import argparse
import torch
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from tqdm import tqdm
import inspect
import time
from math import log2

#initialise logger
logger = Logger()
    
def parse_args():
    parser = argparse.ArgumentParser([],description='Fast Context Adaptation via Meta-Learning (CAVIA),'
                                                 'Clasification experiments.')

    parser.add_argument('--seed', type=int, default=53)
    parser.add_argument('--task', type=str, default='multi', help='problem setting: sine or celeba')
    parser.add_argument('--tasks_per_metaupdate', type=int, default=32, help='number of tasks in each batch per meta-update')

    parser.add_argument('--lr_meta', type=float, default=3e-4, help='outer-loop learning rate (used with Adam optimiser)')
    #parser.add_argument('--lr_meta_decay', type=float, default=0.9, help='decay factor for meta learning rate')

    parser.add_argument('--first_order', action='store_true', default=False, help='run first order approximation of CAVIA')

    parser.add_argument('--data_root', type=str, default="./movielens/ml-1m", help='path to data root')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--test', action='store_true', default=False, help='num of workers to use')
    
    parser.add_argument('--embedding_dim', type=int, default=32, help='num of workers to use')
    parser.add_argument('--first_fc_hidden_dim', type=int, default=64, help='num of workers to use')
    parser.add_argument('--second_fc_hidden_dim', type=int, default=64, help='num of workers to use')
    parser.add_argument('--num_epoch', type=int, default=30, help='num of workers to use')
    parser.add_argument('--num_genre', type=int, default=25, help='num of workers to use')
    parser.add_argument('--num_director', type=int, default=2186, help='num of workers to use')
    parser.add_argument('--num_actor', type=int, default=8030, help='num of workers to use')
    parser.add_argument('--num_rate', type=int, default=6, help='num of workers to use')
    parser.add_argument('--num_gender', type=int, default=2, help='num of workers to use')
    parser.add_argument('--num_age', type=int, default=7, help='num of workers to use')
    parser.add_argument('--num_occupation', type=int, default=21, help='num of workers to use')
    parser.add_argument('--num_zipcode', type=int, default=3402, help='num of workers to use')
    
    parser.add_argument('--rerun', action='store_true', default=False,
                        help='Re-run experiment (will override previously saved results)')
    
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--mlp_hyper_hidden_dim', type=int, default=256)
    parser.add_argument('--context_encoder', type=str, default='mean')
    parser.add_argument('--comment', type=str, default='None')

    args = parser.parse_args()
    # use the GPU if available
    #args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print('Running on device: {}'.format(args.device))
    return args

def NDCG_K(real, predict, k=3):
    predict = predict.reshape(-1)
    real = real.reshape(-1)
    index = torch.topk(predict, k).indices
    score = real[index]
    real_score = torch.topk(real, k).values
    dcg = DCG(score)
    idcg = DCG(real_score)
    result = dcg/idcg
    return result.item()

def DCG(tensor):
    score = 0
    for i in range(1, len(tensor)+1):
        score += (2 ** tensor[i-1] - 1)/log2(i+1)
    return score

def run(args, num_workers=1, log_interval=100, verbose=True, save_path=None):
    code_root = os.path.dirname(os.path.realpath(__file__))
    if not os.path.isdir('{}/{}_result_files/'.format(code_root, args.task)):
        os.mkdir('{}/{}_result_files/'.format(code_root, args.task))
    path = '{}/{}_result_files/'.format(code_root, args.task) + utils.get_path_from_args(args)
    print('File saved in {}'.format(path))

    if os.path.exists(path + '.pkl') and not args.rerun:
        print('File has already existed. Try --rerun')
        return utils.load_obj(path)

    start_time = time.time()
    utils.set_seed(args.seed)


    # ---------------------------------------------------------
    # -------------------- training ---------------------------

    # initialise model
    if args.model == 'mlp':
        model = meta_mlp_context(args).cuda()
    else:
        model = meta_soft_module(args).cuda()

    model.train()
    print(sum([param.nelement() for param in model.parameters()]))
    # set up meta-optimiser for model parameters
    meta_optimiser = torch.optim.Adam(model.parameters(), args.lr_meta, weight_decay=1e-5)
   # scheduler = torch.optim.lr_scheduler.StepLR(meta_optimiser, 5000, args.lr_meta_decay)

    logger.args = args
    # initialise the starting point for the meta gradient (it's faster to copy this than to create new object)
    meta_grad_init = [0 for _ in range(len(model.state_dict()))]
    dataloader_train = DataLoader(Metamovie_new(args),
                                     batch_size=1,num_workers=args.num_workers)
    for epoch in range(args.num_epoch):
        x_spt, y_spt, x_qry, y_qry = [],[],[],[]
        iter_counter = 0
        for step, batch in enumerate(dataloader_train):
            if len(x_spt)<args.tasks_per_metaupdate:
                x_spt.append(batch[0][0].cuda())
                y_spt.append(batch[1][0].cuda())
                x_qry.append(batch[2][0].cuda())
                y_qry.append(batch[3][0].cuda())
                if not len(x_spt)==args.tasks_per_metaupdate:
                    continue
            
            if len(x_spt) != args.tasks_per_metaupdate:
                continue   
            #context generate
            meta_grad = copy.deepcopy(meta_grad_init)
            loss_n1 = []
            loss_n3 = []
            loss_after = []
            
            for i in range(args.tasks_per_metaupdate): 
                model.set_context(x_spt[i], y_spt[i])

                logits_q = model(x_qry[i])
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.mse_loss(logits_q, y_qry[i])
                loss_after.append(loss_q.item())  
                loss_n1.append(NDCG_K(y_qry[i], logits_q, k=1))
                loss_n3.append(NDCG_K(y_qry[i], logits_q, k=3))
                task_grad_test = torch.autograd.grad(loss_q, model.parameters(),allow_unused=True)
                
                for g in range(len(task_grad_test)):
                    meta_grad[g] += task_grad_test[g].detach()
            # -------------- meta update --------------
            meta_optimiser.zero_grad()
            for c, param in enumerate(model.parameters()):
                param.grad = meta_grad[c] / float(args.tasks_per_metaupdate)
                param.grad.data.clamp_(-10, 10)

            # the meta-optimiser only operates on the shared parameters, not the context parameters
            meta_optimiser.step()
            #scheduler.step()
            x_spt, y_spt, x_qry, y_qry = [],[],[],[]
            
            loss_n1 = np.array(loss_n1)
            loss_after = np.array(loss_after)
            loss_n3 = np.array(loss_n3)
            logger.train_loss.append(np.mean(loss_after))
            logger.train_ndcg1.append(np.mean(loss_n1))
            logger.train_ndcg1_conf.append(1.96*np.std(loss_n1, ddof=0)/np.sqrt(len(loss_n1)))
            logger.train_conf.append(1.96*np.std(loss_after, ddof=0)/np.sqrt(len(loss_after)))
            logger.train_ndcg3.append(np.mean(loss_n3))
            logger.train_ndcg3_conf.append(1.96*np.std(loss_n3, ddof=0)/np.sqrt(len(loss_n3)))
    
            utils.save_obj(logger, path)
            # print current results
            logger.print_info(epoch, iter_counter, start_time)
            start_time = time.time()
            
            iter_counter += 1
        if epoch % (2) == 0:
            print('saving model at iter', epoch)
            dataloader_test = DataLoader(Metamovie_new(args,partition='test',test_way='new_user'),#old, new_user, new_item, new_item_user
                                     batch_size=1,num_workers=args.num_workers)
            evaluate_test(args, model, dataloader_test, test='test')
            dataloader_test = DataLoader(Metamovie_new(args,partition='test',test_way='old'),#old, new_user, new_item, new_item_user
                                     batch_size=1,num_workers=args.num_workers)
            evaluate_test(args, model, dataloader_test, test='train')
            model.train()
            logger.valid_model.append(copy.deepcopy(model.state_dict()))

    return logger, model

def evaluate_test(args, model,  dataloader, test):
    model.eval()
    loss_all = []
    loss_n1 = []
    loss_n3 = []
    for c, batch in tqdm(enumerate(dataloader)):
        x_spt = batch[0].cuda()
        y_spt = batch[1].cuda()
        x_qry = batch[2].cuda()
        y_qry = batch[3].cuda()
        for i in range(x_spt.shape[0]):
            # -------------- inner update --------------
            model.set_context(x_spt[i], y_spt[i])

            logits = model(x_qry[i])
            loss_all.append(F.l1_loss(y_qry[i], logits).item())
            loss_n1.append(NDCG_K(y_qry[i], logits, 1))
            loss_n3.append(NDCG_K(y_qry[i], logits, 3))
    loss_all = np.array(loss_all)
    loss_n1 = np.array(loss_n1)
    loss_n3 = np.array(loss_n3)
    
    if test == 'test':
        idx = 0
    else:
        idx = 1
    logger.test_loss[idx].append(np.mean(loss_all))
    logger.test_conf[idx].append(1.96*np.std(loss_all, ddof=0)/np.sqrt(len(loss_all)))
    logger.test_ndcg1[idx].append(np.mean(loss_n1))
    logger.test_ndcg1_conf[idx].append(1.96*np.std(loss_n1, ddof=0)/np.sqrt(len(loss_n1)))
    logger.test_ndcg3[idx].append(np.mean(loss_n3))
    logger.test_ndcg3_conf[idx].append(1.96*np.std(loss_n3, ddof=0)/np.sqrt(len(loss_n3)))
    
    print('{}+/-{}'.format(np.mean(loss_all), 1.96*np.std(loss_all,0)/np.sqrt(len(loss_all))))
    print('{}+/-{}'.format(np.mean(loss_n1), 1.96*np.std(loss_n1,0)/np.sqrt(len(loss_n1))))
    print('{}+/-{}'.format(np.mean(loss_n3), 1.96*np.std(loss_n3,0)/np.sqrt(len(loss_n3))))

if __name__ == '__main__':
    args = parse_args()
    run(args, num_workers=1, log_interval=100, verbose=True, save_path=None)



