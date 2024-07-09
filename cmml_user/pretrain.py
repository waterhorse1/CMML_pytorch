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
from dataset import Metamovie_new, pretrain_dataset
from logger import Logger
from MeLU import meta_mlp_context, meta_soft_module, user_item_embed, DCN, interaction
import argparse
import torch
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from tqdm import tqdm
from gpu_mem_track import MemTracker
import inspect
import time
from math import log2
def parse_args():
    parser = argparse.ArgumentParser([],description='Fast Context Adaptation via Meta-Learning (CAVIA),'
                                                 'Clasification experiments.')

    parser.add_argument('--seed', type=int, default=53)
    parser.add_argument('--task', type=str, default='pretrain', help='problem setting: sine or celeba')
    parser.add_argument('--tasks_per_metaupdate', type=int, default=32, help='number of tasks in each batch per meta-update')

    parser.add_argument('--lr_inner', type=float, default=0.01, help='inner-loop learning rate (per task)')
    parser.add_argument('--lr_meta', type=float, default=1e-3, help='outer-loop learning rate (used with Adam optimiser)')
    #parser.add_argument('--lr_meta_decay', type=float, default=0.9, help='decay factor for meta learning rate')

    parser.add_argument('--num_grad_steps_inner', type=int, default=5, help='number of gradient steps in inner loop (during training)')
    parser.add_argument('--num_grad_steps_eval', type=int, default=1, help='number of gradient updates at test time (for evaluation)')

    parser.add_argument('--first_order', action='store_true', default=False, help='run first order approximation of CAVIA')

    parser.add_argument('--data_root', type=str, default="./movielens/ml-1m", help='path to data root')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--test', action='store_true', default=False, help='num of workers to use')
    
    parser.add_argument('--embedding_dim', type=int, default=32, help='num of workers to use')
    parser.add_argument('--first_fc_hidden_dim', type=int, default=64, help='num of workers to use')
    parser.add_argument('--second_fc_hidden_dim', type=int, default=64, help='num of workers to use')
    parser.add_argument('--num_epoch', type=int, default=11, help='num of workers to use')
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
    
    parser.add_argument('--comment', type=str, default='123')

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
    uie = user_item_embed(args)
    model = DCN(uie).cuda()

    model.train()
    print(sum([param.nelement() for param in model.parameters()]))
    # set up meta-optimiser for model parameters
    optimiser = torch.optim.Adam(model.parameters(), args.lr_meta)
   # scheduler = torch.optim.lr_scheduler.StepLR(meta_optimiser, 5000, args.lr_meta_decay)

    # initialise logger
    logger = Logger()
    logger.args = args
    # initialise the starting point for the meta gradient (it's faster to copy this than to create new object)
    dataloader_train = DataLoader(pretrain_dataset(args),
                                     batch_size=128,num_workers=args.num_workers)
    for epoch in range(args.num_epoch):
        x_spt, y_spt, x_qry, y_qry = [],[],[],[]
        iter_counter = 0
        for step, batch in enumerate(dataloader_train):
            x, y = batch
            x = x.cuda()
            y = y.cuda()
            logits_q = model(x)
            #print(logits_q)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.mse_loss(logits_q, y)
            #loss_after.append(loss_q.item())  
              
            optimiser.zero_grad()
        
            loss_q.backward()

            optimiser.step()
            
            logger.train_loss.append(0)
            logger.valid_loss.append(loss_q.item())
            logger.train_conf.append(0)
            logger.valid_conf.append(0)
            logger.test_loss.append(0)
            logger.test_conf.append(0)
    
            utils.save_obj(logger, path)
            # print current results
            logger.print_info(epoch, iter_counter, start_time)
            start_time = time.time()
            
            iter_counter += 1
        #dataloader_test = DataLoader(Metamovie(args,partition='test',test_way='old'),#old, new_user, new_item, new_item_user
        #                             batch_size=1,num_workers=args.num_workers)
        #evaluate_test(args, model, dataloader_test)
        #model.train()
        model_copy = copy.deepcopy(model)
        if epoch % (2) == 0:
            print('saving model at iter', epoch)
            dataloader_test = DataLoader(Metamovie_new(args,partition='test',test_way='new_user'),#old, new_user, new_item, new_item_user
                                     batch_size=1,num_workers=args.num_workers)
            evaluate_test(args, model, dataloader_test)
            
            dataloader_test = DataLoader(Metamovie_new(args,partition='test',test_way='old'),#old, new_user, new_item, new_item_user
                                     batch_size=1,num_workers=args.num_workers)
            evaluate_test(args, model, dataloader_test)
            model = copy.deepcopy(model_copy)
            model.train()
            logger.valid_model.append(copy.deepcopy(model))

    return logger, model


def evaluate(iter_counter, args, model, logger, dataloader, save_path):
    logger.prepare_inner_loop(iter_counter, mode='valid')

    for c, batch in enumerate(dataloader):
        x_spt = batch[0].cuda()
        y_spt = batch[1].cuda()
        x_qry = batch[2].cuda()
        y_qry = batch[3].cuda()

        for i in range(x_spt.shape[0]):

            # -------------- inner update --------------

            logger.log_pre_update(iter_counter,
                                  x_spt[i], y_spt[i],
                                  x_qry[i], y_qry[i],
                                  model, mode='valid')
            fast_parameters = model.parameters()
            for weight in model.parameters():
                weight.fast = None
            for k in range(args.num_grad_steps_eval):
                logits = model(x_spt[i])
                loss = F.cross_entropy(logits, y_spt[i])
                grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
                fast_parameters = []
                for k, weight in enumerate(model.parameters()):
                    #for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py 
                    if weight.fast is None:
                        weight.fast = weight - args.lr_inner * grad[k] #create weight.fast 
                    else:
                        weight.fast = weight.fast - args.lr_inner * grad[k]  
                    fast_parameters.append(weight.fast)

            logger.log_post_update(iter_counter, x_spt[i], y_spt[i],
                                          x_qry[i], y_qry[i], model, mode='valid')
            
    # this will take the mean over the batches
    logger.summarise_inner_loop(mode='valid')

    # keep track of best models
    logger.update_best_model(model, save_path)

def evaluate_test(args, model,  dataloader):
    model.eval()
    loss_all = []
    loss_n1 = []
    loss_n3 = []
    model_copy = copy.deepcopy(model)
    for c, batch in tqdm(enumerate(dataloader)):
        model = copy.deepcopy(model_copy)
        model.train()
        op = torch.optim.Adam(model.parameters(), lr=2e-4)
        x_spt = batch[0].cuda()
        y_spt = batch[1].cuda()
        x_qry = batch[2].cuda()
        y_qry = batch[3].cuda()
        for i in range(x_spt.shape[0]):
            # -------------- inner update --------------
            #model.set_context(x_spt[i], y_spt[i])
            
            for _ in range(5):
                y = model(x_spt[i])
                loss = F.mse_loss(y, y_spt[i])
                op.zero_grad()  
                loss.backward()
                op.step()
            
            logits = model(x_qry[i])
            loss_all.append(F.l1_loss(y_qry[i], logits).item())
            loss_n1.append(NDCG_K(y_qry[i], logits, 1))
            loss_n3.append(NDCG_K(y_qry[i], logits, 3))
    loss_all = np.array(loss_all)
    loss_n1 = np.array(loss_n1)
    loss_n3 = np.array(loss_n3)
    print('{}+/-{}'.format(np.mean(loss_all), 1.96*np.std(loss_all,0)/np.sqrt(len(loss_all))))
    print('{}+/-{}'.format(np.mean(loss_n1), 1.96*np.std(loss_n1,0)/np.sqrt(len(loss_n1))))
    print('{}+/-{}'.format(np.mean(loss_n3), 1.96*np.std(loss_n3,0)/np.sqrt(len(loss_n3))))

if __name__ == '__main__':
    args = parse_args()
    if not args.test:
        run(args, num_workers=1, log_interval=100, verbose=True, save_path=None)
    else:
        utils.set_seed(args.seed)
        code_root = os.path.dirname(os.path.realpath(__file__))
        mode_path = utils.get_path_from_args(args)
        mode_path = '6d5faad4c9fd9e714081674cf362544b'
        path = '{}/{}_result_files/'.format(code_root, args.task) + mode_path
        logger = utils.load_obj(path)
        uie = user_item_embed(args)
        model = interaction(uie).cuda()
        #model.load_state_dict(logger.valid_model[-1])
        model = logger.valid_model[-1]
        dataloader_test = DataLoader(Metamovie_new(args,partition='test',test_way='new_user'),#old, new_user, new_item, new_item_user
                                     batch_size=1,num_workers=args.num_workers)
        evaluate_test(args, model, dataloader_test)
    # --- settings ---



