import torch
import torch.nn as nn
import torch.optim as optim
import time, functools, itertools, os, argparse
import logging
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils import unserialize, serialize, divide_dataset
from evaluate import topNRecall, multi_mean_measure
from utils import NeighborDict, UserItemEmbeds, filter_statedict
import loss
from dataset import train_generator, evaluate_generator, task_preprocess, divide_support
from modules_multimode import EmbedRecommender, InteractionRecommender, HybridRecommender, Meta_softmodule_cross_InteractionRecommender,Meta_mlp_cross_InteractionRecommender
import random
import os 


#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
logging.basicConfig(format='%(asctime)s - %(levelname)s -   '
                           '%(message)s',
                    datefmt='%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_embedding(user_embedding, item_embedding):
    user_padding_idx = user_embedding.size(0)
    user_embedding = torch.cat(
        (user_embedding, torch.zeros(1, user_embedding.size(1))), dim=0)
    item_padding_idx = item_embedding.size(0)
    item_embedding = torch.cat(
        (item_embedding, torch.zeros(1, item_embedding.size(1))), dim=0)

    user_embedding = torch.nn.Embedding.from_pretrained(user_embedding)
    item_embedding = torch.nn.Embedding.from_pretrained(item_embedding)

    for p in user_embedding.parameters():
        p.requires_grad = False
    for p in item_embedding.parameters():
        p.requires_grad = False

    useritem_embeds = UserItemEmbeds(user_embedding, item_embedding)
    return useritem_embeds, user_padding_idx, item_padding_idx


def get_neighbor_dict(back_ratings, user_dict, item_dict, user_padding_idx, item_padding_idx):
    user_neighbor_dict = [set() for _ in range(len(user_dict))]
    item_neighbor_dict = [set() for _ in range(len(item_dict))]
    for user_id, item_id, _, _ in back_ratings:
        user_id = user_dict[user_id]
        item_id = item_dict[item_id]
        user_neighbor_dict[user_id].add(item_id)
    user_neighbor_dict = [list(items) for items in user_neighbor_dict]
    print("user_neighbor_dict",user_neighbor_dict)
    user_neighbor_dict = NeighborDict(user_neighbor_dict, max_degree=192, padding_idx=item_padding_idx)
    item_neighbor_dict = [list(users) for users in item_neighbor_dict]
    item_neighbor_dict = NeighborDict(item_neighbor_dict, max_degree=192, padding_idx=user_padding_idx)
    return user_neighbor_dict, item_neighbor_dict


def get_data(train_data, test_data, user_dict, item_dict):
    #print('train_data',train_data[1][0])
    i=0
    #print('user_dict.keys',type(user_dict.keys()))
    '''
    for key in list(user_dict.keys()):
        if i > 200:
            continue
        print(key,user_dict[key])
        i+=1
    '''
    train_data = [
        list(
            map(lambda x: (user_dict[x[0]], item_dict[x[1]]),
                filter(lambda x: x[0] in user_dict and x[1] in item_dict, value))) for key, value in train_data
    ]
    print("train_data",len(train_data))
    train_data, valid_data, _ = divide_dataset(train_data, valid_ratio=0.05, test_ratio=0.0)
    logger.info(
        "train tasks: {} ratings: {} valid tasks: {} ratings: {}".format(len(train_data), sum(map(len, train_data)),
                                                                         len(valid_data), sum(map(len, valid_data))))
    train_data = task_preprocess(train_data)
    valid_data = divide_support(valid_data)
    valid_support, valid_eval = list(map(lambda x: x[0], valid_data)), list(map(lambda x: x[1], valid_data))
    valid_candidates = [
        list({item_id
              for user_id, item_id in task})
        for task in map(lambda x: itertools.chain(*x), zip(valid_support, valid_eval))
    ]
    _, _, valid_truth = task_preprocess(valid_eval)
    test_support = [
        list(
            map(lambda x: (user_dict[x[0]], item_dict[x[1]]),
                filter(lambda x: x[0] in user_dict and x[1] in item_dict, support)))
        for key, support, evaluate in test_data
    ]
    test_eval = [
        list(
            map(lambda x: (user_dict[x[0]], item_dict[x[1]]),
                filter(lambda x: x[0] in user_dict and x[1] in item_dict, evaluate)))
        for key, support, evaluate in test_data
    ]

    test_candidates = [
        list({item_id
              for user_id, item_id in task})
        for task in map(lambda x: itertools.chain(*x), zip(test_support, test_eval))
    ]

    _, _, test_truth = task_preprocess(test_eval)
    return train_data, (valid_support, valid_candidates, valid_truth), (test_support, test_candidates, test_truth)


def get_model(useritem_embeds, user_neighbor_dict, item_neighbor_dict, criterion, config):
    print("useritem_embeds",type(useritem_embeds))
    model_type = config['recommender'].pop('model_type').lower()
    hybrid = config['hybrid']
    if model_type == 'mapping':
        model = EmbedRecommender(useritem_embeds, **config['recommender'])
    elif model_type == 'interaction':
        model = InteractionRecommender(useritem_embeds, config['recommender'])
    elif model_type == 'hybrid':
        model = HybridRecommender(useritem_embeds, **config['recommender'])
    elif model_type == 'meta_soft':
        model = Meta_softmodule_cross_InteractionRecommender(useritem_embeds, way='dot')#
    else:
        model = Meta_mlp_cross_InteractionRecommender(useritem_embeds, way='dot', condition_way='film')
    else:
        raise NotImplemented

    return model

def regularizer(o1,o2,o3):
    num = 4
    
    var_1 = 0.
    var_2 = 0.
    var_3 = 0.
    for k in range(num):
        var_1 += torch.var(o1[:,:,k])
        var_2 += torch.var(o2[:,:,k])
        if k ==0:
            var_3 += torch.var(o3[:,k])
    res = (var_1 + var_2 +var_3)
    print(res)
    return res



def evaluate(data_loader, model, useritem_embeds, user_neighbor_dict, item_neighbor_dict, device):
    for data in data_loader:
        support_pairs, candidates, task_iterator = data
        support_pairs = torch.tensor(support_pairs, device=device)
        support_users, support_items = support_pairs[:, 0], support_pairs[:, 1]
        support_users = user_neighbor_dict(support_users)
        support_items = item_neighbor_dict(support_items)

        positive_support_users = support_users
        positive_support_items = support_items
        
        negative_user = []
        negative_items = []
        support_num = len(support_pairs)
        users = support_pairs[:, 0]
        
        negative_user_index = np.random.choice(range(support_num), 2 * support_num)
        for idx in negative_user_index:
            negative_user.append(users[idx])
              
        for idx in range(len(negative_user)):
            negative_item = random.choice(candidates)
            while negative_item == support_pairs[negative_user_index[idx], 1]:
                negative_item = random.choice(candidates)
            negative_items.append(negative_item)
        negative_support_items = item_neighbor_dict(torch.tensor(negative_items, device=device))
        negative_support_users = user_neighbor_dict(torch.tensor(negative_user, device=device))
        
        model.set_context(positive_support_users, positive_support_items, negative_support_users, negative_support_items)

        with torch.no_grad():
            for positive_users, positive_items, negative_users, negative_items in task_iterator:
                query_users = positive_users[:1]
                query_items = positive_items + negative_items
                query_users = torch.tensor(query_users, device=device)
                query_items = torch.tensor(query_items, device=device)
                positive_num = len(positive_users)
                values = model(user_neighbor_dict(query_users), item_neighbor_dict(query_items))
                _, index = values.sort(dim=0, descending=True)
                index = index.tolist()
                yield (range(positive_num), index)


measure_dict = {
    'Recall_1': functools.partial(topNRecall, topn=1),
    'Recall_3': functools.partial(topNRecall, topn=3),
    'Recall_5': functools.partial(topNRecall, topn=5),
    'Recall_10': functools.partial(topNRecall, topn=10),
    'Recall_20': functools.partial(topNRecall, topn=20),
    'Recall_50': functools.partial(topNRecall, topn=50),
    'Recall_100': functools.partial(topNRecall, topn=100)
}

measure_keys, measure_funcs = list(map(lambda x: x[0], measure_dict.items())), list(
    map(lambda x: x[1], measure_dict.items()))


def train(modules, criterion, optimizer, scheduler, task_iter, valid_data, test_data, device,few_num, step_penalty,
          train_steps,
          data_directory, parameters, writer=None):
    model, useritem_embeds, user_neighbor_dict, item_neighbor_dict = modules
    #parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    test_support, test_candidates, test_truth = test_data
    valid_support, valid_candidates, valid_truth = valid_data
    running_loss, running_steps, loss_descend = 0.0, 0, 0.0
    best_values = {}
    model.train()

    #gpu_tracker.track()
    for batch_id, data in tqdm(task_iter):

        if batch_id > train_steps:
            break
        support_pairs, candidates, positive_users, positive_items, negative_users, negative_items = data
        positive_num = len(positive_items)
        support_pairs = torch.tensor(support_pairs, device=device)
        support_users, support_items = support_pairs[:, 0], support_pairs[:, 1]
        #print("before support_users",support_users)
        support_users = user_neighbor_dict(support_users)
        #print("after support_users",support_users)
        support_items = item_neighbor_dict(support_items)
        query_users = user_neighbor_dict(torch.tensor(positive_users + negative_users, device=device))
        query_items = item_neighbor_dict(torch.tensor(positive_items + negative_items, device=device))

        positive_support_users = support_users
        positive_support_items = support_items
        
        negative_user = []
        negative_items = []
        support_num = len(support_pairs)
        users = support_pairs[:, 0]
        
        negative_user_index = np.random.choice(range(support_num), 2 * support_num)
        for idx in negative_user_index:
            negative_user.append(users[idx])
              
        for idx in range(len(negative_user_index)):
            negative_item = random.choice(candidates)
            while negative_item == support_pairs[negative_user_index[idx], 1]:
                negative_item = random.choice(candidates)
            negative_items.append(negative_item)
        negative_support_items = item_neighbor_dict(torch.tensor(negative_items, device=device))
        negative_support_users = user_neighbor_dict(torch.tensor(negative_user, device=device))
        #print("positive_support_users",positive_support_users)
        
        model.set_context(positive_support_users, positive_support_items, negative_support_users, negative_support_items)

        values = model(query_users, query_items, with_attr=False)
        o1,o2,o3 = model.hypernet(model.context_embed)
        positive_values, negative_values = values[:positive_num], values[positive_num:]
        final_loss = criterion(positive_values, negative_values)
        regularizer_loss = 0
        total_loss = 0.0
        total_loss = final_loss 
        optimizer.zero_grad()
        total_loss.backward()
        #gpu_tracker.track()  
        #xtorch.nn.utils.clip_grad_value_(parameters, 0.25)
        optimizer.step()

        running_loss += final_loss.item()
        running_steps += 0
        loss_descend += 0

        if (batch_id + 1) % 1000 == 0:
            if writer is not None:
                writer.add_scalar('loss', running_loss / 1000, batch_id)
                writer.add_scalar('step', running_steps / 1000, batch_id)
                writer.add_scalar('descend', loss_descend / 1000, batch_id)
            print("loss@%5d: %.4f" % (batch_id + 1, running_loss / 1000))
            running_loss, running_steps, loss_descend = 0.0, 0, 0.0
        if (batch_id + 1) % 5000 == 0:
            model.eval()
            valid_values = multi_mean_measure(
                evaluate(
                    evaluate_generator(valid_support, valid_truth, valid_candidates,
                                       few_num=few_num), model, useritem_embeds, user_neighbor_dict,
                    item_neighbor_dict, device), measure_funcs)
            test_values = multi_mean_measure(
                evaluate(
                    evaluate_generator(test_support, test_truth, test_candidates,
                                       few_num=few_num), model, useritem_embeds, user_neighbor_dict,
                    item_neighbor_dict, device), measure_funcs)
            test_output, valid_output = [], []
            update = False
            for i, key in enumerate(measure_keys):
                valid_value, test_value = valid_values[i], test_values[i]
                if key not in best_values or valid_value > best_values[key]:
                    best_values[key] = valid_value
                    if key == 'Recall_20':
                        update = True
                if writer is not None:
                    writer.add_scalar(key, test_value, batch_id)
                valid_output.append("%s:%.4f" % (key, valid_value))
                test_output.append("%s:%.4f" % (key, test_value))
            logger.info("  ".join(valid_output))
            if update:
                logger.info("  ".join(test_output))
                torch.save(
                    filter_statedict(model), os.path.join(data_directory,
                                                                   str(batch_id + 1) + ".dict"))
            scheduler.step(test_values[measure_keys.index('Recall_10')])
            model.train()

def set_seed(seed, cudnn=True):
    """
    Seed everything we can!
    Note that gym environments might need additional seeding (env.seed(seed)),
    and num_workers needs to be set to 1.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # note: the below slows down the code but makes it reproducible
    if (seed is not None) and cudnn:
        torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    print('@@@@@@@@@@')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--root_directory', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--comment', type=str, default='init')
    parser.add_argument('--dataset', type=str, default='ali-movielens')
    args = parser.parse_args()

    config = unserialize(args.config)
    seed = config['seed']
    set_seed(seed)
    print('seed',seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda:0')
    root_directory_higher = args.root_directory
    dataset = args.dataset.split('-')
    #root_directorys = [root_directory_higher+'ali',root_directory_higher+'amazon',root_directory_higher+'movielens']
    root_directorys = [root_directory_higher+e for e in dataset]
    print(root_directorys)
    # load data
    user_embedding, item_embedding= torch.empty(1),torch.empty(1)
    train_data,test_data,valid_data =0,0,0
    start_index_user,start_index_item = 0,0
    for root_directory in root_directorys:

        if "movielens" in root_directory:
            _user_embedding = torch.from_numpy(
            unserialize(os.path.join(root_directory, "embeddings/user_embeddings.npy")).astype(np.float32))
            print(" original each user_embedding",_user_embedding.shape)
            _user_embedding = torch.cat([_user_embedding,_user_embedding],dim=1)

            _item_embedding = torch.from_numpy(
            unserialize(os.path.join(root_directory, "embeddings/item_embeddings.npy")).astype(np.float32))
            print("original each item_embedding",_item_embedding.shape)
            _item_embedding =  torch.cat([_item_embedding, _item_embedding],dim=1)
        else:
            _user_embedding = torch.from_numpy(
                unserialize(os.path.join(root_directory, "embeddings/user_embeddings.npy")).astype(np.float32))

            _item_embedding = torch.from_numpy(
                unserialize(os.path.join(root_directory, "embeddings/item_embeddings.npy")).astype(np.float32))

        print("each user_embedding",_user_embedding.shape)
        print("each item_embedding",_item_embedding.shape)

       
        if user_embedding.shape[0]==1:
            user_embedding = _user_embedding
            item_embedding = _item_embedding
        else:
            #print("user_embeding",user_embedding.shape,"_user_embedding",_user_embedding.shape)
            user_embedding = torch.cat([user_embedding,_user_embedding],dim=0)
            item_embedding = torch.cat([item_embedding,_item_embedding],dim=0)
        _train_data = unserialize(os.path.join(root_directory, "train_data/train_data"))
        _test_data = unserialize(os.path.join(root_directory, "train_data/test_data"))
        user_dict = unserialize(os.path.join(root_directory, "embeddings/user_dict"))

        for key in user_dict.keys():
            user_dict[key] += start_index_user

        item_dict = unserialize(os.path.join(root_directory, "embeddings/item_dict"))
        for key in item_dict.keys():
            item_dict[key] += start_index_item

        start_index_user += _user_embedding.shape[0]
        start_index_item += _item_embedding.shape[0]

        if config['recommender'].get('user_graph', False) or config['recommender'].get('item_graph', False):
            back_ratings = unserialize(os.path.join(root_directory, "train_data/back_ratings"))
            user_neighbor_dict, item_neighbor_dict = get_neighbor_dict(back_ratings, user_dict, item_dict)
        # data preprocess
        
        _train_data, _valid_data, _test_data = get_data(_train_data, _test_data, user_dict, item_dict)
    
        if train_data == 0:
            train_data,valid_data,test_data = list(_train_data),list(_valid_data),list(_test_data)
            print(len(test_data[0]))
        else:
            for i in range(len(train_data)):
                train_data[i] = train_data[i] + _train_data[i]
                valid_data[i] = valid_data[i] + _valid_data[i]
                test_data[i] = test_data[i] + _test_data[i]
            print(len(test_data[0]), len(test_data[1]), len(test_data[2]))

    print("total user_embedding",user_embedding.shape)
    print("total item_embedding",item_embedding.shape)
    useritem_embeds, user_padding_idx, item_padding_idx = get_embedding(user_embedding, item_embedding)
    train_data,valid_data,test_data = tuple(train_data),tuple(valid_data),tuple(test_data)
    if not config['recommender'].get('user_graph', False):
        user_neighbor_dict = NeighborDict(None)
    if not config['recommender'].get('item_graph', False):
        item_neighbor_dict = NeighborDict(None)
    # define model
    criterion = loss.__getattribute__(
        config['training']['loss'])(**config['training']['loss_config'])
    model = get_model(useritem_embeds, user_neighbor_dict, item_neighbor_dict, criterion, config)
    model.to(device)
    # optimizer
    config.setdefault('lr', {})
    for key in ['stop_lr', 'init_lr', 'update_lr']:
        config['lr'].setdefault(key, config['optim']['lr'])

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optim.Adam(parameters, **config['optim'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=2,
                                                           verbose=True, min_lr=1e-6)
    # save
    save_directory = root_directory
    if config['save']:
        project_name = args.comment
        data_directory = os.path.join(save_directory, "log", "-".join((project_name, time.strftime("%m-%d-%H"))))
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        log_file = os.path.join(data_directory, "_".join(("log", time.strftime("%m-%d-%H-%M"))))
        writer = SummaryWriter(log_file, comment='Normal')
        serialize(config, os.path.join(data_directory, "config.json"), in_json=True)
    else:
        data_directory, writer = None, None
    batch_size = config['training']['batch_size']
    task_iter = enumerate(
        train_generator(*train_data, batch_size, negative_ratio=config['training']['negative_ratio'],
                        few_num=config['few_num']))
    modules = (model, useritem_embeds, user_neighbor_dict, item_neighbor_dict)
    train(modules, criterion, optimizer, scheduler, task_iter, valid_data, test_data, device,few_num=config['few_num'],
          step_penalty=config['training']['step_penalty'], train_steps=config['training']['max_steps'],
          data_directory=data_directory, parameters=parameters, writer=writer)
