import random
import torch.utils.data
import copy
from collections import defaultdict
import numpy as np
from random import sample
from torch.utils.data import Dataset

def filter_kcore(ratings, user_k=5, item_k=5):
    while True:
        item_count, user_count = defaultdict(int), defaultdict(int)
        for user_id, item_id, rating, timestamp in ratings:
            item_count[item_id] += 1
            user_count[user_id] += 1
        user_num, item_num = len(user_count), len(item_count)
        print(user_num, item_num)
        # filter the user_set and item_set
        user_set = set(filter(lambda x: user_count[x] >= user_k, user_count.keys()))
        item_set = set(filter(lambda x: item_count[x] >= item_k, item_count.keys()))

        if len(user_set) == user_num and len(item_set) == item_num:
            break
        ratings = list(filter(lambda x: x[0] in user_set and x[1] in item_set, ratings))
    return ratings, user_set, item_set


def task_preprocess(tasks):
    task2candidates = [set(map(lambda x: x[1], ratings)) for ratings in tasks]
    task2candidates = [list(candidates) for candidates in task2candidates]
    user2itemset = []
    for ratings in tasks:
        itemset = {}
        for user_id, item_id in ratings:
            if user_id not in itemset:
                itemset[user_id] = set()
            itemset[user_id].add(item_id)
        user2itemset.append(itemset)
    return tasks, task2candidates, user2itemset


# divide the support and evaluate data
def divide_support(task_ratings, support_limit=512, evaluate_limit=None):
    task_usertruth = []
    for ratings in task_ratings:
        task_usertruth.append({})
        for user_id, *others in ratings:
            if user_id not in task_usertruth[-1]:
                task_usertruth[-1][user_id] = []
            if others not in task_usertruth[-1][user_id]:
                task_usertruth[-1][user_id].append(others)
    divide_data = []
    for i in range(len(task_usertruth)):
        key = task_ratings[i][0]
        support_ratings, eval_ratings = [], []
        u2i = list(task_usertruth[i].items())
        random.shuffle(u2i)
        for j, (user_id, itemset) in enumerate(u2i):
            if j < len(u2i) // 2 and len(support_ratings) < support_limit:
                aim = support_ratings
            elif evaluate_limit is None or len(eval_ratings) < evaluate_limit:
                aim = eval_ratings
            else:
                break
            for item_id in itemset:
                aim.append((user_id, *item_id))
        divide_data.append((support_ratings, eval_ratings))
    return divide_data


def batch_generator(data, batch_size, shuffle=True):
    """Yield elements from data in chunks of batch_size."""
    if shuffle:
        sampler = torch.utils.data.RandomSampler(data)
    else:
        sampler = torch.utils.data.SequentialSampler(data)
    minibatch = []
    for idx in sampler:
        minibatch.append(data[idx])
        if len(minibatch) == batch_size:
            yield minibatch
            minibatch = []
    if minibatch:
        yield minibatch


def train_generator(task_ratings, task2candidates, user2itemset, batch_size, few_num=64, negative_ratio=1,
                    shuffle=True):
    user2negatives = []
    for idx in range(len(task2candidates)):
        candidates = set(task2candidates[idx])
        user2negatives.append({user_id:list(candidates - itemset) for user_id, itemset in user2itemset[idx].items()})
    while True:
        if shuffle:
            sampler = torch.utils.data.RandomSampler(task_ratings)
        else:
            sampler = torch.utils.data.SequentialSampler(task_ratings)
        for idx in sampler:
            positives = task_ratings[idx]
            candidates = task2candidates[idx]
            random.shuffle(positives)
            if len(positives) > few_num:
                support_pairs, unselected_pairs = positives[:few_num], positives[few_num:]
            else:
                support_pairs, unselected_pairs = positives[:len(positives) // 2], positives[len(positives) // 2:]
                while len(support_pairs) < few_num:
                    support_pairs.append(random.choice(support_pairs))
            
            if len(unselected_pairs) < batch_size:
                positive_pairs = [random.choice(unselected_pairs) for _ in range(batch_size)]
            else:
                positive_pairs = random.sample(unselected_pairs, batch_size)
            
            positive_users, positive_items = [pair[0] for pair in positive_pairs], [pair[1] for pair in positive_pairs]
            negative_users = copy.copy(positive_users) * negative_ratio
            negative_items = []
            for i in range(len(negative_users)):
                if len(user2negatives[idx][negative_users[i]]) > 0:
                    negative_item = random.choice(user2negatives[idx][negative_users[i]])
                else:
                    negative_item = random.choice(candidates)
                negative_items.append(negative_item)
            yield support_pairs, candidates, positive_users, positive_items, negative_users, negative_items


def train_contrast_generator(task_ratings, task2candidates, user2itemset, batch_size, few_num=64, negative_ratio=1,
                    shuffle=True):
    scenario_num = len(task2candidates)
    user2negatives = []
    for idx in range(len(task2candidates)):
        candidates = set(task2candidates[idx])
        user2negatives.append({user_id:list(candidates - itemset) for user_id, itemset in user2itemset[idx].items()})
    while True:
        if shuffle:
            sampler = torch.utils.data.RandomSampler(task_ratings)
        else:
            sampler = torch.utils.data.SequentialSampler(task_ratings)
        for idx in sampler:
            positives = task_ratings[idx]
            candidates = task2candidates[idx]
            random.shuffle(positives)
            if len(positives) > few_num:
                support_pairs, unselected_pairs = positives[:few_num], positives[few_num:]
            else:
                support_pairs, unselected_pairs = positives[:len(positives) // 2], positives[len(positives) // 2:]
                while len(support_pairs) < few_num:
                    support_pairs.append(random.choice(support_pairs))
            if len(unselected_pairs) < batch_size:
                positive_pairs = [random.choice(unselected_pairs) for _ in range(batch_size)]
            else:
                positive_pairs = random.sample(unselected_pairs, batch_size)
            positive_users, positive_items = [pair[0] for pair in positive_pairs], [pair[1] for pair in positive_pairs]
            negative_users = copy.copy(positive_users) * negative_ratio
            negative_items = []
            for i in range(len(negative_users)):
                if len(user2negatives[idx][negative_users[i]]) > 0:
                    negative_item = random.choice(user2negatives[idx][negative_users[i]])
                else:
                    negative_item = random.choice(candidates)
                negative_items.append(negative_item)

            index = np.random.choice(len(task2candidates), 64, replace=False)
            x1 = []
            x2 = []
            candidate_list = []
            for ix in index:
                positives = task_ratings[ix]
                candidates_contrast = task2candidates[ix]
                assert len(positives) >= few_num
                support_pairs_2, support_pairs_3 = sample(positives, few_num), sample(positives, few_num)
                x1.append(support_pairs_2)
                x2.append(support_pairs_3)
                candidate_list.append(candidates_contrast)
            
            yield support_pairs, candidates, positive_users, positive_items, negative_users, negative_items, x1, x2, candidate_list


def evaluate_generator(task_support, eval_user2itemset, task2candidates, few_num=8):
    def task_iterator(candidates, itemsets):
        for user_id, itemset in itemsets.items():
            if len(itemset) > 0:
                positive_users, positive_items = [], []
                for item_id in itemset:
                    positive_users.append(user_id)
                    positive_items.append(item_id)
                negative_users, negative_items = [], []
                for item_id in candidates:
                    if item_id not in itemset:
                        negative_users.append(user_id)
                        negative_items.append(item_id)
                if len(positive_users) > 0 and len(negative_users) > 0:
                    yield positive_users, positive_items, negative_users, negative_items

    for idx in range(len(eval_user2itemset)):
        if few_num is None:
            few_size = len(task_support[idx])
        else:
            few_size = few_num
        # Consistent for different iterations
        support_pairs = task_support[idx][:few_size]
        yield support_pairs, task2candidates[idx], task_iterator(task2candidates[idx], eval_user2itemset[idx])
        
def evaluate_generator_cross_domain(task_support, eval_user2itemset, task2candidates, few_num=8, source_data=None, item_padding_idx=0):
    tasks_source, task_candidate_source, user_item_source = source_data
    user2negatives_source = []
    for idx in range(len(task_candidate_source)):
        candidates = set(task_candidate_source[idx])
        user2negatives_source.append({user_id:list(candidates - itemset) for user_id, itemset in user_item_source[idx].items()})
    
    def task_iterator(candidates, itemsets):
        for user_id, itemset in itemsets.items():
            if len(itemset) > 0:
                positive_users, positive_items = [], []
                #source_positive_items = []
                for item_id in itemset:
                    positive_users.append(user_id)
                    positive_items.append(item_id)
                    #if user_id in user_item_source[0].keys():                 
                    #    source_positive_items.append(random.choice(list(user_item_source[0][user_id])))
                    #else:
                    #    source_positive_items.append(item_padding_idx)
                negative_users, negative_items = [], []
                #source_negative_items = []
                for item_id in candidates:
                    if item_id not in itemset:
                        negative_users.append(user_id)
                        negative_items.append(item_id)
                        #if user_id in user2negatives_source[0].keys():                 
                        #    source_n_item = random.choice(user2negatives_source[0][user_id])
                        #else:
                        #    source_n_item = item_padding_idx
                        #source_negative_items.append(source_n_item)
                if len(positive_users) > 0 and len(negative_users) > 0:
                    #yield positive_users, positive_items, source_positive_items, negative_users, negative_items, source_negative_items
                     yield positive_users, positive_items, negative_users, negative_items

    for idx in range(len(eval_user2itemset)):
        if few_num is None:
            few_size = len(task_support[idx])
        else:
            few_size = few_num
        # Consistent for different iterations
        support_pairs = task_support[idx][:few_size]
        #source_support_pairs = []
        
        #for d in support_pairs:
        #    user = d[0]
        #    if user in user_item_source[0].keys():
        #        item = random.choice(list(user_item_source[0][user]))
        #    else:
        #        item = item_padding_idx
        #    source_support_pairs.append((user, item))
        
        yield support_pairs, task2candidates[idx], (tasks_source, task_candidate_source, user_item_source, user2negatives_source), task_iterator(task2candidates[idx], eval_user2itemset[idx])

def evaluate_generator_item(task_support, eval_user2itemset, task2candidates, few_num=8):
    def task_iterator(candidates, itemsets):
        for user_id, itemset in itemsets.items():
            if len(itemset) > 0:
                positive_users, positive_items = [], []
                for item_id in itemset:
                    positive_users.append(user_id)
                    positive_items.append(item_id)
                negative_users, negative_items = [], []
                for item_id in candidates:
                    if item_id not in itemset:
                        negative_users.append(user_id)
                        negative_items.append(item_id)
                if len(positive_users) > 0 and len(negative_users) > 0:
                    yield positive_users, positive_items, negative_users, negative_items

    for idx in range(len(eval_user2itemset)):
        if few_num is None:
            few_size = len(task_support[idx])
        else:
            few_size = few_num
        # Consistent for different iterations
        support_pairs = task_support[idx][:few_size]
        yield support_pairs, task2candidates[idx], task_iterator(task2candidates[idx], eval_user2itemset[idx]), idx
        
class pretrain_dataset(Dataset):
    def __init__(self, train_data):
        super(pretrain_dataset, self).__init__()
        tasks, task_candidate, user_item = train_data
        self.data = tasks
        self.data_candidate = task_candidate
        self.user_item = user_item
        self.task = len(self.data)
        self.all_samples = int(sum([len(d) for d in self.data]))
        self.index = np.random.choice(len(self.data), self.all_samples)
        
        user2negatives = []
        for idx in range(len(task_candidate)):
            candidates = set(task_candidate[idx])
            user2negatives.append({user_id:list(candidates - itemset) for user_id, itemset in user_item[idx].items()})
            
        self.user2negatives = user2negatives
        
    def __getitem__(self, item):   
        idx = self.index[item]
        positives = self.data[idx]
        user_item = positives[np.random.choice(len(positives))]
        user = user_item[0]
        candidates = self.data_candidate[idx]
        positive_user_item = torch.tensor(user_item)
        if len(self.user2negatives[idx][user]) > 0:
            negative_item = random.choice(self.user2negatives[idx][user])
        else:
            negative_item = random.choice(self.data_candidate[idx])
        negative_user_item = torch.tensor([user, negative_item])
        return positive_user_item, negative_user_item
        
    def __len__(self):
        return self.all_samples
    
class cross_domain_dataset(Dataset):
    def __init__(self, train_data, source_data, item_padding_idx):
        super(cross_domain_dataset, self).__init__()
        self.item_padding_idx = item_padding_idx
        tasks, task_candidate, user_item = train_data
        tasks_source, task_candidate_source, user_item_source = source_data
        self.data = tasks
        self.data_candidate = task_candidate
        self.data_candidate_source = task_candidate_source
        self.user_item = user_item
        self.source_user_item = user_item_source
        self.task = len(self.data)
        self.all_samples = int(sum([len(d) for d in self.data]))
        self.index = np.random.choice(len(self.data), self.all_samples)
        
        self.data_source = tasks_source
        self.data_candidate_source = task_candidate_source
        
        user2negatives = []
        for idx in range(len(task_candidate)):
            candidates = set(task_candidate[idx])
            user2negatives.append({user_id:list(candidates - itemset) for user_id, itemset in user_item[idx].items()})
            
        self.user2negatives = user2negatives
        
        user2negatives_source = []
        for idx in range(len(task_candidate_source)):
            candidates = set(task_candidate_source[idx])
            user2negatives_source.append({user_id:list(candidates - itemset) for user_id, itemset in user_item_source[idx].items()})
            
        self.user2negatives_source = user2negatives_source
        
    def __getitem__(self, item):   
        idx = self.index[item]
        positives = self.data[idx]
        user_item = positives[np.random.choice(len(positives))]
        user = user_item[0]
        candidates = self.data_candidate[idx]
        positive_user_item = torch.tensor(user_item)
        if len(self.user2negatives[idx][user]) > 0:
            negative_item = random.choice(self.user2negatives[idx][user])
        else:
            negative_item = random.choice(self.data_candidate[idx])
        negative_user_item = torch.tensor([user, negative_item])
        
        positives_source = self.data_source[0]
        #user_item_source = positives[np.random.choice(len(positives_source))]
        user_source = user
        if user_source in self.source_user_item[0].keys():
            item_source = random.choice(list(self.source_user_item[0][user_source]))
        else:
            item_source = self.item_padding_idx
        source_positive_user_item = torch.tensor([user_source, item_source])
        
        if user_source in self.user2negatives_source[0]:
            if len(self.user2negatives_source[0][user_source]) > 0:
                source_negative_item = random.choice(self.user2negatives_source[0][user_source])
            else:
                source_negative_item = random.choice(self.data_candidate_source[0])
        else:
            source_negative_item = self.item_padding_idx
        source_negative_user_item = torch.tensor([user_source, source_negative_item])
        return positive_user_item, negative_user_item, source_positive_user_item, source_negative_user_item
        
    def __len__(self):
        return self.all_samples