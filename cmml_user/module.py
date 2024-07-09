import torch
import numpy as np
from copy import deepcopy

from torch.autograd import Variable
from torch.nn import functional as F
from collections import OrderedDict

from embeddings import item, user
import torch.nn as nn

class Linear(nn.Linear): #used in MAML to forward input with fast weight 
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__(in_features, out_features)
        self.weight.fast = None #Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast) #weight.fast (fast weight) is the temporaily adapted weight
        else:
            out = super(Linear, self).forward(x)
        return out

class user_preference_estimator(torch.nn.Module):
    def __init__(self, config):
        super(user_preference_estimator, self).__init__()
        self.embedding_dim = config.embedding_dim
        self.fc1_in_dim = config.embedding_dim * 8
        self.fc2_in_dim = config.first_fc_hidden_dim
        self.fc2_out_dim = config.second_fc_hidden_dim

        self.item_emb = item(config)
        self.user_emb = user(config)
        self.fc1 = Linear(self.fc1_in_dim, self.fc2_in_dim)
        self.fc2 = Linear(self.fc2_in_dim, self.fc2_out_dim)
        self.linear_out = Linear(self.fc2_out_dim, 1)
        self.final_part = nn.Sequential(self.fc1, nn.ReLU(), self.fc2, nn.ReLU(), self.linear_out)
    
    def forward(self, x, training = True):
        rate_idx = x[:, 0]
        genre_idx = x[:, 1:26]
        director_idx = x[:, 26:2212]
        actor_idx = x[:, 2212:10242]
        gender_idx = x[:, 10242]
        age_idx = x[:, 10243]
        occupation_idx = x[:, 10244]
        area_idx = x[:, 10245]

        item_emb = self.item_emb(rate_idx, genre_idx, director_idx, actor_idx)
        user_emb = self.user_emb(gender_idx, age_idx, occupation_idx, area_idx)
        x = torch.cat((item_emb, user_emb), 1)
        x = self.final_part(x)
        return x
    
class soft_module_route(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=64, num=4):
        super(soft_module_route, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 128)
        
        self.fc12 = nn.Linear(128, 16)
        self.fc22 = nn.Linear(128, 4)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        o1 = self.fc12(x)
        o1 = F.softmax(o1.reshape(x.shape[0], 4, 4), dim=1)#batch*4*4
        x = F.relu(self.fc2(x))
        o2 = self.fc22(x)
        o2 = F.softmax(o2.reshape(x.shape[0], 4, 1), dim=1)

        return o1, o2
    
class soft_module(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=32, num=4):
        super(soft_module, self).__init__()
        self.fc1 = nn.ModuleList([nn.Linear(in_dim, hidden_dim) for _ in range(num)])
        self.fc2 = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num)])
        self.final_layer = nn.Linear(hidden_dim, 1)
        self.num = num
    def forward(self, x, c1, c2):
        o1 = []
        for i in range(len(self.fc1)):
            o1.append(F.relu(self.fc1[i](x)))#[batch * 64 * 4] batch * 4 * 4
        o1 = torch.stack(o1, dim=-1)
        o2 = []
        in2 = o1 @ c1#b*64*4
        for i in range(self.num):
            o2.append(F.relu(self.fc2[i](in2[:,:,i])))
        o2 = torch.stack(o2, dim=-1)
        in3 = (o2 @ c2).squeeze(dim=-1)#b*64*1
        out = self.final_layer(in3)
        return out
    
class Mean_encoder(nn.Module):
    def __init__(self, useritem_embeds):
        super(Mean_encoder, self).__init__()
        self.user_item_embed = useritem_embeds
        self.score_mlp = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 64))
        self.mlp = nn.Sequential(nn.Linear(256+64, 64), nn.ReLU(), nn.Linear(64, 256))

    def forward(self, x_spt, y_spt):
        x = self.user_item_embed(x_spt).detach()#batch * 256
        y = self.score_mlp(y_spt)
        embeds = torch.cat([x, y], dim=-1)#batch * 257
        
        meta_embeds = self.mlp(embeds)

        return meta_embeds.mean(dim=0) #1 * 128    
    
class LSTM_encoder(nn.Module):
    def __init__(self, useritem_embeds):
        super(LSTM_encoder, self).__init__()
        self.user_item_embed = useritem_embeds
        self.mlp = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 256))
        self.gru = nn.GRU(input_size=256+64, hidden_size=64)
        self.score_mlp = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 64))

    def forward(self, x_spt, y_spt):
        x = self.user_item_embed(x_spt).detach()#batch * 256
        y = self.score_mlp(y_spt)
        embeds = torch.cat([x, y], dim=-1)#batch * 257
        embeds = embeds.unsqueeze(dim=1)
        
        output, h_n = self.gru(embeds.transpose(1,0))
        meta_embeds = self.mlp(output[0,-1])
        return meta_embeds  #256

class film_condition_mlp(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=64):
        super(film_condition_mlp, self).__init__()
        self.f1 = nn.Linear(in_dim, hidden_dim)
        self.f2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.l_relu = nn.LeakyReLU()
        self.hidden_dim= hidden_dim
        self.in_dim = in_dim
        
    def forward(self, x, o1, o2):
        x = self.f1(x)#self.l_relu(self.f1(x))
        weight = o1[:, :self.hidden_dim]
        bias = o1[:, self.hidden_dim:]
        x = x * weight + bias
        x = self.l_relu(x)
        x = self.f2(x)
        weight = o2[:, :self.hidden_dim]
        bias = o2[:, self.hidden_dim:]
        x = x * weight + bias
        x = self.l_relu(x)      
        return x
    
class mlp_hypernet(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=256, out_dim=128):
        super(mlp_hypernet, self).__init__()
        self.f1 = nn.Linear(in_dim, hidden_dim)
        self.f2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.f11 =  nn.Linear(hidden_dim, out_dim)
        self.f21 =  nn.Linear(hidden_dim, out_dim) 
        self.relu = nn.ReLU()
        
    def forward(self, inp):
        x = self.relu(self.f1(inp))
        o1 = self.f11(x)
        x = self.relu(self.f2(x))
        o2 = self.f21(x)
        return o1, o2

class user_item_embed(nn.Module):
    def __init__(self, config):
        super(user_item_embed, self).__init__()
        self.user_emb = user(config)
        self.item_emb = item(config)
    def forward(self, x):
        #print(123123123)
        #print('111', x.shape)
        rate_idx = x[:, 0]
        genre_idx = x[:, 1:26]
        director_idx = x[:, 26:2212]
        actor_idx = x[:, 2212:10242]
        gender_idx = x[:, 10242]
        age_idx = x[:, 10243]
        occupation_idx = x[:, 10244]
        area_idx = x[:, 10245]

        item_emb = self.item_emb(rate_idx, genre_idx, director_idx, actor_idx)
        user_emb = self.user_emb(gender_idx, age_idx, occupation_idx, area_idx)
        x = torch.cat((item_emb, user_emb), 1)
        return x#batch * 256
        
class meta_mlp_context(torch.nn.Module):
    def __init__(self, config):
        super(meta_mlp_context, self).__init__()
        self.embedding_dim = config.embedding_dim
        self.fc1_in_dim = config.embedding_dim * 8
        self.fc2_in_dim = config.first_fc_hidden_dim
        self.fc2_out_dim = config.second_fc_hidden_dim
        
        self.user_item_embed = user_item_embed(config)
        if config.context_encoder == 'mean':
            self.context_encoder = Mean_encoder(self.user_item_embed)
        else:
            self.context_encoder = LSTM_encoder(self.user_item_embed)
        self.mlp = film_condition_mlp()
        self.hypernet = mlp_hypernet(hidden_dim=config.mlp_hyper_hidden_dim)
        self.final = nn.Linear(64, 1)
        
        self.context_embed = None
    
    def forward(self, x, training = True):
        x = self.user_item_embed(x)#bs*256
        
        #print(self.context_embed.shape)
        c_embed = self.context_embed[None,:].repeat(x.shape[0], 1)#bs*256
        embeds = x * c_embed
        #embeds = c_embed
        
        o1, o2 = self.hypernet(embeds)#bs*128, bs*128
        #print(o1.shape,o2.shape)
        x = self.mlp(x, o1, o2)
        
        x = self.final(x)
        return x
    
    def set_context(self, x_spt, y_spt):
        context_embed = self.context_encoder(x_spt, y_spt)  # 1 * 128
        self.context_embed = context_embed# 256
             
class meta_soft_module(torch.nn.Module):
    def __init__(self, config):
        super(meta_soft_module, self).__init__()
        self.embedding_dim = config.embedding_dim
        self.fc1_in_dim = config.embedding_dim * 8
        self.fc2_in_dim = config.first_fc_hidden_dim
        self.fc2_out_dim = config.second_fc_hidden_dim
        
        self.user_item_embed = user_item_embed(config)
        self.context_encoder = Mean_encoder(self.user_item_embed)
        self.mlp = soft_module()
        self.hypernet = soft_module_route()
        
        self.context_embed = None
    
    def forward(self, x, training = True):
        x = self.user_item_embed(x)#bs*256
        
        #print(self.context_embed.shape)
        c_embed = self.context_embed[None,:].repeat(x.shape[0], 1)#bs*256
        embeds = x * c_embed
        o1, o2 = self.hypernet(embeds)#bs*128, bs*128
        #print(o1.shape,o2.shape)
        x = self.mlp(x, o1, o2)
        return x
    
    def set_context(self, x_spt, y_spt):
        context_embed = self.context_encoder(x_spt, y_spt)  # 1 * 128
        self.context_embed = context_embed# 256
        
        
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pdb


class DeepNet(nn.Module):
    """
    Deep part of Cross and Deep Network
    All of the layer in this module are full-connection layers
    """

    def __init__(self, input_feature_num, deep_layer: list):
        """
        :param input_feature_num: total num of input_feature, including of the embedding feature and dense feature
        :param deep_layer: a list contains the num of each hidden layer's units
        """
        super(DeepNet, self).__init__()
        fc_layer_list = []
        fc_layer_list.append(nn.Linear(input_feature_num, deep_layer[0]))
        #fc_layer_list.append(nn.BatchNorm1d(deep_layer[0], affine=False))
        fc_layer_list.append(nn.ReLU(inplace=True))
        for i in range(1, len(deep_layer)):
            fc_layer_list.append(nn.Linear(deep_layer[i - 1], deep_layer[i]))
            #fc_layer_list.append(nn.BatchNorm1d(deep_layer[i], affine=False))
            fc_layer_list.append(nn.ReLU(inplace=True))
        self.deep = nn.Sequential(*fc_layer_list)

    def forward(self, x):
        dense_output = self.deep(x)
        return dense_output


class CrossNet(nn.Module):
    """
    Cross layer part in Cross and Deep Network
    The ops in this module is x_0 * x_l^T * w_l + x_l + b_l for each layer l, and x_0 is the init input of this module
    """

    def __init__(self, input_feature_num, cross_layer: int):
        """
        :param input_feature_num: total num of input_feature, including of the embedding feature and dense feature
        :param cross_layer: the number of layer in this module expect of init op
        """
        super(CrossNet, self).__init__()
        self.cross_layer = cross_layer + 1  # add the first calculate
        weight_w = []
        weight_b = []
        for i in range(self.cross_layer):
            weight_w.append(nn.Parameter(torch.nn.init.normal_(torch.empty(input_feature_num))))
            weight_b.append(nn.Parameter(torch.nn.init.normal_(torch.empty(input_feature_num))))
            #batchnorm.append(nn.BatchNorm1d(input_feature_num, affine=False))
        self.weight_w = nn.ParameterList(weight_w)
        self.weight_b = nn.ParameterList(weight_b)

    def forward(self, x):
        output = x
        x = x.reshape(x.shape[0], -1, 1)
        for i in range(self.cross_layer):
            output = torch.matmul(torch.bmm(x, torch.transpose(output.reshape(output.shape[0], -1, 1), 1, 2)),self.weight_w[i]) + self.weight_b[i] + output
        return output


class DCN(nn.Module):
    def __init__(self, useritem_embedding):
        super(DCN, self).__init__()
        self.useritem = useritem_embedding
        self.l = nn.Linear(256, 64)#, nn.ReLU(), nn.Linear(128, 64))#nn.Sequential(nn.Linear(256, 64))
        self.cross = CrossNet(64, 3)
        self.deep = DeepNet(64, [64,64])
        self.out = nn.Linear(64 + 64, 1)
    def forward(self, x):
        o = self.useritem(x)
        o = self.l(o)
        cross = self.cross(o)
        deep = self.deep(o)
        x = torch.cat([deep, cross], dim=-1)
        x = self.out(x)
        return x
    
class meta_final_context(torch.nn.Module):
    def __init__(self, config):
        super(meta_final_context, self).__init__()
        self.embedding_dim = config.embedding_dim
        self.fc1_in_dim = config.embedding_dim * 8
        self.fc2_in_dim = config.first_fc_hidden_dim
        self.fc2_out_dim = config.second_fc_hidden_dim
        
        self.user_item_embed = user_item_embed(config)
        self.context_encoder = Mean_encoder(self.user_item_embed)#LSTM_encoder(self.user_item_embed)
        self.hypernet = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64,65))
        self.main = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64,64))
        
        self.context_embed = None
    
    def forward(self, x, training = True):
        x = self.user_item_embed(x)#bs*256
        
        #print(self.context_embed.shape)
        c_embed = self.context_embed[None,:].repeat(x.shape[0], 1)#bs*256
        embeds = x * c_embed
        #embeds = c_embed
        
        final = self.hypernet(embeds)#bs*128, bs*128
        #print(o1.shape,o2.shape)
        x = self.main(x)#batch * 64
        
        weight = final[:, :64]
        bias = final[:, -1]
        
        x = torch.sum(x * weight, dim=-1) + bias
        
        return x
    
    def set_context(self, x_spt, y_spt):
        context_embed = self.context_encoder(x_spt, y_spt)  # 1 * 128
        self.context_embed = context_embed# 256


