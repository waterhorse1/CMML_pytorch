import torch
import torch.nn as nn
import torch.nn.functional as F
import math, random
from utils import activation_method
from collections import defaultdict
import numpy as np

def create_module(module_type, **config):
    module_type = module_type.lower()
    if module_type == 'mlp':
        return MLP(**config)
    elif module_type == 'gcn':
        return AttentionGCN(**config)
    elif module_type == 'empty':
        return nn.Sequential()
    else:
        raise NotImplementedError


class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, final_size=0, final_activation="none", normalization="batch_norm",
                 activation='relu'):
        """
        :param input_size:
        :param hidden_layers: [(unit_num, normalization, dropout_rate)]
        :param final_size:
        :param final_activation:
        """
        nn.Module.__init__(self)
        self.input_size = input_size
        fcs = []
        last_size = self.input_size
        for size, to_norm, dropout_rate in hidden_layers:
            linear = nn.Linear(last_size, size)
            linear.bias.data.fill_(0.0)
            fcs.append(linear)
            last_size = size
            if to_norm:
                if normalization == 'batch_norm':
                    fcs.append(nn.BatchNorm1d(last_size))
                elif normalization == 'layer_norm':
                    fcs.append(nn.LayerNorm(last_size))
            fcs.append(activation_method(activation))
            if dropout_rate > 0.0:
                fcs.append(nn.Dropout(dropout_rate))
        self.fc = nn.Sequential(*fcs)
        if final_size > 0:
            linear = nn.Linear(last_size, final_size)
            linear.bias.data.fill_(0.0)
            finals = [linear, activation_method(final_activation)]
        else:
            finals = []
        self.final_layer = nn.Sequential(*finals)

    def forward(self, x):
        out = self.fc(x)
        out = self.final_layer(out)
        return out




class Recommender(nn.Module):
    def __init__(self, useritem_embeds, user_graph=False, item_graph=False):
        nn.Module.__init__(self)
        self.useritem_embeds = useritem_embeds
        self.user_graph = user_graph
        self.item_graph = item_graph
        #self.preprocess = nn.Linear(64,128)
    def forward(self, query_users, query_items, with_attr=False):
       #print("$$$$$$$$$$$$$$$$$$$$$$$$$$  enter")
        #print('query_users',query_users[0][0])
        #query_users[0] = torch.zeros(64)
        #query_users = [0,1,2,3]
        if query_users[0].dim() > 1:
            query_users = list(map(lambda x: x.squeeze(0), query_users))
        if query_items[0].dim() > 1:
            query_items = list(map(lambda x: x.squeeze(0), query_items))
        if not with_attr:
            query_users = self.useritem_embeds(*query_users, is_user=True, with_neighbor=self.user_graph)
            query_items = self.useritem_embeds(*query_items, is_user=False, with_neighbor=self.item_graph)
        #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        '''
        print(query_users[0].shape)
        cond = torch.tensor(query_users[0].shape[1] == 128).cuda(0)
        print(cond)
        query_users = torch.where(cond, query_users, self.preprocess(query_users[0]).cuda(0))
        query_items = torch.where(cond, query_items, self.preprocess(query_items[0]).cuda(0))
        '''
        return query_users, query_items


class InteractionRecommender(Recommender):
    def __init__(self, useritem_embeds, mlp_config):
        super(InteractionRecommender, self).__init__(useritem_embeds)
        self.mlp = MLP(**mlp_config)

    def forward(self, query_users, query_items, support_users=None, support_items=None, with_attr=False):
        query_users, query_items = super(InteractionRecommender, self).forward(query_users, query_items,
                                                                               with_attr=with_attr)
        query_users, query_items = query_users[0], query_items[0]
        if query_users.size(0) == 1:
            query_users = query_users.expand(query_items.size(0), -1)
        query_embeds = torch.cat((query_users, query_items), dim=1)
        return self.mlp(query_embeds).squeeze(1)


class Meta_final_InteractionRecommender(Recommender):
    def __init__(self, useritem_embeds):
        super(Meta_final_InteractionRecommender, self).__init__(useritem_embeds)
        self.mlp = nn.Sequential(nn.Linear(128, 128), nn.LeakyReLU(), nn.Linear(128, 128), nn.LeakyReLU())
        self.context_encoder = LSTM_encoder(useritem_embeds)
        self.hypernet = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                                      nn.Linear(128, 129))
        self.final_weight = None
        self.final_bias = None

    def forward(self, query_users, query_items, support_users=None, support_items=None, with_attr=False):
        query_users, query_items = super(Meta_final_InteractionRecommender, self).forward(query_users, query_items,
                                                                                          with_attr=with_attr)
        query_users, query_items = query_users[0], query_items[0]
        if query_users.size(0) == 1:
            query_users = query_users.expand(query_items.size(0), -1)
        query_embeds = torch.cat((query_users, query_items), dim=1)
        output = self.mlp(query_embeds)
        output = F.linear(output, weight=self.final_weight, bias=self.final_bias)

        return output.squeeze(1)

    def set_context(self, support_positive_users, support_positive_items, suppoer_negative_users, support_negative_items):
        context_embed = self.context_encoder(support_positive_users, support_positive_items, suppoer_negative_users, support_negative_items)  # 1 * 128
        self.final = self.hypernet(context_embed)
        self.final_weight = self.final[:, :128]
        self.final_bias = self.final[0, -1]
        
class Meta_final_cross_InteractionRecommender(Recommender):
    def __init__(self, useritem_embeds, way='dot'):
        super(Meta_final_cross_InteractionRecommender, self).__init__(useritem_embeds)
        self.mlp = nn.Sequential(nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, 128), nn.LeakyReLU())
        self.context_encoder = LSTM_encoder(useritem_embeds)#self_atn_mean(useritem_embeds)#LSTM_encoder(useritem_embeds)
        if way == 'dot':
            self.hypernet = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                                      nn.Linear(128, 129))
            print('dot')
        else:
            self.hypernet = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                                      nn.Linear(128, 129))
            print(way)
        self.way = way
        self.context_embed = None

    def forward(self, query_users, query_items, support_users=None, support_items=None, with_attr=False):
        query_users, query_items = super(Meta_final_cross_InteractionRecommender, self).forward(query_users, query_items,
                                                                                          with_attr=with_attr)
        query_users, query_items = query_users[0], query_items[0]
        if query_users.size(0) == 1:
            query_users = query_users.expand(query_items.size(0), -1)
        query_embeds = torch.cat((query_users, query_items), dim=1)#64 * 128?
        
        if not self.way == 'dot':
            embeds = torch.cat([query_embeds, self.context_embed.repeat(query_embeds.shape[0], 1)], dim=-1)# 64 * 128
        else:
            c_embed = self.context_embed.repeat(query_embeds.shape[0], 1)
            #c_embed = self.dropout(c_embed)
            embeds = query_embeds * c_embed#batch * 256 
            
        weight = self.hypernet(embeds)
        final_weight = weight[:, :128]# 64 * 128
        final_bias = weight[:, -1]
        
        output = self.mlp(query_embeds)# 64 * 128
             
        output = torch.sum(output * final_weight, dim=-1) + final_bias

        return output

    def set_context(self, support_positive_users, support_positive_items, suppoer_negative_users, support_negative_items):
        context_embed = self.context_encoder(support_positive_users, support_positive_items, suppoer_negative_users, support_negative_items)  # 1 * 128
        self.context_embed = context_embed# t * 128

class mlp_final_hypernet(nn.Module):
    def __init__(self):
        super(mlp_hypernet, self).__init__()
        self.f1 = nn.Linear(256, 128)
        self.f2 = nn.Linear(128, 128)
        self.f3 = nn.Linear(128, 128)
        
        self.f11 = nn.Linear(128, 256)
        self.f21 = nn.Linear(128, 256)
        
        self.relu = nn.ReLU()
    
    def forward(self, inp):
        x = self.relu(self.f1(inp))
        o1 = self.f11(x)
        x = self.relu(self.f2(x))
        o2 = self.f21(x)
        o3 = self.f3(x)
        return o1, o2, o3
    #64 * 128 64 * 128 64 * 1
    
class mlp_hypernet(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=64, out_dim=128):
        super(mlp_hypernet, self).__init__()
        self.f1 = nn.Linear(in_dim, hidden_dim)
        self.f2 = nn.Linear(hidden_dim, hidden_dim)
        self.f3 = nn.Linear(hidden_dim, hidden_dim)

        self.f11 = nn.Linear(hidden_dim, out_dim)
        self.f21 = nn.Linear(hidden_dim, out_dim)
        self.f31 = nn.Linear(hidden_dim, out_dim)
        
        self.relu = nn.ReLU()
        
    def forward(self, inp):
        x = self.relu(self.f1(inp))
        o1 = self.f11(x)
        x = self.relu(self.f2(x))
        o2 = self.f21(x)
        x = self.relu(self.f3(x))
        o3 = self.f31(x)
        return o1, o2,o3
    #64 * 128 64 * 128 64 * 1
    
class film_condition_mlp(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=64):
        super(film_condition_mlp, self).__init__()
        self.f1 = nn.Linear(in_dim, hidden_dim)
        self.f2 = nn.Linear(hidden_dim, hidden_dim)
        self.f3 = nn.Linear(hidden_dim, hidden_dim)
        
        self.l_relu = nn.LeakyReLU()
        self.hidden_dim= hidden_dim
        self.in_dim = in_dim
        
    def forward(self, x, o1, o2,o3):
        #print('enter@@@@@@@@@@@@@@@@@@')
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
        x = self.f3(x)
        weight = o3[:, :self.hidden_dim]
        bias = o3[:, self.hidden_dim:]
        x = x * weight + bias
        x = self.l_relu(x)        
        return x
    
class sigmoid_condition_mlp(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=128):
        super(sigmoid_condition_mlp, self).__init__()
        self.f1 = nn.Linear(in_dim, hidden_dim)
        self.f2 = nn.Linear(hidden_dim, hidden_dim)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        
        self.l_relu = nn.LeakyReLU()
        
    def forward(self, x, o1, o2):
        x = self.f1(x)
        weight = F.sigmoid(o1)
        x = x * weight
        x = self.l_relu(x)
        x = self.f2(x)
        weight = F.sigmoid(o2)
        x = x * weight
        x = self.l_relu(x)
        return x

class soft_module_route(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=32, num=4):
        super(soft_module_route, self).__init__()
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3= nn.Linear(64, 32)
        self.fc12 = nn.Linear(64, num*num)
        self.fc22 = nn.Linear(64, num*num)
        self.fc32 = nn.Linear(32, num)
        self.num = num
    def forward(self, x):
        x = F.relu(self.fc1(x))
        o1 = self.fc12(x)
        o1 = F.softmax(o1.reshape(x.shape[0],self.num,self.num),dim=1)
        x = F.relu(self.fc2(x))
        o2 = self.fc22(x)
        o2 = F.softmax(o2.reshape(x.shape[0],self.num,self.num),dim=1)     #(bs,4,4)
        x = F.relu(self.fc3(x))
        o3 = self.fc32(x)
        o3 = F.softmax(o3.reshape(x.shape[0],self.num,1),dim=1)       #(bs,4,4)        
        return o1, o2,o3


class soft_module(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=32, num=4):
        super(soft_module, self).__init__()
        self.fc1 = nn.ModuleList([nn.Linear(in_dim, hidden_dim) for _ in range(num)])
        self.fc2 = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num)])
        self.fc3 = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num)])
        self.final_layer = nn.Linear(hidden_dim, 1)
        self.num = num
    def forward(self, x, c1, c2,c3):    #x:(714,256),c1:(714,4,4) c2: (714,4,1) 
        #print('c1',c1.shape,'c2',c2.shape)
        o1 = []
        for i in range(len(self.fc1)):
            o1.append(F.relu(self.fc1[i](x)))#[batch * 64 * 4] batch * 4 * 4
        o1 = torch.stack(o1, dim=-1)   #(bs,64,4)
        #print('o1',o1.shape)
        o2 = []
        in2 = o1 @ c1#b*64*4   #(bs,64,4)
        #print('in2',in2.shape)
        for i in range(self.num):
            o2.append(F.relu(self.fc2[i](in2[:,:,i])))
        o2 = torch.stack(o2, dim=-1)   #(bs,64,4)
        #print('o2',o2.shape)
        in3 = (o2 @ c2).squeeze(dim=-1)#b*64*1   #(bs,64,4)
        #print('in3',in3.shape)
        o3=[]
        for i in range(self.num):
            o3.append(F.relu(self.fc3[i](in3[:,:,i])))
        o3 = torch.stack(o3, dim=-1)   #(bs,64,4)
        #print('o2',o2.shape)
        in4= (o3 @ c3).squeeze(dim=-1)#b*64*1   #(bs,64,4)
        out = self.final_layer(in4)
        return out

          
class Meta_mlp_final_cross_InteractionRecommender(Recommender):
    def __init__(self, useritem_embeds, way='dot'):
        super(Meta_mlp_cross_InteractionRecommender, self).__init__(useritem_embeds)
        self.mlp = film_condition_mlp()#nn.Sequential(nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, 128), nn.LeakyReLU())
        self.context_encoder = LSTM_encoder(useritem_embeds)#self_atn_mean(useritem_embeds)#LSTM_encoder(useritem_embeds)
        if way == 'dot':
            self.hypernet = mlp_final_hypernet(in_dim=256, hidden_dim=128, out_dim=256)
            print(way)
        self.way = way
        #self.final_weight = None
        #self.final_bias = None
        self.context_embed = None
        #self.dropout = nn.Dropout(0.3)

    def forward(self, query_users, query_items, support_users=None, support_items=None, with_attr=False):
        query_users, query_items = super(Meta_mlp_cross_InteractionRecommender, self).forward(query_users, query_items,
                                                                                          with_attr=with_attr)
        query_users, query_items = query_users[0], query_items[0]
        if query_users.size(0) == 1:
            query_users = query_users.expand(query_items.size(0), -1)
        query_embeds = torch.cat((query_users, query_items), dim=1)#64 * 128?
        
        #context dropout
        if not self.way == 'dot':
            embeds = torch.cat([query_embeds, self.context_embed.repeat(query_embeds.shape[0], 1)], dim=-1)# 64 * 128
        else:
            c_embed = self.context_embed.repeat(query_embeds.shape[0], 1)
            #c_embed = self.dropout(c_embed)
            embeds = query_embeds * c_embed#batch * 256 
        o1, o2, o3 = self.hypernet(embeds)
        final_weight = o3[:, :128]# 64 * 128
        final_bias = o3[:, -1]
        
        output = self.mlp(query_embeds, o1, o2)# 64 * 128
             
        output = torch.sum(output * final_weight, dim=-1) + final_bias

        return output

    def set_context(self, support_positive_users, support_positive_items, suppoer_negative_users, support_negative_items):
        context_embed = self.context_encoder(support_positive_users, support_positive_items, suppoer_negative_users, support_negative_items)  # 1 * 128
        self.context_embed = context_embed# t * 128

class Meta_softmodule_cross_InteractionRecommender(Recommender):
    def __init__(self, useritem_embeds, way='dot'):
        super(Meta_softmodule_cross_InteractionRecommender, self).__init__(useritem_embeds)
        print('softmodule')
        self.mlp = soft_module()
        self.context_encoder = LSTM_encoder(useritem_embeds)#self_atn_mean(useritem_embeds)#LSTM_encoder(useritem_embeds)
        if way == 'dot':
            self.hypernet = soft_module_route()
            print(way)
        self.way = way
        self.context_embed = None
        #self.preprocess = nn.Linear(128,256)

    def forward(self, query_users, query_items, support_users=None, support_items=None, with_attr=False):
        #print("query_users",query_users.shape,"query_items",query_items.shape)
        query_users, query_items = super(Meta_softmodule_cross_InteractionRecommender, self).forward(query_users, query_items,
                                                                                          with_attr=with_attr)
        query_users, query_items = query_users[0], query_items[0]   #(714,128)
        if query_users.size(0) == 1:
            query_users = query_users.expand(query_items.size(0), -1)
        query_embeds = torch.cat((query_users, query_items), dim=1)#64 * 128?   #
        if not self.way == 'dot':
            embeds = torch.cat([query_embeds, self.context_embed.repeat(query_embeds.shape[0], 1)], dim=-1)# 64 * 128
        else:
            c_embed = self.context_embed.repeat(query_embeds.shape[0], 1)    #(714,256 )   
            embeds = query_embeds * c_embed#batch * 256   (714,256)
        
        o1, o2, o3= self.hypernet(embeds)
        output = self.mlp(query_embeds, o1, o2,o3)# 64 * 128

        return output.squeeze(dim=-1)

    def set_context(self, support_positive_users, support_positive_items, suppoer_negative_users, support_negative_items):
        context_embed = self.context_encoder(support_positive_users, support_positive_items, suppoer_negative_users, support_negative_items)  # 1 * 128
        
        self.context_embed = context_embed# t * 128
        
class Meta_mlp_cross_InteractionRecommender(Recommender):
    def __init__(self, useritem_embeds, way='dot', condition_way='film'):
        super(Meta_mlp_cross_InteractionRecommender, self).__init__(useritem_embeds)
        print(condition_way)
        if condition_way == 'film':
            self.mlp = film_condition_mlp()
            hyper_dim = 128
        elif condition_way == 'sigmoid':
            self.mlp = sigmoid_condition_mlp()
            hyper_dim = 128
        else:
            self.mlp = group_film_condition_mlp()
            hyper_dim = 32
        self.context_encoder = LSTM_encoder(useritem_embeds)#self_atn_mean(useritem_embeds)#LSTM_encoder(useritem_embeds)
        if way == 'dot':
            self.hypernet = mlp_hypernet(out_dim=hyper_dim)
            print(way)
        self.way = way
        
        self.context_embed = None
        self.final_layer = nn.Linear(64, 1)

    def forward(self, query_users, query_items, support_users=None, support_items=None, with_attr=False):
        query_users, query_items = super(Meta_mlp_cross_InteractionRecommender, self).forward(query_users, query_items,
                                                                                          with_attr=with_attr)
        query_users, query_items = query_users[0], query_items[0]
        if query_users.size(0) == 1:
            query_users = query_users.expand(query_items.size(0), -1)
        query_embeds = torch.cat((query_users, query_items), dim=1)#64 * 128?
        
        #context dropout
        if not self.way == 'dot':
            embeds = torch.cat([query_embeds, self.context_embed.repeat(query_embeds.shape[0], 1)], dim=-1)# 64 * 128
        else:
            c_embed = self.context_embed.repeat(query_embeds.shape[0], 1)
            #c_embed = self.dropout(c_embed)
            embeds = query_embeds * c_embed#batch * 256 
        o1, o2, o3 = self.hypernet(embeds)
        
        output = self.mlp(query_embeds, o1, o2, o3)# 64 * 128
             
        output = self.final_layer(output).squeeze(dim=-1)

        return output

    def set_context(self, support_positive_users, support_positive_items, suppoer_negative_users, support_negative_items):
        context_embed = self.context_encoder(support_positive_users, support_positive_items, suppoer_negative_users, support_negative_items)  # 1 * 128
        self.context_embed = context_embed# t * 128
        

class Mean_encoder(Recommender):
    def __init__(self, useritem_embeds):
        super(Mean_encoder, self).__init__(useritem_embeds)
        self.mlp = nn.Sequential(nn.Linear(258, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 256))

    def forward(self, users, items, n_users, n_items, with_attr=False):
        users, items = super(Mean_encoder, self).forward(users, items, with_attr=with_attr)
        users, items = users[0], items[0]
        if users.size(0) == 1:
            users = users.expand(items.size(0), -1)
        embeds = torch.cat((users, items), dim=1)  # bs * 128
        label_tensor = torch.zeros([embeds.shape[0], 2]).cuda(0)
        label_tensor[:,0] = 1.
        embeds = torch.cat([embeds, label_tensor], dim=-1)#bs * 130

        n_users, n_items = super(Mean_encoder, self).forward(n_users, n_items, with_attr=with_attr)
        n_users, n_items = n_users[0], n_items[0]
        if n_users.size(0) == 1:
            n_users = n_users.expand(n_items.size(0), -1)
        n_embeds = torch.cat((n_users, n_items), dim=1)  # bs * 128
        label_tensor = torch.zeros([n_embeds.shape[0], 2]).cuda(0)
        label_tensor[:, 1] = 1.
        n_embeds = torch.cat([n_embeds, label_tensor], dim=-1)  # bs * 130

        embeds = torch.cat([embeds, n_embeds], dim=0)

        meta_embeds = self.mlp(embeds).mean(dim=0, keepdim=True)
        return meta_embeds  #1 * 128


class LSTM_encoder(Recommender):
    def __init__(self, useritem_embeds):
        super(LSTM_encoder, self).__init__(useritem_embeds)
        input_dim=256
        mlp_input_dim = 128
        self.mlp = nn.Sequential(nn.Linear(mlp_input_dim, 64), nn.ReLU(), nn.Linear(64, input_dim))
        self.gru = nn.GRU(input_size=input_dim +2, hidden_size=mlp_input_dim)

    def forward(self, users, items, n_users, n_items, with_attr=False):
        users, items = super(LSTM_encoder, self).forward(users, items, with_attr=with_attr)

        users, items = users[0], items[0]# task * bs * 128
        #print('users',users.shape,'items',items.shape)
        if len(users.shape) == 2:
            users = users[None]
            items = items[None]
        embeds = torch.cat((users, items), dim=-1)  # t * bs * 128
        label_tensor = torch.zeros([embeds.shape[0], embeds.shape[1], 2]).cuda(0)
        label_tensor[:, :, 0] = 1.
        embeds = torch.cat([embeds, label_tensor], dim=-1)  # t * bs * 130
        #print("embeds",embeds.shape)
        n_users, n_items = super(LSTM_encoder, self).forward(n_users, n_items, with_attr=with_attr)
        n_users, n_items = n_users[0], n_items[0]
        if len(n_users.shape) == 2:
            n_users = n_users[None]
            n_items = n_items[None]
        n_embeds = torch.cat((n_users, n_items), dim=-1)  # t * bs * 128
        label_tensor = torch.zeros([n_embeds.shape[0], n_embeds.shape[1], 2]).cuda(0)
        label_tensor[:, :, 1] = 1.
        n_embeds = torch.cat([n_embeds, label_tensor], dim=-1)  # t * bs * 130

        embeds = torch.cat([embeds, n_embeds], dim=1)
        
        #index = np.array(range(len(embeds)))
        #random.shuffle(index)
        #embeds = embeds[index]
        #print('embeds',embeds.shape)
        #cond = (embeds.shape[1]==256).cuda(0)
        output, h_n = self.gru(embeds.transpose(1,0))

        meta_embeds = self.mlp(output[-1])
        return meta_embeds  # t * 128



class EmbedRecommender(Recommender):
    def __init__(self, useritem_embeds, user_config, item_config, user_graph=True, item_graph=True):
        super(EmbedRecommender, self).__init__(useritem_embeds, user_graph, item_graph)
        self.user_model = create_module(**user_config)
        self.item_model = create_module(**item_config)

    def forward(self, query_users, query_items, with_attr=False):
        """
        :param with_attr:
        :param query_users: (batch_size,)
        :param query_items: (batch_size)
        :return:
        """
        query_users, query_items = Recommender.forward(self, query_users, query_items, with_attr=with_attr)
        query_users = self.user_model(*query_users)
        query_items = self.item_model(*query_items)
        return (query_users * query_items).sum(dim=1)


class CoNet(nn.Module):
    def __init__(self, useritem_embeds, source_ratings, item_padding_idx, input_size, hidden_layers):
        nn.Module.__init__(self)
        self.useritem_embeds = useritem_embeds
        self.source_ratings = source_ratings
        self.item_padding_idx = item_padding_idx
        last_size = input_size * 2
        layers1, layers2, transfer_layers = [], [], []
        for hidden_size in hidden_layers:
            layers1.append(nn.Linear(last_size, hidden_size))
            layers2.append(nn.Linear(last_size, hidden_size))
            transfer_layers.append(nn.Linear(last_size, hidden_size))
            last_size = hidden_size
        self.target_layers = nn.ModuleList(layers1)
        self.auxiliary_layers = nn.ModuleList(layers2)
        self.transfer_layers = nn.ModuleList(transfer_layers)
        self.target_output = nn.Linear(last_size, 1)
        self.auxiliary_output = nn.Linear(last_size, 1)

    def forward(self, query_users, target_items, auxiliary_items=None):
        only_target = False
        if auxiliary_items is None:
            only_target = True
            auxiliary_items = [
                random.choice(self.source_ratings[user_id.item()]) if len(
                    self.source_ratings[user_id.item()]) > 0 else self.item_padding_idx for user_id in query_users[0]]
            auxiliary_items = (torch.tensor(auxiliary_items, dtype=torch.long, device=query_users[0].device),)
        query_users = list(map(lambda x: x.expand(target_items[0].size(0)), query_users))
        auxiliary_items = list(map(lambda x: x.expand(target_items[0].size(0)), auxiliary_items))
        query_users = self.useritem_embeds(*query_users, is_user=True)
        target_items, auxiliary_items = self.useritem_embeds(*target_items, is_user=False), self.useritem_embeds(
            *auxiliary_items, is_user=False)
        target_x = torch.cat((*query_users, *target_items), dim=1)
        auxiliary_x = torch.cat((*query_users, *auxiliary_items), dim=1)
        for target_layer, auxiliary_layer, transfer_layer in zip(self.target_layers, self.auxiliary_layers,
                                                                 self.transfer_layers):
            new_target_x = target_layer(target_x) + transfer_layer(auxiliary_x)
            new_auxiliary_x = auxiliary_layer(auxiliary_x) + transfer_layer(target_x)
            target_x, auxiliary_x = new_target_x, new_auxiliary_x
            target_x, auxiliary_x = torch.relu_(target_x), torch.relu_(auxiliary_x)
        if only_target:
            return self.target_output(target_x).squeeze(-1)
        else:
            return self.target_output(target_x).squeeze(-1), self.auxiliary_output(auxiliary_x).squeeze(-1)


class HybridRecommender(Recommender):
    def __init__(self, useritem_embeds, input_size, hidden_layers, final_size, activation='relu',
                 normalization="batch_norm"):
        super(HybridRecommender, self).__init__(useritem_embeds, False, False)
        self.interaction_model = MLP(input_size=2 * input_size, hidden_layers=hidden_layers, activation=activation,
                                     normalization=normalization, final_activation='none', final_size=final_size)
        self.final_layer = nn.Linear(input_size + final_size, 1)

    def forward(self, query_users, query_items, with_attr=False):
        query_users, query_items = Recommender.forward(self, query_users, query_items, with_attr=with_attr)
        query_users, query_items = query_users[0], query_items[0]
        if query_users.size(0) == 1:
            query_users = query_users.expand(query_items.size(0), -1)
        interactions = torch.cat((query_users, query_items), dim=-1)
        interactions = self.interaction_model(interactions)
        product = query_users * query_items
        concatenation = torch.cat((interactions, product), dim=-1)
        return self.final_layer(concatenation).squeeze(-1)

        
        