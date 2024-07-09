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

    def forward(self, query_users, query_items, with_attr=False, pretrain=False):
        if query_users[0].dim() > 1:
            query_users = list(map(lambda x: x.squeeze(0), query_users))
        if query_items[0].dim() > 1:
            query_items = list(map(lambda x: x.squeeze(0), query_items))
        if pretrain:
            query_users = self.useritem_embeds(query_users, is_user=True, with_neighbor=self.user_graph)
            query_items = self.useritem_embeds(query_items, is_user=False, with_neighbor=self.item_graph)
        else:
            if not with_attr:
                query_users = self.useritem_embeds(*query_users, is_user=True, with_neighbor=self.user_graph)
                query_items = self.useritem_embeds(*query_items, is_user=False, with_neighbor=self.item_graph)
    
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
    def __init__(self, useritem_embeds, input_shape=128, hyper_hidden=64):
        super(Meta_final_InteractionRecommender, self).__init__(useritem_embeds)
        self.mlp = nn.Sequential(nn.Linear(input_shape, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU())
        self.context_encoder = LSTM_encoder(useritem_embeds, in_dim=input_shape)
        self.hypernet = nn.Sequential(nn.Linear(input_shape, hyper_hidden), nn.ReLU(), nn.Linear(hyper_hidden, hyper_hidden), nn.ReLU(),
                                      nn.Linear(hyper_hidden, 65))
        self.final_weight = None
        self.final_bias = None
        print('final')

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
        self.final_weight = self.final[:, :64]
        self.final_bias = self.final[0, -1]

class Meta_final_cross_InteractionRecommender(Recommender):
    def __init__(self, useritem_embeds, way='dot', input_shape=128, hyper_hidden=64, lstm_hidden=128, bias_init=False, lstm_bias=False, bias_value=0.0):
        super(Meta_final_cross_InteractionRecommender, self).__init__(useritem_embeds)
        self.mlp = nn.Sequential(nn.Linear(input_shape, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU())
        if lstm_hidden == 0:
            self.context_encoder  = Mean_encoder(useritem_embeds, in_dim=input_shape)
        else:
            self.context_encoder  = LSTM_encoder(useritem_embeds, in_dim=input_shape, hidden_dim=lstm_hidden, lstm_bias=lstm_bias, bias_value=bias_value)
        self.way = way
        print(way)
        if way == 'dot':
            self.hypernet = nn.Sequential(nn.Linear(input_shape, hyper_hidden), nn.ReLU(), nn.Linear(hyper_hidden, hyper_hidden), nn.ReLU(),
                                      nn.Linear(hyper_hidden, 65))
        elif self.way == 'mlp':
            self.hypernet = nn.Sequential(nn.Linear(input_shape*2, hyper_hidden), nn.ReLU(), nn.Linear(hyper_hidden, hyper_hidden), nn.ReLU(),
                                      nn.Linear(hyper_hidden, 65))
        elif self.way == 'no':
            self.hypernet = nn.Sequential(nn.Linear(input_shape, hyper_hidden), nn.ReLU(), nn.Linear(hyper_hidden, hyper_hidden), nn.ReLU(),
                                      nn.Linear(hyper_hidden, 65))
        else:
            raise NotImplementedError
        if bias_init:
            self.mlp[0].bias.data.fill_(bias_value)
            self.mlp[2].bias.data.fill_(bias_value)
            self.mlp[4].bias.data.fill_(bias_value)
            self.hypernet[0].bias.data.fill_(bias_value)
            self.hypernet[2].bias.data.fill_(bias_value)
            self.hypernet[4].bias.data.fill_(bias_value)
        self.context_embed = None

    def forward(self, query_users, query_items, support_users=None, support_items=None, with_attr=False):
        query_users, query_items = super(Meta_final_cross_InteractionRecommender, self).forward(query_users, query_items,
                                                                                          with_attr=with_attr)
        query_users, query_items = query_users[0], query_items[0]
        if query_users.size(0) == 1:
            query_users = query_users.expand(query_items.size(0), -1)
        query_embeds = torch.cat((query_users, query_items), dim=1)#64 * 128?
        #print(query_embeds.shape)
        
        if self.way == 'mlp':
            embeds = torch.cat([query_embeds, self.context_embed.repeat(query_embeds.shape[0], 1)], dim=-1)# 64 * 128
        elif self.way == 'dot':
            c_embed = self.context_embed.repeat(query_embeds.shape[0], 1)
            embeds = query_embeds * c_embed#batch * 256 
        elif self.way == 'no':
            c_embed = self.context_embed.repeat(query_embeds.shape[0], 1)
            embeds = c_embed          
        else:
            raise NotImplementedError
            
        weight = self.hypernet(embeds)
        final_weight = weight[:, :64]# 64 * 128
        final_bias = weight[:, -1]
        
        output = self.mlp(query_embeds)# 64 * 128
             
        output = torch.sum(output * final_weight, dim=-1) + final_bias

        return output

    def set_context(self, support_positive_users, support_positive_items, suppoer_negative_users, support_negative_items):
        context_embed = self.context_encoder(support_positive_users, support_positive_items, suppoer_negative_users, support_negative_items)  # 1 * 128
        self.context_embed = context_embed# t * 128

class mlp_hypernet(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=128, out_dim=64, bias_init=False, bias_value=0.0):
        super(mlp_hypernet, self).__init__()
        self.f1 = nn.Linear(in_dim, hidden_dim)
        self.f2 = nn.Linear(hidden_dim, hidden_dim)
        self.f3 = nn.Linear(hidden_dim, hidden_dim)
        
        if bias_init:
            self.f1.bias.data.fill_(bias_value)  
            self.f2.bias.data.fill_(bias_value)
            self.f3.bias.data.fill_(bias_value)
      
        self.f11 =  nn.Linear(hidden_dim, out_dim)
        self.f21 =  nn.Linear(hidden_dim, out_dim)
        self.f31 =  nn.Linear(hidden_dim, out_dim)
        
        if bias_init:     
            self.f11.bias.data.fill_(bias_value)     
            self.f21.bias.data.fill_(bias_value) 
            self.f31.bias.data.fill_(bias_value)
       
        self.relu = nn.ReLU()
        
    def forward(self, inp):
        x = self.relu(self.f1(inp))
        o1 = self.f11(x)
        x = self.relu(self.f2(x))
        o2 = self.f21(x)
        x = self.relu(self.f3(x))
        o3 = self.f31(x)
        return o1, o2, o3
    #64 * 128 64 * 128 64 * 1
    
class film_condition_mlp(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=64, bias_init=False, bias_value=0.0):
        super(film_condition_mlp, self).__init__()
        self.f1 = nn.Linear(in_dim, hidden_dim)
        self.f2 = nn.Linear(hidden_dim, hidden_dim)
        self.f3 = nn.Linear(hidden_dim, hidden_dim)
        
        if bias_init:
            self.f1.bias.data.fill_(bias_value)
            self.f2.bias.data.fill_(bias_value) 
            self.f3.bias.data.fill_(bias_value)
        
        self.l_relu = nn.LeakyReLU()
        self.hidden_dim= hidden_dim
        self.in_dim = in_dim
        
    def forward(self, x, o1, o2, o3):
        
        x = self.f1(x)
        weight = o1[:, :self.hidden_dim]
        bias = o1[:, self.hidden_dim:]
        #x = self.l_relu(x)
        x = x * weight + bias
        x = self.l_relu(x)
        x = self.f2(x)
        #x = self.l_relu(x)
        weight = o2[:, :self.hidden_dim]
        bias = o2[:, self.hidden_dim:]
        x = x * weight + bias
        x = self.l_relu(x)
        x = self.f3(x)
        #x = self.l_relu(x)
        weight = o3[:, :self.hidden_dim]
        bias = o3[:, self.hidden_dim:]
        x = x * weight + bias
        x = self.l_relu(x)
        
        return x
    
class sigmoid_condition_mlp(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=64, bias_init=False, bias_value=0.0):
        super(sigmoid_condition_mlp, self).__init__()
        self.f1 = nn.Linear(in_dim, hidden_dim)
        self.f2 = nn.Linear(hidden_dim, hidden_dim)
        self.f3 = nn.Linear(hidden_dim, hidden_dim)
        
        if bias_init:     
            self.f1.bias.data.fill_(bias_value)
            self.f2.bias.data.fill_(bias_value)
            self.f3.bias.data.fill_(bias_value)
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        
        self.l_relu = nn.LeakyReLU()
        
    def forward(self, x, o1, o2, o3):
        x = self.f1(x)
        weight = F.sigmoid(o1)
        x = x * weight
        x = self.l_relu(x)
        x = self.f2(x)
        weight = F.sigmoid(o2)
        x = x * weight
        x = self.l_relu(x)
        x = self.f3(x)
        weight = F.sigmoid(o3)
        x = x * weight
        x = self.l_relu(x)
        
        return x
    
class soft_module_route(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=32, num=4):
        super(soft_module_route, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        self.fc12 = nn.Linear(hidden_dim, 16)
        self.fc22 = nn.Linear(hidden_dim, 16)
        self.fc32 = nn.Linear(hidden_dim, 4)

        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        o1 = self.fc12(x)
        o1 = F.softmax(o1.reshape(x.shape[0], 4, 4), dim=1)#batch*4*4
        x = F.relu(self.fc2(x))
        o2 = self.fc22(x)
        o2 = F.softmax(o2.reshape(x.shape[0], 4, 4), dim=1)
        x = F.relu(self.fc3(x))
        o3 = self.fc32(x)
        o3 = F.softmax(o3.reshape(x.shape[0], 4, 1), dim=1)

        return o1, o2, o3
    
class soft_module(nn.Module):
    def __init__(self, in_dim=256, first_dim=32, second_dim=32, out_dim=64, num=4):
        super(soft_module, self).__init__()
        self.fc1 = nn.ModuleList([nn.Linear(in_dim, first_dim) for _ in range(num)])
        self.fc2 = nn.ModuleList([nn.Linear(second_dim, hidden_dim) for _ in range(num)])
        self.fc3 = nn.ModuleList([nn.Linear(hidden_dim, out_dim) for _ in range(num)])

        self.final_layer = nn.Linear(out_dim, 1)
        self.num = num
    def forward(self, x, c1, c2, c3):
        o1 = []
        for i in range(len(self.fc1)):
            o1.append(F.relu(self.fc1[i](x)))
        o1 = torch.stack(o1, dim=-1)
        o2 = []
        in2 = o1 @ c1
        for i in range(len(self.fc2)):
            o2.append(F.relu(self.fc2[i](in2[:,:,i])))
        o2 = torch.stack(o2, dim=-1)
        o3 = []
        in3 = (o2 @ c2)
        for i in range(len(self.fc3)):
            o3.append(F.relu(self.fc3[i](in3[:,:,i])))
        o3 = torch.stack(o3, dim=-1)
        in4 = (o3 @ c3).squeeze(dim=-1)
        out = self.final_layer(in4)
        return out

class Meta_softmodule_cross_InteractionRecommender(Recommender):
    def __init__(self, useritem_embeds, way='dot', input_shape=128, hyper_hidden=64, lstm_hidden=128):
        super(Meta_softmodule_cross_InteractionRecommender, self).__init__(useritem_embeds)
        print('softmodule')
        self.mlp = soft_module(in_dim=input_shape)
        self.context_encoder = LSTM_encoder(useritem_embeds, in_dim=input_shape, hidden_dim=lstm_hidden)
        if way == 'dot':
            self.hypernet = soft_module_route(in_dim=input_shape, hidden_dim=hyper_hidden)
            print(way)
        else:
            raise NotImplementedError
            #self.hypernet = nn.Sequential(nn.Linear(input_shape*2, hyper_hidden), nn.ReLU(), nn.Linear(hyper_hidden, hyper_hidden), nn.ReLU(),
            #                          nn.Linear(hyper_hidden, 65))
        self.way = way
        self.context_embed = None

    def forward(self, query_users, query_items, support_users=None, support_items=None, with_attr=False):
        query_users, query_items = super(Meta_softmodule_cross_InteractionRecommender, self).forward(query_users, query_items,
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
        #print(embeds.shape)
        o1, o2, o3 = self.hypernet(embeds)
        
        output = self.mlp(query_embeds, o1, o2, o3)# 64 * 128

        #return output.squeeze(dim=-1), o1, o2, o3
        return output.squeeze(dim=-1)

    def set_context(self, support_positive_users, support_positive_items, suppoer_negative_users, support_negative_items):
        context_embed = self.context_encoder(support_positive_users, support_positive_items, suppoer_negative_users, support_negative_items)  # 1 * 128
        self.context_embed = context_embed# t * 128
        
class Meta_mlp_cross_InteractionRecommender(Recommender):
    def __init__(self, useritem_embeds, way='dot', condition_way='film', input_shape=128, hyper_hidden=64, lstm_hidden=128, bias_init=False, lstm_bias=False, bias_value = 0.0):
        super(Meta_mlp_cross_InteractionRecommender, self).__init__(useritem_embeds)
        print(condition_way)
        if condition_way == 'film':
            self.mlp = film_condition_mlp(in_dim=input_shape, bias_init=bias_init, bias_value=bias_value)
            hyper_dim = 128
        elif condition_way == 'sigmoid':
            self.mlp = sigmoid_condition_mlp(in_dim=input_shape, bias_init=bias_init, bias_value=bias_value)
            hyper_dim = 64
        else:
            raise NotImplementedError
        self.context_encoder = LSTM_encoder(useritem_embeds, in_dim=input_shape, hidden_dim=lstm_hidden, lstm_bias=lstm_bias, bias_value=bias_value)
        if way == 'dot':
            self.hypernet = mlp_hypernet(in_dim=input_shape, hidden_dim=hyper_hidden, out_dim=hyper_dim, bias_init=bias_init, bias_value=bias_value)
        else:
            raise NotImplementedError
        #else
        self.way = way
        #self.final_weight = None
        #self.final_bias = None
        self.context_embed = None
        self.final_layer = nn.Linear(64, 1)
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
        
        output = self.mlp(query_embeds, o1, o2, o3)# 64 * 128
             
        output = self.final_layer(output).squeeze(dim=-1)

        return output

    def set_context(self, support_positive_users, support_positive_items, suppoer_negative_users, support_negative_items):
        context_embed = self.context_encoder(support_positive_users, support_positive_items, suppoer_negative_users, support_negative_items)  # 1 * 128
        self.context_embed = context_embed# t * 128

    
class Mean_encoder(Recommender):
    def __init__(self, useritem_embeds, in_dim, hidden_dim=128):
        super(Mean_encoder, self).__init__(useritem_embeds)
        self.mlp = nn.Sequential(nn.Linear(in_dim+2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, in_dim))

    def forward(self, users, items, n_users, n_items, with_attr=False):
        users, items = super(Mean_encoder, self).forward(users, items, with_attr=with_attr)
        users, items = users[0], items[0]
        if users.size(0) == 1:
            users = users.expand(items.size(0), -1)
        embeds = torch.cat((users, items), dim=1)  # bs * 128
        label_tensor = torch.zeros([embeds.shape[0], 2]).cuda(0)
        label_tensor[:,0] = 1.
        embeds = torch.cat([embeds, label_tensor], dim=-1)#bs * 130
        '''
        n_users, n_items = super(Mean_encoder, self).forward(n_users, n_items, with_attr=with_attr)
        n_users, n_items = n_users[0], n_items[0]
        if n_users.size(0) == 1:
            n_users = n_users.expand(n_items.size(0), -1)
        n_embeds = torch.cat((n_users, n_items), dim=1)  # bs * 128
        label_tensor = torch.zeros([n_embeds.shape[0], 2]).cuda(0)
        label_tensor[:, 1] = 1.
        n_embeds = torch.cat([n_embeds, label_tensor], dim=-1)  # bs * 130

        embeds = torch.cat([embeds, n_embeds], dim=0)
        '''
        embeds = embeds
        meta_embeds = self.mlp(embeds).mean(dim=0, keepdim=True)
        return meta_embeds  #1 * 128

class LSTM_encoder(Recommender):
    def __init__(self, useritem_embeds, in_dim, hidden_dim, lstm_bias=False, bias_value=0.0):
        super(LSTM_encoder, self).__init__(useritem_embeds)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, in_dim))
        self.gru = nn.GRU(input_size=in_dim+2, hidden_size=hidden_dim)
        if lstm_bias:
            self.mlp[0].bias.data.fill_(bias_value)
            self.mlp[2].bias.data.fill_(bias_value)
                                    
    def forward(self, users, items, n_users, n_items, with_attr=False):
        users, items = super(LSTM_encoder, self).forward(users, items, with_attr=with_attr)

        users, items = users[0], items[0]# task * bs * 128
        if len(users.shape) == 2:
            users = users[None]
            items = items[None]
        embeds = torch.cat((users, items), dim=-1)  # t * bs * 128
        label_tensor = torch.zeros([embeds.shape[0], embeds.shape[1], 2]).cuda(0)
        label_tensor[:, :, 0] = 1.
        embeds = torch.cat([embeds, label_tensor], dim=-1)  # t * bs * 130
        
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
        #print(target_x.shape)
        #print(auxiliary_x.shape)
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

class NCE(nn.Module):
    def __init__(self, dim):
        super(NCE, self).__init__()
        self.dim = dim
        self.W = nn.Parameter(0.1 * torch.randn(self.dim, self.dim))
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, x1, x2):#bs * dim
        assert x1.shape[-1] == self.dim and x2.shape[-1] == self.dim
        bs = x1.shape[0]
        mat = x1 @ self.W @ x2.t() # bs * bs
        label = torch.tensor(range(bs)).long().cuda(0)
        c_loss = self.loss(mat, label)
        return c_loss
    
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
    
class DCN(Recommender):
    def __init__(self, useritem_embeds):
        super(DCN, self).__init__(useritem_embeds)
        #self.l = nn.Sequential(nn.Linear(256, 64),nn.)
        self.cross = CrossNet(256, 3)
        
        self.deep = DeepNet(256, [64,64])
        self.out = nn.Linear(256 + 64, 1)
    def forward(self, user, item):
        query_users, query_items = super(DCN, self).forward(user, item)
        query_users, query_items = query_users[0], query_items[0]
        #print(query_users.shape, query_items.shape)
        if query_users.size(0) == 1:
            query_users = query_users.expand(query_items.size(0), -1)
        o = torch.cat([query_users, query_items], dim=-1)
        cross = self.cross(o)
        deep = self.deep(o)
        x = torch.cat([deep, cross], dim=-1)
        x = self.out(x)
        return x.squeeze(dim=-1)
