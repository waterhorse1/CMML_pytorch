import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedding(nn.Embedding): #used in MAML to forward input with fast weight 
    def __init__(self, num_embeddings, embedding_dim):
        super(Embedding, self).__init__(num_embeddings, embedding_dim)
        self.weight.fast = None #Lazy hack to add fast weight link

    def forward(self, x):
        if self.weight.fast is not None:
            out = F.embedding(x, self.weight.fast) #weight.fast (fast weight) is the temporaily adapted weight
        else:
            out = super(Embedding, self).forward(x)
        return out
    
class Linear_no_bias(nn.Linear): #used in MAML to forward input with fast weight 
    def __init__(self, in_features, out_features, bias):
        super(Linear_no_bias, self).__init__(in_features, out_features, bias)
        self.weight.fast = None #Lazy hack to add fast weight link
        #self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None:
            out = F.linear(x, self.weight.fast, None) #weight.fast (fast weight) is the temporaily adapted weight
        else:
            out = super(Linear_no_bias, self).forward(x)
        return out   
    
class item(torch.nn.Module):
    def __init__(self, config):
        super(item, self).__init__()
        self.num_rate = config.num_rate
        self.num_genre = config.num_genre
        self.num_director = config.num_director
        self.num_actor = config.num_actor
        self.embedding_dim = config.embedding_dim

        self.embedding_rate = Embedding(
            num_embeddings=self.num_rate, 
            embedding_dim=self.embedding_dim
        )
        
        self.embedding_genre = Linear_no_bias(
            in_features=self.num_genre,
            out_features=self.embedding_dim,
            bias=False
        )
        
        self.embedding_director = Linear_no_bias(
            in_features=self.num_director,
            out_features=self.embedding_dim,
            bias=False
        )
        
        self.embedding_actor = Linear_no_bias(
            in_features=self.num_actor,
            out_features=self.embedding_dim,
            bias=False
        )

    def forward(self, rate_idx, genre_idx, director_idx, actors_idx, vars=None):
        rate_emb = self.embedding_rate(rate_idx)
        genre_emb = self.embedding_genre(genre_idx.float()) / torch.sum(genre_idx.float(), 1).view(-1, 1)
        director_emb = self.embedding_director(director_idx.float()) / torch.sum(director_idx.float(), 1).view(-1, 1)
        actors_emb = self.embedding_actor(actors_idx.float()) / torch.sum(actors_idx.float(), 1).view(-1, 1)
        return torch.cat((rate_emb, genre_emb, director_emb, actors_emb), 1)


class user(torch.nn.Module):
    def __init__(self, config):
        super(user, self).__init__()
        self.num_gender = config.num_gender
        self.num_age = config.num_age
        self.num_occupation = config.num_occupation
        self.num_zipcode = config.num_zipcode
        self.embedding_dim = config.embedding_dim

        self.embedding_gender = Embedding(
            num_embeddings=self.num_gender,
            embedding_dim=self.embedding_dim
        )

        self.embedding_age = Embedding(
            num_embeddings=self.num_age,
            embedding_dim=self.embedding_dim
        )

        self.embedding_occupation = Embedding(
            num_embeddings=self.num_occupation,
            embedding_dim=self.embedding_dim
        )

        self.embedding_area = Embedding(
            num_embeddings=self.num_zipcode,
            embedding_dim=self.embedding_dim
        )

    def forward(self, gender_idx, age_idx, occupation_idx, area_idx):
        gender_emb = self.embedding_gender(gender_idx)
        age_emb = self.embedding_age(age_idx)
        occupation_emb = self.embedding_occupation(occupation_idx)
        area_emb = self.embedding_area(area_idx)
        return torch.cat((gender_emb, age_emb, occupation_emb, area_emb), 1)
