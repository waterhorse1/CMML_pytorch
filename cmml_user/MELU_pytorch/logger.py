import time

import numpy as np


class Logger:

    def __init__(self):
        self.args = None
        self.train_loss = []
        self.train_conf = []

        self.train_ndcg1 = []
        self.train_ndcg1_conf = []

        self.train_ndcg3 = []
        self.train_ndcg3_conf = []
        
        self.test_loss = [[],[]]
        self.test_conf = [[],[]]
        
        self.test_ndcg1 = [[],[]]
        self.test_ndcg1_conf = [[],[]]
        
        self.test_ndcg3 = [[],[]]
        self.test_ndcg3_conf = [[],[]]

        self.best_valid_model = None
        self.best_encoder_valid_model = None
        self.best_gradient_place_valid_model = None
        self.best_embedding_place_valid_model = None
        self.best_decoder_model = None
        self.best_g_valid_model = None
        
        self.valid_model = []
        self.encoder_valid_model = []
        self.place_valid_model = []

    def print_info(self, epoch, iter_idx, start_time):
        print(
            'epoch {:<4} Iter {:<4} - time: {:<5} - loss: {:<6} (+/-{:<6}) - NDCG@1: {:<6} (+/-{:<6}) - NDCG@3: {:<6} (+/-{:<6})'.format(
                epoch,
                iter_idx,
                int(time.time() - start_time),
                np.round(self.train_loss[-1], 4),
                np.round(self.train_conf[-1], 4),
                np.round(self.train_ndcg1[-1], 4),
                np.round(self.train_ndcg1_conf[-1], 4),
                np.round(self.train_ndcg3[-1], 4),
                np.round(self.train_ndcg3_conf[-1], 4),
            )
        )
