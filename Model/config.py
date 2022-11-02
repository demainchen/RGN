import torch.nn as nn
import torch

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


import numpy as np
class Config():
    def __init__(self):
        self.epochs = 50
        self.lr = 0.0005
        self.train_data_path = '../数据集/train_335.pkl'
        self.test_data_path = '../数据集/Test_60.pkl'
        self.weight_path = '../数据集/weight.pkl'
        self.case_path='../数据集/case_study1.pkl'
        self.seed = 10
        self.split_rate = 0.2
        self.batch_size = 1
        self.save_path = '../result/'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.loss_fun = nn.CrossEntropyLoss().to(self.device)
        self.Threashold = 0.18
        self.save_txt = 'result.txt'
        self.save_model_path = 'model/'
        if not os.path.exists(self.save_path + self.save_model_path):
            os.mkdir(self.save_path + self.save_model_path)
        self.best_txt = 'best.txt'
        self.MAP_CUTOFF = 14
        self.HIDDEN_DIM = 256
        self.LAYER = 6
        self.DROPOUT = 0.3
        self.ALPHA = 0.1
        self.LAMBDA = 0.9
        self.VARIANT = True  # From GCNII
        self.best_prc=0
        self.WEIGHT_DECAY = 0
        self.BATCH_SIZE = 1
        self.NUM_CLASSES = 2  # [not bind, bind]
        self.INPUT_DIM = 256*2
        self.negative_slope = 0.3
        self.leaky_relu = nn.ReLU()
        self.heads = 1

