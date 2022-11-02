import numpy

from config import Config

config = Config()
device = config.device
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from layers import deepGCN
from data import Train_Data, Test_Data
from sklearn.metrics import roc_curve
import warnings
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_recall_curve

warnings.filterwarnings("ignore")


config = Config()
from datetime import datetime
import sklearn.metrics as metrics

from sklearn.metrics import average_precision_score
from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score

from sklearn.metrics import roc_curve, auc,roc_auc_score
import torch

import pandas as pd
def eval_rs(pred, label, mode, model):
    fpr, tpr, _ = roc_curve(label, pred)

    precision, recall, thresholds = precision_recall_curve(label, pred)

    auroc = metrics.roc_auc_score(label, pred)
    auprc = average_precision_score(label, pred)

    for i in range(0, len(pred)):
        if (pred[i] > config.Threashold):
            pred[i] = 1
        else:
            pred[i] = 0
    rs=''
    for i in pred:
        rs+=str(i)
    print(rs)
    acc1 = accuracy_score(label, pred, sample_weight=None)
    # spec1 = spec1 + (cm1[0, 0]) / (cm1[0, 0] + cm1[0, 1])
    recall = recall_score(label, pred, sample_weight=None)
    prec1 = precision_score(label, pred, sample_weight=None)
    f1 = f1_score(label, pred)
    mcc = matthews_corrcoef(label, pred)
    rs = 'mode={},acc={},precision={},recall={},F1={},MCC={},AUROC={},AUPRC={},time={}'.format(mode,
                                                                                               acc1,
                                                                                               prec1, recall,
                                                                                               f1,
                                                                                               mcc, auroc,
                                                                                               auprc,
                                                                                               datetime.now().strftime(
                                                                                                   "%Y-%m-%d %H:%M:%S"))

    print(rs)
    return auroc, auprc, acc1, recall, prec1, f1, mcc


def eval(eval_data_loader, model):
    best_prc=0
    pred = []
    label = []
    l = 0
    for i, (dssp, hmm, pssm, seq_emb, structure_emb, labels) in enumerate(eval_data_loader):
        # Every data instance is an input + label pair
        seq_emb = seq_emb.squeeze().to(torch.float32).to(config.device)
        structure_emb = structure_emb.squeeze().to(torch.float32).to(config.device)
        labels = labels.squeeze().unsqueeze(dim=-1).to(torch.float32).to(config.device)
        dssp = dssp.squeeze().to(torch.float32).to(config.device)
        hmm = hmm.squeeze().to(torch.float32).to(config.device)
        pssm = pssm.squeeze().to(torch.float32).to(config.device)
        node_features = torch.cat((pssm, hmm, dssp), dim=1).to(torch.float).to(config.device)
        adj = structure_emb

        y_pred = model(node_features, adj, seq_emb)
        softmax = torch.nn.Softmax(dim=1)
        y_pred = softmax(y_pred)

        y_pred = y_pred.cpu().detach().numpy()
        pred += [p[1] for p in y_pred]
        label += [float(l) for l in labels]

    eval_rs(pred, label, 'test', model)
    return l / len(eval_data_loader)




HIDDEN_DIM = config.HIDDEN_DIM
DROPOUT = config.DROPOUT
ALPHA = config.ALPHA
LAMBDA = config.LAMBDA
VARIANT = config.VARIANT  # From GCNII

NUM_CLASSES = config.NUM_CLASSES  # [not bind, bind]
INPUT_DIM = config.INPUT_DIM
Heads = config.heads
Layer = config.LAYER

# Seed
SEED = config.seed
np.random.seed(SEED)
torch.manual_seed(SEED)


import os
base_path='saved/335_60'
save_model_name=os.listdir(base_path)
save_model_path=[base_path+'/'+i for i in save_model_name]

test_60_path='Test_60.pkl'

from config import Config
test_data = Test_Data(data_path=test_60_path)

for i in save_model_path:
    model = deepGCN(Layer, INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, DROPOUT, LAMBDA, ALPHA, VARIANT,
                    heads=Heads).to(
        config.device)
    model.load_state_dict(torch.load(i,map_location='cpu')['state_dict'])
    eval(test_data, model)





