import pickle

from config import Config

config = Config()
device = config.device
import torch
import numpy as np

from torch.utils.data import Dataset


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** (-0.5)).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_inv[np.isnan(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    result = r_mat_inv @ mx @ r_mat_inv
    return result


class Train_Data(Dataset):
    def __init__(self, data_path):
        df = open(data_path, 'rb')
        self.raw_data = pickle.load(df)
        self.protein_list = list(self.raw_data.keys())

    def __getitem__(self, index):
        protein_name = self.protein_list[index]
        protein_inf = self.raw_data[protein_name]
        labels = torch.tensor(np.array([float(i) for i in protein_inf['label']]), requires_grad=False).float().to(
            config.device)
        seq_emb = torch.tensor(np.squeeze(protein_inf['seq_emb']), requires_grad=True).squeeze().to(config.device)
        structure_emb = torch.tensor(np.squeeze(normalize(protein_inf['s2'])),
                                     requires_grad=True).squeeze().to(config.device)

        dssp = torch.tensor(np.squeeze(protein_inf['dssp']), requires_grad=True).to(config.device)
        hmm = torch.tensor(np.squeeze(protein_inf['hmm']), requires_grad=True).to(config.device)
        pssm = torch.tensor(np.squeeze(protein_inf['pssm']), requires_grad=True).to(config.device)


        return dssp, hmm, pssm, seq_emb, structure_emb, labels

    def __len__(self):
        return len(self.protein_list)


class Test_Data(Dataset):
    def __init__(self, data_path):
        df = open(data_path, 'rb')
        self.raw_data = pickle.load(df)
        self.protein_list = list(self.raw_data.keys())

    def __getitem__(self, index):
        protein_name = self.protein_list[index]
        protein_inf = self.raw_data[protein_name]
        labels = torch.tensor(np.array([float(i) for i in protein_inf['label']]), requires_grad=True).float().to(
            config.device)
        seq_emb = torch.tensor(np.squeeze(protein_inf['seq_emb']), requires_grad=True).squeeze().to(config.device)
        structure_emb = torch.tensor(np.squeeze(normalize(protein_inf['s2'])),
                                     requires_grad=True).squeeze().to(config.device)

        dssp = torch.tensor(np.squeeze(protein_inf['dssp']), requires_grad=True).to(config.device)
        hmm = torch.tensor(np.squeeze(protein_inf['hmm']), requires_grad=True).to(config.device)
        pssm = torch.tensor(np.squeeze(protein_inf['pssm']), requires_grad=True).to(config.device)




        return dssp, hmm, pssm, seq_emb, structure_emb, labels

    def __len__(self):
        return len(self.protein_list)
