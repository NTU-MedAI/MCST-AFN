import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

atom_dict = {'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'S': 6, 'P': 7, 'Br':  8, 'I': 9, 'Cl':10, '<unk>':11}

class PDBMDDataset():
    def __init__(self, backprop, pdb_list, prompt=None, data_dir='data/md/', max_len=1e2, noise=True, echo=False):
        self.backprop = backprop
        self.noise = noise
        x_0_, x_t_ = [], []
        feats_, prompt_ = [], []
        # if prompt is None or len(prompt) <= 1: prompt = [4] #修改对应
        prompt = prompt

        # 遍历所有的pdb
        for file in pdb_list:
            if not file.endswith('.npy'):
                if echo: print(f'Warning: data {file} is not a numpy file.')
                continue
            d = np.load(data_dir + file, allow_pickle=True).item()

            # x.shape: (T,N,3)
            x = torch.tensor(d['pos'])
            output_list = replace_elements(d['atoms'][:x.shape[1]])
            # print(output_list)
            # print([atom_dict[x] for x in output_list])
            z = torch.tensor([atom_dict[x1] for x1 in output_list])
            n_times = len(x)
            if backprop:
                x = x[:int(0.8 * n_times)]
                x = x[::5]
            else:
                x = x[int(0.8 * n_times):]
                x = x[::5]

            # 遍历所有的prompt
            for i in prompt:
                n = i         #修改对应
            n_x_split = len(x)
            for pt in prompt:
                x_0 = x[:-n]
                x_t = x[pt:n_x_split-n+pt]
                x_0_.append(x_0)
                x_t_.append(x_t)
                feats_ += [z] * len(x_0)
                prompt_ += [pt] * len(x_0)
        self.mole_idx = pad_sequence(feats_, batch_first=True, padding_value=0)[:, :max_len]
        self.prompt = torch.tensor(prompt_).unsqueeze(-1)
        for i in range(len(x_0_)):
            x_0_[i] = F.pad(x_0_[i], (0, 0, 0, self.mole_idx.shape[-1] - x_0_[i].shape[1]))
            x_t_[i] = F.pad(x_t_[i], (0, 0, 0, self.mole_idx.shape[-1] - x_t_[i].shape[1]))
        self.x_0, self.x_t = torch.cat(x_0_)[:, :max_len], torch.cat(x_t_)[:, :max_len]
        if echo: print('Got {:d} molecule!'.format(len(x_0_)))

    def __getitem__(self, i):
        # noise均值为0，标准差为0.1
        if self.backprop and self.noise:
            return self.x_0[i] + torch.randn_like(self.x_0[i]) / 10, self.mole_idx[i], self.x_t[i], self.prompt[i]

        # 返初始时刻的坐标，原子种类，结束时刻的坐标
        return self.x_0[i], self.mole_idx[i], self.x_t[i], self.prompt[i]

    def __len__(self):
        return len(self.x_0)

def replace_elements(lst):
    atom_dict = {'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'S': 6, 'P': 7, 'Br': 8, 'I': 9, 'Cl': 10}
    replaced_lst = [key for c in lst for key, value in atom_dict.items() if c == value]
    return replaced_lst






