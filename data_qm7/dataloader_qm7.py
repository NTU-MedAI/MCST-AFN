import os
from rdkit import Chem
from torch import tensor
import torch
from torch.utils.data import DataLoader
from rdkit.Chem import AllChem  # noqa
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
from torch.nn.utils.rnn import pad_sequence

atom_list = {'<pad>': 0, 'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'S': 6, 'P': 7, 'Br': 8, 'I': 9, 'Cl': 10,
                 '<unk>': 11}

suppl = Chem.SDMolSupplier('./gdb7.sdf')

with open('./gdb7.sdf.csv', 'r') as f:
    target = f.read().split('\n')[1:-1]
    target = [float(x) for x in target]
    # target = torch.tensor(target, dtype=torch.float)

data_list = []
for i, mol in tqdm(enumerate(suppl)):
    if mol is None:
        continue
    try:
        text = suppl.GetItemText(i)
        mol = Chem.AddHs(mol)

        # Example 130669 has an error and yields a different number of atoms.
        # We discard it.
        # if i == 130669:
        #     continue
        num_atoms = mol.GetNumAtoms()
        # print(i)

        #存原子
        x_tmp = []
        atoms = mol.GetAtoms()
        for a in atoms:
            element = a.GetSymbol()
            if element in atom_list.keys():
                x_tmp.append(atom_list[element])
            else:
                x_tmp.append(atom_list['<unk>'])

        #存坐标
        pos = text.split('\n')[4:4 + num_atoms]
        pos = [[float(x) for x in line.split()[:3]] for line in pos]

        #存标签
        y = target[i]

        assert len(pos) == len(x_tmp)

        data_list.append([x_tmp, pos, y])

    except:
        print(f"Error processing molecule {i + 1}")
        continue


print(len(data_list))
data_list_length = int(0.8 * len(data_list))
train_size = (data_list_length // 8) * 8
val_test_size = len(data_list) - train_size
val_size = int(0.5 * val_test_size)
test_size = val_test_size - val_size
print(train_size)
print(val_size)
print(test_size)

# 随机打乱数据列表
random.shuffle(data_list)

# 切分数据列表
train_dataset = data_list[:train_size]
val_dataset = data_list[train_size:train_size+val_size]
test_dataset = data_list[train_size+val_size:]

def qm7_loader(dataset, mode='train'):
    x, pos_sum, y_sum = [], [], []
    for data in dataset:
        x.append(tensor(data[0]))
        pos_sum.append(tensor(data[1]))
        y_sum.append(data[2])

    x = pad_sequence(x, batch_first=True, padding_value=0)
    pos_sum = pad_sequence(pos_sum, batch_first=True, padding_value=0)
    y_sum = tensor(y_sum)
    torch.save([x, pos_sum, y_sum], f'./qm7/{mode}.pt')


qm7_loader(train_dataset,mode='train')
qm7_loader(val_dataset,mode='val')
qm7_loader(test_dataset,mode='test')



