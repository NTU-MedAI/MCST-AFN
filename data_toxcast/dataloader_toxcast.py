import torch
from torch import tensor
from torch.nn.utils.rnn import pad_sequence
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms
import numpy as np
from openbabel import openbabel
from sklearn.model_selection import train_test_split
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import csv
from tqdm import tqdm

data = pd.read_csv('toxcast_data.csv')
data = data.dropna(axis='index', how='all', subset=['TOX21_TR_LUC_GH3_Antagonist'])

train, test = train_test_split(data, test_size=0.2, random_state=42)
test, val = train_test_split(test, test_size=0.5, random_state=42)



train_set = pd.DataFrame(data=train)
test_set = pd.DataFrame(data=test)
val_set = pd.DataFrame(data=val)

train_set.to_csv('./toxcast/train.csv',index=False)
test_set.to_csv('./toxcast/val.csv',index=False)
val_set.to_csv('./toxcast/test.csv',index=False)

def obsmitosmile(smi):
    conv = openbabel.OBConversion()
    conv.SetInAndOutFormats("smi", "can")
    conv.SetOptions("K", conv.OUTOPTIONS)
    mol = openbabel.OBMol()
    conv.ReadString(mol, smi)
    smile = conv.WriteString(mol)
    smile = smile.replace('\t\n', '')
    return smile

def toxcast_loader(dict, mode='train.csv', path='toxcast/', dist_bar=6):
    with open(path+mode) as f:
        reader = csv.reader(f)
        header = next(reader)
        header.remove('smiles')
        target_list = header
        print(target_list)
        f.close()
    df = pd.read_csv(path+mode)
    print(len(df))
    df = df.dropna(axis='index', how='all', subset=['TOX21_TR_LUC_GH3_Antagonist'])
    print(len(df))
    mols = []
    delete_list = []

    for i, row in df.iterrows():
        try:
            mol = Chem.MolFromSmiles(row['smiles'])
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=-1, maxAttempts=1000)
            AllChem.MMFFOptimizeMolecule(mol)
            mols.append(mol)
        except Exception as e:
            print(row['smiles'])
            delete_list.append(i)
            print(f"Error processing molecule {i + 1}: {e}")

    df = df.drop(index=df.index[delete_list]).reset_index(drop=True)
    print(df)

    if mode=='train.csv':
        Mode = 'train'
        Path='toxcast/toxcast_3D_train.mol2'
    elif mode=='test.csv':
        Mode = 'test'
        Path='toxcast/toxcast_3D_test.mol2'
    else :
        Mode = 'val'
        Path='toxcast/toxcast_3D_val.mol2'

    # 生成包含所有有效分子的分子对象的sdf文件
    w = Chem.SDWriter(Path)
    for index,mol in enumerate(mols):
        for target in target_list:
            target = str(target)
            label = df[target][index]
            mol.SetProp(target, str(label))
        try:
            w.write(mol)
        except Exception as e:
            mols.remove(mol)
            continue
    w.close()

    sdf_file = f'{Path}'
    suppl = Chem.SDMolSupplier(sdf_file)
    x, pos, y_sum = [], [], []
    i = 0
    for target in target_list:
        i += 1
        target = str(target)
        y = []
        for mol in tqdm(suppl):
            if mol is None: continue
            if mode == 'train' and mol.GetProp('Set') != '1': continue
            if mode == 'test' and mol.GetProp('Set') != '2': continue
            if mode == 'val' and mol.GetProp('Set') != '3': continue

            y_value = int(float(mol.GetProp(target)))
            if y_value == 1:
                y += [1]
            else:
                y += [0]

            if i == 1:
                conf = mol.GetConformer()
                atoms = mol.GetAtoms()
                positions = np.array([list(conf.GetAtomPosition(a.GetIdx())) for a in atoms])

                pos.append(tensor(positions[:]))

                x_tmp = []
                for a in atoms:
                    element = a.GetSymbol()
                    if element in dict.keys():
                        x_tmp.append(dict[element])
                    else:
                        x_tmp.append(dict['<unk>'])
                x.append(tensor(x_tmp))
        y_sum.append(y)
        print(len(y_sum))

    x = pad_sequence(x, batch_first=True, padding_value=0)

    #RuntimeError: The size of tensor a (3) must match the size of tensor b (0) at non-singleton dimension 1
    pos = pad_sequence(pos,batch_first=True, padding_value=0)
    y_sum = tensor(y_sum)
    torch.save([x, pos, y_sum], f'toxcast/{Mode}.pt')
    print(f'Loading {len(x)} data samples with {len(dict)} atom types.')
    # return dict

def load_final():
    atom_list = {'<pad>': 0, 'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'S': 6, 'P': 7, 'Br': 8, 'I': 9, 'Cl': 10,
                 '<unk>': 11}
    toxcast_loader(atom_list, mode='train.csv')
    toxcast_loader(atom_list, mode='test.csv')
    toxcast_loader(atom_list, mode='val.csv')

if __name__ == '__main__':
    load_final()