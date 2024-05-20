import torch
from torch import tensor
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from openbabel import openbabel
from sklearn.model_selection import train_test_split
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import csv
from tqdm import tqdm


data = pd.read_csv('./tox21.csv')
train, test = train_test_split(data, test_size=0.2, random_state=42)
test, val = train_test_split(test, test_size=0.5, random_state=42)


train_set = pd.DataFrame(data=train)
test_set = pd.DataFrame(data=test)
val_set = pd.DataFrame(data=val)

train_set.to_csv('./tox21/train.csv',index=False)
test_set.to_csv('./tox21/val.csv',index=False)
val_set.to_csv('./tox21/test.csv',index=False)

def obsmitosmile(smi):
    conv = openbabel.OBConversion()
    conv.SetInAndOutFormats("smi", "can")
    conv.SetOptions("K", conv.OUTOPTIONS)
    mol = openbabel.OBMol()
    conv.ReadString(mol, smi)
    smile = conv.WriteString(mol)
    smile = smile.replace('\t\n', '')
    return smile

def tox21_loader(dict, mode='train.csv', path='tox21/', dist_bar=6):
    with open(path+mode) as f:
        reader = csv.reader(f)
        header = next(reader)
        header.remove('smiles')
        header.remove('mol_id')
        target_list = header
        f.close()
    df = pd.read_csv(path+mode)
    print(df)
    mols = []

    for i, row in df.iterrows():
        try:
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol is None:
                mol = Chem.MolFromSmiles(obsmitosmile(row['smiles']))
            mol = Chem.AddHs(mol)
            try:
                AllChem.EmbedMolecule(mol, randomSeed=-1, maxAttempts=10)
                AllChem.MMFFOptimizeMolecule(mol)
            except Exception as e:
                pass
            mols.append(mol)
        except Exception as e:
            print(f"Error processing molecule {i + 1}: {e}")

    if mode=='train.csv':
        Mode = 'train'
        Path='tox21/tox21_3D_train.mol2'
    elif mode=='test.csv':
        Mode = 'test'
        Path='tox21/tox21_3D_test.mol2'
    else :
        Mode = 'val'
        Path='tox21/tox21_3D_val.mol2'

    # 生成包含所有有效分子的分子对象的sdf文件
    w = Chem.SDWriter(Path)
    for mol in mols:
        for target in target_list:
            target = str(target)
            for label in df[target]:
                if label == '':
                    continue
                else:
                    mol.SetProp(target, str(label))
        try:
            w.write(mol)
        except Exception as e:
            pass
    w.close()

    sdf_file = f'{Path}'
    suppl = Chem.SDMolSupplier(sdf_file)
    x, pos = [], []
    y_sum = []
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

            y_value = int(mol.GetProp(target))
            if y_value == 1:
                y += [1]
            else:
                y += [0]

            if i == 1:
                num_conformers = mol.GetNumConformers()
                if num_conformers > 0:
                    conf = mol.GetConformer(0)  # 获取第一个构象
                else:
                    AllChem.Compute2DCoords(mol)
                    try:
                        AllChem.EmbedMolecule(mol, randomSeed=-1, maxAttempts=1000)
                        AllChem.MMFFOptimizeMolecule(mol)
                        conf = mol.GetConformer()
                    except Exception as e:
                        pass
                # conf = mol.GetConformer()
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
        print(len(y))
        y_sum.append(y)
        print(len(y_sum))

    x = pad_sequence(x, batch_first=True, padding_value=0)
    pos = pad_sequence(pos,batch_first=True, padding_value=0)
    y_sum = tensor(y_sum)
    torch.save([x, pos, y_sum], f'tox21/{Mode}.pt')
    print(f'Loading {len(x)} data samples with {len(dict)} atom types.')
    # return dict



def load_final():
    atom_list = {'<pad>': 0, 'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'S': 6, 'P': 7, 'Br': 8, 'I': 9, 'Cl': 10,
                 '<unk>': 11, '<mask>': 12}
    tox21_loader(atom_list, mode='train.csv')
    tox21_loader(atom_list, mode='test.csv')
    tox21_loader(atom_list, mode='val.csv')

if __name__ == '__main__':
    load_final()