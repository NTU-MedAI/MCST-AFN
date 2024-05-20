import torch
from torch import tensor
from torch.nn.utils.rnn import pad_sequence
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms
import numpy as np
from openbabel import openbabel
from sklearn.model_selection import train_test_split
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import math

data = pd.read_csv('./clintox.csv')

def obsmitosmile(smi):
    conv = openbabel.OBConversion()
    conv.SetInAndOutFormats("smi", "can")
    conv.SetOptions("K", conv.OUTOPTIONS)
    mol = openbabel.OBMol()
    conv.ReadString(mol, smi)
    smile = conv.WriteString(mol)
    smile = smile.replace('\t\n', '')
    return smile

def Clintox_loader(dict, path='clintox.csv'):
    df = pd.read_csv(path)
    mols = []
    delete_list = []
    print(df)

    for i, row in df.iterrows():
        try:
            mol = Chem.MolFromSmiles(row['smiles'])
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=-1, maxAttempts=1000)
            AllChem.MMFFOptimizeMolecule(mol)
            mols.append(mol)
            # mol = Chem.MolFromSmiles(row['mol'])
            # mol = Chem.AddHs(mol)
            # AllChem.EmbedMolecule(mol, randomSeed=-1, maxAttempts=1000)
            # AllChem.MMFFOptimizeMolecule(mol)
            # mols.append(mol)
        except Exception as e:
            print(row['smiles'])
            delete_list.append(i)
            print(f"Error processing molecule {i + 1}: {e}")
    # 删去无法处理的分子所在行
    df = df.drop(index=df.index[delete_list]).reset_index(drop=True)
    print(df)

    Mode = 'full'
    Path='Clintox/Clintox_3D_full.mol2'

    # 生成包含所有有效分子的分子对象的sdf文件
    w = Chem.SDWriter(Path)
    for mol, name, p_np in zip(mols, df['FDA_APPROVED'], df['CT_TOX']):
        mol.SetProp('FDA_APPROVED', str(name))
        mol.SetProp('CT_TOX', str(p_np))
        try:
            w.write(mol)
        except Exception as e:
            mols.remove(mol)
            continue
    w.close()

    sdf_file = f'{Path}'
    suppl = Chem.SDMolSupplier(sdf_file)
    x, pos, y_1, y_2 = [], [], [], []
    for mol in suppl:
        if mol is None: continue

        y_1_value = int(mol.GetProp('FDA_APPROVED'))
        y_2_value = int(mol.GetProp('CT_TOX'))
        if y_1_value == 1:
            y_1 += [1]
        else:
            y_1 += [0]
        if y_2_value == 1:
            y_2 += [1]
        else:
            y_2 += [0]

        conf = mol.GetConformer()
        atoms = mol.GetAtoms()
        positions = np.array([list(conf.GetAtomPosition(a.GetIdx())) for a in atoms])

        pos.append(tensor(positions[:]))
        # conf = mol.GetConformer()
        # atoms = mol.GetAtoms()
        # positions = np.array([list(conf.GetAtomPosition(a.GetIdx())) for a in atoms])
        #
        # pos.append(tensor(positions[:]))
        x_tmp = []
        for a in atoms:
            element = a.GetSymbol()
            if element in dict.keys():
                x_tmp.append(dict[element])
            else:
                x_tmp.append(dict['<unk>'])
        x.append(tensor(x_tmp))

    x = pad_sequence(x, batch_first=True, padding_value=0)
    pos = pad_sequence(pos, batch_first=True, padding_value=0)
    y_1 = tensor(y_1)
    y_2 = tensor(y_2)
    torch.save([x, pos, y_1, y_2], f'Clintox/{Mode}.pt')
    print(f'Loading {len(x)} data samples with {len(dict)} atom types.')
    # return dict



def load_final():
    atom_list = {'<pad>':0, 'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'S': 6, 'P': 7, 'Br':  8, 'I': 9, 'Cl':10, '<unk>':11,'<mask>':12}
    Clintox_loader(atom_list)

if __name__ == '__main__':
    load_final()