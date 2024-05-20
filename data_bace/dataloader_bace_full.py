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

data = pd.read_csv('./bace.csv')

def obsmitosmile(smi):
    conv = openbabel.OBConversion()
    conv.SetInAndOutFormats("smi", "can")
    conv.SetOptions("K", conv.OUTOPTIONS)
    mol = openbabel.OBMol()
    conv.ReadString(mol, smi)
    smile = conv.WriteString(mol)
    smile = smile.replace('\t\n', '')
    return smile

def bace_loader(dict, path='bace.csv'):
    df = pd.read_csv(path)
    mols = []
    delete_list = []
    print(df)

    for i, row in df.iterrows():
        try:
            mol = Chem.MolFromSmiles(row['mol'])
            if mol is None:
                mol = Chem.MolFromSmiles(obsmitosmile(row['mol']))
            mol = Chem.AddHs(mol)
            try:
                AllChem.EmbedMolecule(mol, randomSeed=-1, maxAttempts=1000)
                AllChem.MMFFOptimizeMolecule(mol)
            except Exception as e:
                pass
            mols.append(mol)

        except Exception as e:
            print(row['mol'])
            delete_list.append(i)
            print(f"Error processing molecule {i + 1}: {e}")

    df = df.drop(index=df.index[delete_list]).reset_index(drop=True)
    print(df)

    Mode = 'full'
    Path='bace/bace_3D_full.mol2'

    # 生成包含所有有效分子的分子对象的sdf文件
    w = Chem.SDWriter(Path)
    for mol, name, label in zip(mols, df['CID'], df['Class']):
        mol.SetProp('CID', str(name))
        mol.SetProp('Class', str(label))
        w.write(mol)
    w.close()

    sdf_file = f'{Path}'
    suppl = Chem.SDMolSupplier(sdf_file)
    x, pos, y = [], [], []
    for mol in suppl:
        if mol is None: continue

        y_value = int(mol.GetProp('Class'))
        if y_value == 1:
            y += [1]
        else:
            y += [0]

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

    x = pad_sequence(x, batch_first=True, padding_value=0)
    pos = pad_sequence(pos, batch_first=True, padding_value=0)
    y = tensor(y)
    torch.save([x, pos, y], f'bace/{Mode}.pt')
    print(f'Loading {len(x)} data samples with {len(dict)} atom types.')
    # return dict



def load_final():
    atom_list = {'<pad>':0, 'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'S': 6, 'P': 7, 'Br':  8, 'I': 9, 'Cl':10, '<unk>':11,'<mask>':12}
    bace_loader(atom_list)

if __name__ == '__main__':
    load_final()