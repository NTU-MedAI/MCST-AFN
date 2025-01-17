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
import math


def obsmitosmile(smi):
    conv = openbabel.OBConversion()
    conv.SetInAndOutFormats("smi", "can")
    conv.SetOptions("K", conv.OUTOPTIONS)
    mol = openbabel.OBMol()
    conv.ReadString(mol, smi)
    smile = conv.WriteString(mol)
    smile = smile.replace('\t\n', '')
    return smile

def ESOL_loader(dict, path='delaney-processed.csv'):
    df = pd.read_csv(path)
    print(len(df))
    df = df.dropna(axis='index', how='all', subset=['measured log solubility in mols per litre'])
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

    # 删去无法处理的分子所在行
    df = df.drop(index=df.index[delete_list]).reset_index(drop=True)
    print(df)

    Mode = 'full'
    Path = 'ESOL/ESOL_3D_full.mol2'

    # 生成包含所有有效分子的分子对象的sdf文件
    w = Chem.SDWriter(Path)
    for mol, label in zip(mols, df['measured log solubility in mols per litre']):
        if label == 'nan' or mol is None or math.isnan(label):
            continue
            print(label)
            print(mol)
        else:
            mol.SetProp('measured log solubility in mols per litre', str(label))
            w.write(mol)
    w.close()

    sdf_file = f'{Path}'
    suppl = Chem.SDMolSupplier(sdf_file)
    x, pos, y = [], [], []
    for mol in suppl:
        if mol is None: continue

        y_value = float(mol.GetProp('measured log solubility in mols per litre'))
        y.append(y_value)

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

    x = pad_sequence(x, batch_first=True, padding_value=0)
    pos = pad_sequence(pos,batch_first=True, padding_value=0)
    y = tensor(y)
    torch.save([x, pos, y], f'ESOL/{Mode}.pt')
    print(f'Loading {len(x)} data samples with {len(dict)} atom types.')
    # return dict



def load_final():
    atom_list = {'<pad>': 0, 'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'S': 6, 'P': 7, 'Br': 8, 'I': 9, 'Cl': 10,
                 '<unk>': 11}
    ESOL_loader(atom_list)

if __name__ == '__main__':
    load_final()