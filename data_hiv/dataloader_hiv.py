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


data = pd.read_csv('./HIV.csv')
data = data.dropna(axis='index', how='all', subset=['HIV_active'])

train, test = train_test_split(data, test_size=0.2, random_state=42)
test, val = train_test_split(test, test_size=0.5, random_state=42)


train_set = pd.DataFrame(data=train,columns=['name','smiles','HIV_active'])
test_set = pd.DataFrame(data=test,columns=['name','smiles','HIV_active'])
val_set = pd.DataFrame(data=val,columns=['name','smiles','HIV_active'])

train_set.to_csv('./hiv/train.csv',index=False)
test_set.to_csv('./hiv/val.csv',index=False)
val_set.to_csv('./hiv/test.csv',index=False)

def obsmitosmile(smi):
    conv = openbabel.OBConversion()
    conv.SetInAndOutFormats("smi", "can")
    conv.SetOptions("K", conv.OUTOPTIONS)
    mol = openbabel.OBMol()
    conv.ReadString(mol, smi)
    smile = conv.WriteString(mol)
    smile = smile.replace('\t\n', '')
    return smile

def hiv_loader(dict, mode='train.csv', path='hiv/', dist_bar=6):
    df = pd.read_csv(path+mode)
    # df = df.iloc[0:500]
    print(len(df))
    df = df.dropna(axis='index', how='all', subset=['HIV_active'])
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

    if mode=='train.csv':
        Mode = 'train'
        Path='hiv/hiv_3D_train.mol2'
    elif mode=='test.csv':
        Mode = 'test'
        Path='hiv/hiv_3D_test.mol2'
    else :
        Mode = 'val'
        Path='hiv/hiv_3D_val.mol2'

    # 生成包含所有有效分子的分子对象的sdf文件
    w = Chem.SDWriter(Path)
    for mol, name, label in zip(mols, df['name'], df['HIV_active']):
        mol.SetProp('name', str(name))
        mol.SetProp('HIV_active', str(label))
        try:
            w.write(mol)
        except Exception as e:
            mols.remove(mol)
            continue
    w.close()

    sdf_file = f'{Path}'
    suppl = Chem.SDMolSupplier(sdf_file)
    x, pos, y = [], [], []
    for mol in suppl:
        if mol is None: continue
        if mode == 'train' and mol.GetProp('Set') != '1': continue
        if mode == 'test' and mol.GetProp('Set') != '2': continue
        if mode == 'val' and mol.GetProp('Set') != '3': continue

        y_value = int(mol.GetProp('HIV_active'))
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
    pos = pad_sequence(pos,batch_first=True, padding_value=0)
    y = tensor(y)
    torch.save([x, pos, y], f'hiv/{Mode}.pt')
    print(f'Loading {len(x)} data samples with {len(dict)} atom types.')
    # return dict



def load_final():
    atom_list = {'<pad>': 0, 'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'S': 6, 'P': 7, 'Br': 8, 'I': 9, 'Cl': 10,
                 '<unk>': 11}
    hiv_loader(atom_list, mode='train.csv')
    hiv_loader(atom_list, mode='test.csv')
    hiv_loader(atom_list, mode='val.csv')

if __name__ == '__main__':
    load_final()