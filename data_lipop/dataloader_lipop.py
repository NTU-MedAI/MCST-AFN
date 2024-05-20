import torch
from torch import tensor
from torch.nn.utils.rnn import pad_sequence
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms
import numpy as np

from sklearn.model_selection import train_test_split
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


data = pd.read_csv('./Lipophilicity.csv')
train, test = train_test_split(data, test_size=0.2, random_state=42)
test, val = train_test_split(test, test_size=0.5, random_state=42)


train_set = pd.DataFrame(data=train,columns=['smiles','exp'])
test_set = pd.DataFrame(data=test,columns=['smiles','exp'])
val_set = pd.DataFrame(data=val,columns=['smiles','exp'])

train_set.to_csv('./lipop/train.csv',index=False)
test_set.to_csv('./lipop/val.csv',index=False)
val_set.to_csv('./lipop/test.csv',index=False)

def lipop_loader(dict, mode='train.csv', path='lipop/', dist_bar=6):
    df = pd.read_csv(path+mode)
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
        except Exception as e:
            print(row['smiles'])
            delete_list.append(i)
            print(f"Error processing molecule {i + 1}: {e}")
            # mol = Chem.MolFromSmiles(row['smiles'], sanitize=False)
            # mol = Chem.AddHs(mol)
            # AllChem.Compute2DCoords(mol)
            # AllChem.MMFFOptimizeMolecule(mol)
            # mols.append(mol)

    # 删去无法处理的分子所在行
    df = df.drop(index=df.index[delete_list]).reset_index(drop=True)
    print(df)

    if mode=='train.csv':
        Mode = 'train'
        Path='lipop/lipop_3D_train.mol2'
    elif mode=='test.csv':
        Mode = 'test'
        Path='lipop/lipop_3D_test.mol2'
    else :
        Mode = 'val'
        Path='lipop/lipop_3D_val.mol2'

    # 生成包含所有有效分子的分子对象的sdf文件
    w = Chem.SDWriter(Path)
    print(len(mols),len(df['exp']))
    for mol, label in zip(mols, df['exp']):
        mol.SetProp('exp', str(label))
        w.write(mol)
    w.close()

    sdf_file = f'{Path}'
    suppl = Chem.SDMolSupplier(sdf_file)
    x, pos, y = [], [], []
    for mol in suppl:
        if mol is None: continue
        if mode == 'train' and mol.GetProp('Set') != '1': continue
        if mode == 'test' and mol.GetProp('Set') != '2': continue
        if mode == 'val' and mol.GetProp('Set') != '3': continue

        y_value = float(mol.GetProp('exp'))
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

    #用的时候再填充
    x = pad_sequence(x, batch_first=True, padding_value=0)
    pos = pad_sequence(pos,batch_first=True, padding_value=0)
    y = tensor(y)
    torch.save([x, pos, y], f'lipop/{Mode}.pt')
    # print(f'Loading {len(x)} data samples with {len(dict)} atom types.')
    # return dict



def load_final():
    atom_list = {'<pad>': 0, 'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'S': 6, 'P': 7, 'Br': 8, 'I': 9, 'Cl':10, '<unk>': 11}
    lipop_loader(atom_list, mode='train.csv')
    lipop_loader(atom_list, mode='test.csv')
    lipop_loader(atom_list, mode='val.csv')

if __name__ == '__main__':
    load_final()



