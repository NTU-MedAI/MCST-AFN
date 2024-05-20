import torch
from torch import tensor
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from openbabel import openbabel
from sklearn.model_selection import train_test_split
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


data = pd.read_csv('./clintox.csv')
train, test = train_test_split(data, test_size=0.19, random_state=42)
test, val = train_test_split(test, test_size=0.5, random_state=42)


train_set = pd.DataFrame(data=train,columns=['smiles','FDA_APPROVED','CT_TOX'])
test_set = pd.DataFrame(data=test,columns=['smiles','FDA_APPROVED','CT_TOX'])
val_set = pd.DataFrame(data=val,columns=['smiles','FDA_APPROVED','CT_TOX'])

train_set.to_csv('./Clintox/train.csv',index=False)
test_set.to_csv('./Clintox/val.csv',index=False)
val_set.to_csv('./Clintox/test.csv',index=False)

def obsmitosmile(smi):
    conv = openbabel.OBConversion()
    conv.SetInAndOutFormats("smi", "can")
    conv.SetOptions("K", conv.OUTOPTIONS)
    mol = openbabel.OBMol()
    conv.ReadString(mol, smi)
    smile = conv.WriteString(mol)
    smile = smile.replace('\t\n', '')
    return smile

def Clintox_loader(dict, mode='train.csv', path='Clintox/', dist_bar=6):
    df = pd.read_csv(path+mode)
    # df = df.iloc[0:500]
    print(df)
    print(type(df['smiles']))
    mols = []
    delete_list = []

    for i, row in df.iterrows():
        try:
            # mol = Chem.MolFromSmiles(row['smiles'])
            # if mol is None:
            #     mol = Chem.MolFromSmiles(obsmitosmile(row['smiles']))
            # mol = Chem.AddHs(mol)
            # try:
            #     AllChem.EmbedMolecule(mol, randomSeed=-1, maxAttempts=1000)
            #     AllChem.MMFFOptimizeMolecule(mol)
            # except Exception as e:
            #     pass
            # mols.append(mol)
            mol = Chem.MolFromSmiles(row['smiles'])
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=-1, maxAttempts=1000)
            AllChem.MMFFOptimizeMolecule(mol)
            mols.append(mol)
        except Exception as e:
            print(row['smiles'])
            delete_list.append(i)
            print(f"Error processing molecule {i + 1}: {e}")

    #删去无法处理的分子所在行
    df = df.drop(index=df.index[delete_list]).reset_index(drop=True)
    print(df)

    if mode=='train.csv':
        Mode = 'train'
        Path='Clintox/Clintox_3D_train.mol2'
    elif mode=='test.csv':
        Mode = 'test'
        Path='Clintox/Clintox_3D_test.mol2'
    else :
        Mode = 'val'
        Path='Clintox/Clintox_3D_val.mol2'

    # 生成包含所有有效分子的分子对象的sdf文件
    w = Chem.SDWriter(Path)
    for mol, label_1, label_2 in zip(mols, df['FDA_APPROVED'], df['CT_TOX']):
        mol.SetProp('FDA_APPROVED', str(label_1))
        mol.SetProp('CT_TOX', str(label_2))
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
        if mode == 'train' and mol.GetProp('Set') != '1': continue
        if mode == 'test' and mol.GetProp('Set') != '2': continue
        if mode == 'val' and mol.GetProp('Set') != '3': continue

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

        # num_conformers = mol.GetNumConformers()
        # if num_conformers > 0:
        #     conf = mol.GetConformer(0)  # 获取第一个构象
        # else:
        #     AllChem.Compute2DCoords(mol)
        #     try:
        #         AllChem.EmbedMolecule(mol, randomSeed=-1, maxAttempts=1000)
        #         AllChem.MMFFOptimizeMolecule(mol)
        #         conf = mol.GetConformer()
        #     except Exception as e:
        #         pass
        # # conf = mol.GetConformer()
        # atoms = mol.GetAtoms()
        # positions = np.array([list(conf.GetAtomPosition(a.GetIdx())) for a in atoms])
        #
        # pos.append(tensor(positions[:]))
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
    y_1 = tensor(y_1)
    y_2 = tensor(y_2)
    torch.save([x, pos, y_1, y_2], f'Clintox/{Mode}.pt')
    print(f'Loading {len(x)} data samples with {len(dict)} atom types.')
    # return dict



def load_final():
    atom_list = {'<pad>': 0, 'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'S': 6, 'P': 7, 'Br': 8, 'I': 9, 'Cl': 10,
                 '<unk>': 11, '<mask>': 12}
    Clintox_loader(atom_list, mode='train.csv')
    Clintox_loader(atom_list, mode='test.csv')
    Clintox_loader(atom_list, mode='val.csv')

if __name__ == '__main__':
    load_final()