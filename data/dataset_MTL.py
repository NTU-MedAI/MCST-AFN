import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops
from rdkit.Chem import AllChem
from openbabel import openbabel
# import openbabel as ob
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

smiles_str2num = {'<pad>':0, 'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'S': 6, 'P': 7, 'Br':  8, 'I': 9, 'Cl':10, '<unk>':11,'<mask>':12}

smiles_num2str =  {i:j for j,i in smiles_str2num.items()}

#很重要，为缩减数据处理时间，可用for+if+continue组合跳过无效分子
def obsmitosmile(smi):
    conv = openbabel.OBConversion()
    conv.SetInAndOutFormats("smi", "can")
    conv.SetOptions("K", conv.OUTOPTIONS)
    mol = openbabel.OBMol()
    conv.ReadString(mol, smi)
    smile = conv.WriteString(mol)
    smile = smile.replace('\t\n', '')
    return smile


def smiles2adjoin(smiles,explicit_hydrogens=True,canonical_atom_order=False):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print('error')
        mol = Chem.MolFromSmiles(obsmitosmile(smiles))
        # assert mol is not None, smiles + ' is not valid '
        if mol is None:
            return None, None

    if explicit_hydrogens:
        mol = Chem.AddHs(mol)
    else:
        mol = Chem.RemoveHs(mol)

    try:
        AllChem.EmbedMolecule(mol, randomSeed=-1, maxAttempts=1000)
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception as e:
        pass

    if canonical_atom_order:
        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        mol = rdmolops.RenumberAtoms(mol, new_order)
    num_atoms = mol.GetNumAtoms()
    atoms_list = []
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        atoms_list.append(atom.GetSymbol())
    # conf = mol.GetConformer()
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
            return None, None

    positions = conf.GetPositions()  # 获取坐标数组
    adjoin_matrix = np.eye(num_atoms)

    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        adjoin_matrix[u,v] = 1.0
        adjoin_matrix[v,u] = 1.0
    return atoms_list,positions

class Smiles_Bert_Dataset(Dataset):
    def __init__(self, path, Smiles_head):
        if path.endswith('txt'):
            self.df = pd.read_csv(path,sep='\t',header=None,names=[Smiles_head])
        else:
            self.df = pd.read_csv(path)

        self.data = self.df[Smiles_head].to_numpy().reshape(-1).tolist()  #模型导致对数量有要求
        self.vocab = smiles_str2num
        self.devocab = smiles_num2str

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        smiles = self.data[item]
        x, pos, y, weights = self.numerical_smiles(smiles)
        return x, pos, y, weights

    def numerical_smiles(self, smiles):
        atoms_list, pos = smiles2adjoin(smiles, explicit_hydrogens=True)
        if atoms_list is None and pos is None:
            return None,None,None,None
        nums_list = [smiles_str2num.get(i, smiles_str2num['<unk>']) for i in atoms_list]

        choices = np.random.permutation(len(nums_list) - 1)[:max(int(len(nums_list) * 0.15), 1)] + 1
        y = np.array(nums_list).astype('int16')
        weight = np.zeros(len(nums_list))
        for i in choices:
            rand = np.random.rand()
            weight[i] = 1
            if rand < 0.8:
                nums_list[i] = smiles_str2num['<mask>']
            elif rand < 0.9:
                nums_list[i] = int(np.random.rand() * 14 + 1)

        x = np.array(nums_list).astype('int16')
        weights = weight.astype('float16')
        return x, pos, y, weights


class Pretrain_Collater():
    def __init__(self):
        super(Pretrain_Collater, self).__init__()
    def __call__(self,data):
        xs, poss, ys, weights = zip(*data)

        xs = pad_sequence([torch.from_numpy(np.array(x)) for x in xs], batch_first=True).long().to(device)
        ys = pad_sequence([torch.from_numpy(np.array(y)) for y in ys], batch_first=True).long().to(device)
        poss = pad_sequence([torch.from_numpy(np.array(pos)) for pos in poss], batch_first=True).float().to(device)
        weights = pad_sequence([torch.from_numpy(np.array(weight)) for weight in weights], batch_first=True).float().to(device)

        return xs, poss, ys, weights

