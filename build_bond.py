import torch
import numpy as np

from rdkit import Chem, Geometry

from src import const
import json
import os

# add
import tqdm
from src import metrics, utils, delinker
from src.datasets import (
    ZincDataset, MOADDataset, create_templates_for_linker_generation, get_dataloader, collate
)

def create_conformer(coords):
    conformer = Chem.Conformer()
    for i, (x, y, z) in enumerate(coords):
        conformer.SetAtomPosition(i, Geometry.Point3D(x, y, z))
    return conformer


def build_molecules(one_hot, x, node_mask, is_geom, margins=const.MARGINS_EDM):
    molecules = []
    for i in range(len(one_hot)):
        mask = node_mask[i].squeeze() == 1
        atom_types = one_hot[i][mask].argmax(dim=1).detach().cpu()
        positions = x[i][mask].detach().cpu()
        mol = build_molecule(positions, atom_types, is_geom, margins=margins)
        molecules.append(mol)

    return molecules


def build_molecule(positions, atom_types, is_geom, margins=const.MARGINS_EDM):
    idx2atom = const.GEOM_IDX2ATOM if is_geom else const.IDX2ATOM
    X, A, E = build_xae_molecule(positions, atom_types, is_geom=is_geom, margins=margins)
    mol = Chem.RWMol()
    mol1 = Chem.RWMol()
    fralab=torch.load('./datasets/test_protac_0.pt')[0]['bonds']   #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!refine it
    mtot=fralab.shape[0]
    for atom in X:
        a = Chem.Atom(idx2atom[atom.item()])
        mol.AddAtom(a)
        mol1.AddAtom(a)
    

    all_bonds = torch.nonzero(A)
    msta=1
    for bond in all_bonds:
       # print(bond[0].item())
       # print(bond[1].item())
       # print(E[bond[0], bond[1]].item())
       # print(const.BOND_DICT[2])
        mol.AddBond(bond[0].item(), bond[1].item(), const.BOND_DICT[E[bond[0], bond[1]].item()])
        if msta <= mtot :
            bond0=int(fralab[msta-1][0])
            bond1=int(fralab[msta-1][1])
            mol1.AddBond(bond0, bond1, const.BOND_DICT[int(fralab[msta-1][2])])
        else:
            mol1.AddBond(bond[0].item(), bond[1].item(), const.BOND_DICT[E[bond[0], bond[1]].item()])
        msta = msta+1

    mol.AddConformer(create_conformer(positions.detach().cpu().numpy().astype(np.float64)))
    return mol1


def build_xae_molecule(positions, atom_types, is_geom, margins=const.MARGINS_EDM):
    """ Returns a triplet (X, A, E): atom_types, adjacency matrix, edge_types
        args:
        positions: N x 3  (already masked to keep final number nodes)
        atom_types: N
        returns:
        X: N         (int)
        A: N x N     (bool) (binary adjacency matrix)
        E: N x N     (int)  (bond type, 0 if no bond) such that A = E.bool()
    """
    n = positions.shape[0]
    X = atom_types
    A = torch.zeros((n, n), dtype=torch.bool)
    E = torch.zeros((n, n), dtype=torch.int)

    idx2atom = const.GEOM_IDX2ATOM if is_geom else const.IDX2ATOM

    pos = positions.unsqueeze(0)
    dists = torch.cdist(pos, pos, p=2).squeeze(0)
    # print(pos,'**********', pos.shape)
    # print(dists,'**********', dists.shape)
    # print(atom_types,'**********', atom_types.shape)
    # exit()
    for i in range(n):
        for j in range(i):

            pair = sorted([atom_types[i], atom_types[j]])
            order = get_bond_order(idx2atom[pair[0].item()], idx2atom[pair[1].item()], dists[i, j], margins=margins)

            # TODO: a batched version of get_bond_order to avoid the for loop
            if order > 0:
                # Warning: the graph should be DIRECTED
                if order==1:
                    print(i+1, j+1, order, '%0.1f' % (dists[i, j].item()*100), idx2atom[pair[0].item()], idx2atom[pair[1].item()], const.BONDS_1[idx2atom[pair[0].item()]][idx2atom[pair[1].item()]])
                elif order==2:
                    print(i+1, j+1, order, '%0.1f' % (dists[i, j].item()*100), idx2atom[pair[0].item()], idx2atom[pair[1].item()], const.BONDS_2[idx2atom[pair[0].item()]][idx2atom[pair[1].item()]])
                elif order==3:
                    print(i+1, j+1, order, '%0.1f' % (dists[i, j].item()*100), idx2atom[pair[0].item()], idx2atom[pair[1].item()], const.BONDS_3[idx2atom[pair[0].item()]][idx2atom[pair[1].item()]])
                else:
                    print(i+1, j+1, order, '%0.1f' % (dists[i, j].item()*100), idx2atom[pair[0].item()], idx2atom[pair[1].item()], const.BONDS_4[idx2atom[pair[0].item()]][idx2atom[pair[1].item()]])

                
                A[i, j] = 1
                E[i, j] = order
            else:
                pass
                # print('no bond', i, j, order, '%0.1f' % (dists[i, j].item()*100), idx2atom[pair[0].item()], idx2atom[pair[1].item()], const.BONDS_1[idx2atom[pair[0].item()]][idx2atom[pair[1].item()]])


    return X, A, E



def get_bond_order(atom1, atom2, distance, check_exists=True, margins=const.MARGINS_EDM):
    distance = 100 * distance  # We change the metric

    # Check exists for large molecules where some atom pairs do not have a
    # typical bond length.
    if check_exists:
        if atom1 not in const.BONDS_1:
            return 0
        if atom2 not in const.BONDS_1[atom1]:
            return 0
    
    MARGINS_EDM_C = [10, 3, 2, 5]
    # margin1, margin2, margin3 and margin4, have been tuned to maximize the stability of the true samples
    if distance < const.BONDS_1[atom1][atom2] + margins[0]:
        # Check if atoms is aromatic
        # if atom1 == 'C' and atom2 == 'C' and distance < const.BONDS_4[atom1][atom2] + MARGINS_EDM_C[3]:
        #     if distance < const.BONDS_2[atom1][atom2] + MARGINS_EDM_C[1]:
        #         if distance < const.BONDS_3[atom1][atom2] + MARGINS_EDM_C[2]:
        #             return 3
        #         return 2
        #     return 4

        # Check if atoms in bonds2 dictionary.
        if atom1 in const.BONDS_2 and atom2 in const.BONDS_2[atom1]:
            thr_bond2 = const.BONDS_2[atom1][atom2] + margins[1]
            if distance < thr_bond2:
                if atom1 in const.BONDS_3 and atom2 in const.BONDS_3[atom1]:
                    thr_bond3 = const.BONDS_3[atom1][atom2] + margins[2]
                    if distance < thr_bond3:
                        return 3  # Triple
                return 2  # Double
        return 1  # Single
    return 0  # No bond

    # # if aromatic atom
    # if atom1 == 'C' and atom2 == 'C' and (const.BONDS_4[atom1][atom2] - margins[3] < distance < const.BONDS_4[atom1][atom2] + margins[3]):
    #     return 4
    # # margin1, margin2 and margin3 have been tuned to maximize the stability of the QM9 true samples
    # if distance < const.BONDS_1[atom1][atom2] + margins[0]:
    #     # Check if atoms in bonds2 dictionary.
    #     if atom1 in const.BONDS_2 and atom2 in const.BONDS_2[atom1]:
    #         thr_bond2 = const.BONDS_2[atom1][atom2] + margins[1]
    #         if distance < thr_bond2:
    #             if atom1 in const.BONDS_3 and atom2 in const.BONDS_3[atom1]:
    #                 thr_bond3 = const.BONDS_3[atom1][atom2] + margins[2]
    #                 if distance < thr_bond3:
    #                     return 3  # Triple
    #             return 2  # Double
    #     return 1  # Single
    # return 0  # No bond


def load_xyz_files(path, suffix=''):
    files = []
    for fname in os.listdir(path):
        if fname.endswith(f'_{suffix}.xyz'):
            files.append(fname)
    files = sorted(files, key=lambda f: -int(f.replace(f'_{suffix}.xyz', '').split('_')[-1]))
    return [os.path.join(path, fname) for fname in files]


def load_molecule_xyz(file):
    atom2idx = const.ATOM2IDX
    idx2atom = const.IDX2ATOM
    atom2charges = const.CHARGES
    with open(file, encoding='utf8') as f:
        n_atoms = int(f.readline())
        one_hot = torch.zeros(n_atoms, len(idx2atom))
        charges = torch.zeros(n_atoms, 1)
        positions = torch.zeros(n_atoms, 3)
        f.readline()
        atoms = f.readlines()
        for i in range(n_atoms):
            atom = atoms[i].split(' ')
            atom_type = atom[0]
            one_hot[i, atom2idx[atom_type]] = 1
            position = torch.Tensor([float(e) for e in atom[1:]])
            positions[i, :] = position
            charges[i, :] = atom2charges[atom_type]
        return positions, one_hot, charges

    

'''
从test_30000_reindex.pt重建真实数据的化学键，效果不错
'''
def sample_and_analyze(dataloader):
    res = {}
    for b, data in enumerate(dataloader):
        uuid = data['uuid'].item()
        print(uuid,'**********')
        name = data['name']
        x = data['positions']
        one_hot = data['one_hot']
        charges = data['charges']
        anchors = data['anchors']
        frag_mask = data['fragment_mask']
        link_mask = data['linker_mask']
        num_atoms = data['num_atoms']
        # print(name)
        # print(x)
        # print(one_hot)
        # print(charges)
        # print(anchors)
        # print(frag_mask)
        # print(link_mask)
        # print(num_atoms)
        node_mask = data['fragment_mask'] + data['linker_mask']
        recon_molecules = build_molecules(one_hot, x, node_mask, is_geom=False)
        res[uuid] = recon_molecules[0]
    
    out_sdf_path = 'mol_recon.sdf'
    with Chem.SDWriter(open(out_sdf_path, 'w')) as writer:
        for uuid, mol in zip(res.keys(), res.values()):
            mol.SetProp('_Name', 'No_%d' % uuid)
            writer.write(mol)


def train_dataloader(collate_fn=collate):
    dataset = torch.load('datasets/test_30000_reindex.pt')
    batch_size = 1
    return get_dataloader(dataset[:10], batch_size, collate_fn=None, shuffle=False)


def json2sdf(file_path):
    idx2atom = const.IDX2ATOM
    data = read_file(file_path)
    for i, m in enumerate(data):
        print(m)
        exit()
        position = m['positions_out']
        bonds = m['graph_out']
        one_hot = np.array(m['node_features_out'])
        atom_types = np.argmax(one_hot, axis=1)
        
        mol = Chem.RWMol()
        for atom in atom_types:
            a = Chem.Atom(idx2atom[atom.item()])
            mol.AddAtom(a)
        
        for bond in bonds:
            mol.AddBond(bond[0], bond[2], const.BOND_DICT[bond[1]+1])

def cal_dist(a, b):
    flag = False
    if flag: lamb = 0.537
    else: lamb = 1
    res = (a[0]*lamb-b[0]*lamb)**2 + (a[1]*lamb-b[1]*lamb)**2 + (a[2]*lamb-b[2]*lamb)**2
    return res ** 0.5 

def check_bond(file_path):
    idx2atom = const.IDX2ATOM
    data = read_file(file_path)
    print(len(data))
    for i, m in enumerate(data):
        bonds = m['bonds']
        pos = m['positions']
        one_hot = m['one_hot']
        for bond in bonds[:]:
            atom_idx1 = bond[0]
            atom_idx2 = bond[1]
            a1 = np.argmax(np.array(one_hot[atom_idx1]))
            a2 = np.argmax(np.array(one_hot[atom_idx2]))

            bond_type = bond[2]
            dist = cal_dist(pos[atom_idx1], pos[atom_idx2]) * 100
            if bond_type == 1:
                refer_dist = const.BONDS_1[idx2atom[a1]][idx2atom[a2]]
            elif bond_type == 2:
                refer_dist = const.BONDS_2[idx2atom[a1]][idx2atom[a2]]
            elif bond_type == 3:
                refer_dist = const.BONDS_3[idx2atom[a1]][idx2atom[a2]]
            else:
                print(f'Warning: no bond_type: {bond_type}')
            print(atom_idx1.item(), atom_idx2.item(), bond_type.item(), dist.item(), idx2atom[a1], idx2atom[a2], refer_dist)


        exit()


def read_file(file_path):
    print("Loading data from %s" % file_path)
    data = torch.load(file_path)
    return data

'''
加载sample出来的.xyz文件，重建化学键, 若被定义为芳香键但是无环，就会报错，这时候把判定芳香键的代码去掉就好
''' 
def xyz2sdf(file_path , out_path):
    one_hot, x, node_mask = [], [], []
    for path in file_path:
        out = load_molecule_xyz(path)
        x.append(out[0])
        one_hot.append(out[1])
        node_mask.append(torch.ones_like(out[2]))
    
    
    recon_molecules = build_molecules(one_hot, x, node_mask, is_geom=False)
    res = recon_molecules

    #out_sdf_path = 'g0.sdf'
    #print(out_path[0])
    with Chem.SDWriter(open(out_path[0], 'a')) as writer:
       for mol in res:
            writer.write(mol)
            f = open('protac_0/5_7_smi.txt','a')
            f.write(Chem.MolToSmiles(mol)+'\n')





if __name__ == '__main__':
    # dataloader = train_dataloader()
    # sample_and_analyze(dataloader)
    xyz_inp_path=[]
    for i in range(100):
    
        xyz_inp_path.append('/home/anfeng/george/samples/test_protac_0/sampled_size/size_gnn_128/protac_difflinker_998_N_-5_linker_size_7/0/' + str(i) + '_.xyz') 
        #= [
       # '/home/anfeng/george/samples/test_protac_0/sampled_size/size_gnn_128/protac_difflinker_998_N_-10_linker_size_15/0/1_.xyz'
       # '/home/anfeng/george/samples/test_protac_0/sampled_size/size_gnn_128/protac_difflinker_998_N_-10_linker_size_15/0/2_.xyz'
       # '/home/anfeng/george/samples/test_protac_0/sampled_size/size_gnn_128/protac_difflinker_998_N_-10_linker_size_15/0/0_.xyz'
#        /home/test2/george/DiffLinker/samples/test_30000_reindex/sampled_size/sizeGNN_classifier_313/difflinker_new_737/3/11_.xyz',
#        '/home/test2/george/DiffLinker/samples/test_30000_reindex/sampled_size/sizeGNN_classifier_313/difflinker_new_737/3/true_.xyz'
    #]
    sdf_oup_path = [
        'protac_0/protac_difflinker_998_N_-10_linker_size_15/5_7_0_.sdf'   

    ]
    xyz2sdf(xyz_inp_path,sdf_oup_path)
    # json2sdf('/home/test2/george/DiffLinker/ourdatasets/test_1500_modify.json')
    # json2sdf('/home/test2/george/DiffLinker/ourdatasets/train_all_modify.json')
    # check_bond('/home/test2/george/DiffLinker/datasets/test_30000.pt')
