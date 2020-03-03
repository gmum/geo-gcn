import os
import logging

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles
from torch_geometric.data import Data


def get_edge_indices(adj):
    edges_list = []
    for i in range(adj.shape[0]):
        for j in range(i, adj.shape[0]):
            if adj[i, j] == 1:
                edges_list.append((i, j))
    return edges_list


def transform_molecule_pg(mol):
    afm, adj, positions, label = mol

    x = torch.tensor(afm)
    y = torch.tensor(label)
    edge_index = torch.tensor(get_edge_indices(adj)).t().contiguous()
    pos = torch.tensor(positions)

    return Data(x=x, y=y, edge_index=edge_index, pos=pos)


def transform_dataset_pg(dataset):
    dataset_pg = []

    for mol in dataset:
        dataset_pg.append(transform_molecule_pg(mol))

    return dataset_pg


def load_dataset(dataset_name, fold_name, path='../data/molecules'):
    filename = dataset_name.lower() + '_' + fold_name + '.csv'
    filepath = os.path.join(path, filename)
    x, y = load_data_from_df(filepath)
    return transform_dataset_pg([[*i, j] for i, j in zip(x, y)])


def load_data_from_df(dataset_path):
    data_df = pd.read_csv(dataset_path)

    data_x = data_df.iloc[:, 0].values
    data_y = data_df.iloc[:, 1].values

    if data_y.dtype == np.float64:
        data_y = data_y.astype(np.float32)

    x_all, y_all = load_data_from_smiles(data_x, data_y)
    return x_all, y_all


def load_data_from_smiles(x_smiles, labels, normalize_features=False):
    x_all, y_all = [], []
    for smiles, label in zip(x_smiles, labels):
        try:
            if len(smiles) < 2:
                raise ValueError

            mol = MolFromSmiles(smiles)

            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            AllChem.UFFOptimizeMolecule(mol)
            mol = Chem.RemoveHs(mol)

            afm, adj, mat_positions = featurize_mol(mol)
            x_all.append([afm, adj, mat_positions])
            y_all.append([label])
        except ValueError as e:
            logging.warning('the SMILES ({}) can not be converted to a graph.\nREASON: {}'.format(smiles, e))

    if normalize_features:
        x_all = feature_normalize(x_all)
    return x_all, y_all


def featurize_mol(mol):
    conf = mol.GetConformer()
    node_features = np.array([get_atom_features(atom)
                              for atom in mol.GetAtoms()])
    adj_matrix = np.eye(mol.GetNumAtoms())
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtom().GetIdx()
        end_atom = bond.GetEndAtom().GetIdx()
        adj_matrix[begin_atom, end_atom] = adj_matrix[end_atom, begin_atom] = 1
    pos_matrix = np.array([[conf.GetAtomPosition(k).x, conf.GetAtomPosition(k).y, conf.GetAtomPosition(k).z]
                           for k in range(mol.GetNumAtoms())])
    return node_features, adj_matrix, pos_matrix


def get_atom_features(atom):
    attributes = []

    attributes += one_hot_vector(
        atom.GetAtomicNum(),
        [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999]
    )

    attributes += one_hot_vector(
        len(atom.GetNeighbors()),
        [0, 1, 2, 3, 4, 5]
    )

    attributes += one_hot_vector(
        atom.GetTotalNumHs(),
        [0, 1, 2, 3, 4]
    )

    attributes.append(atom.GetFormalCharge())
    attributes.append(atom.IsInRing())
    attributes.append(atom.GetIsAromatic())

    return np.array(attributes, dtype=np.float32)


def one_hot_vector(val, lst):
    """Converts a value to a one-hot vector based on options in lst"""
    if val not in lst:
        val = lst[-1]
    return map(lambda x: x == val, lst)


def feature_normalize(x_all):
    """Min Max Feature Scalling for Atom Feature Matrix"""
    min_vec, max_vec = x_all[0][0].min(axis=0), x_all[0][0].max(axis=0)
    for x in x_all:
        min_vec = np.minimum(min_vec, x[0].min(axis=0))
        max_vec = np.maximum(max_vec, x[0].max(axis=0))
    diff = max_vec - min_vec
    diff[diff == 0] = 1.

    for x in x_all:
        afm = x[0]
        afm = (afm - min_vec) / diff
        x[0] = afm

    return x_all
