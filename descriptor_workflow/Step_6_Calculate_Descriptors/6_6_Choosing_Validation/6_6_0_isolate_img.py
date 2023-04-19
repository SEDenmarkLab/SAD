import pandas as pd
import numpy as np
from pprint import pprint
import os
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.PropertyMol import PropertyMol
import pickle
from datetime import date
import _alkene_type_filter as atf

today = date.today()

dt = today.strftime("%m_%d_%Y")

def sort_ids(s: str):
    '''This will correctly sort any type of reaction IDs'''
    _, b = s.split('_')
    return int(b)

def visualize_mols(name, mol_list):

    obj_name = name
    if len(mol_list) != 0:
        _img = Draw.MolsToGridImage(mol_list, molsPerRow=5, subImgSize=(400,400), useSVG=True,returnPNG=False, highlightAtomLists=None, highlightBondLists=None,  legends=[i.GetProp("_Name") for i in mol_list], maxMols=20000)
        with open(f'{obj_name}.svg', 'w') as f:
            f.write(_img.data)

def give_unique_react(df: pd.DataFrame):
    df_iso = df[['Reactant ID', 'Reactant SMILES']]
    return df_iso.drop_duplicates()

def create_mol_list(react_df: pd.DataFrame):
    id = react_df['Reactant ID'].values
    smiles_str = react_df['Reactant SMILES'].values

    mol_list = list()

    for i, smiles in zip(id, smiles_str):
        mol = Chem.MolFromSmiles(smiles)
        mol.SetProp("_Name", i)
        mol_list.append(mol)

    return mol_list


full_df = pd.read_csv('p8_reduced_database_column_update_1001.csv')

mono = full_df.loc[atf.find_mono(full_df)]
mono_iso = give_unique_react(mono)
mono_mol_list = create_mol_list(mono_iso)
visualize_mols('0_mono', mono_mol_list)

gem_di = full_df.loc[atf.find_gem_di(full_df)]
gem_di_iso = give_unique_react(gem_di)
gem_di_mol_list = create_mol_list(gem_di_iso)
visualize_mols('1_gem_di', gem_di_mol_list)

cis_di = full_df.loc[atf.find_cis_di(full_df)]
cis_di_iso = give_unique_react(cis_di)
cis_di_mol_list = create_mol_list(cis_di_iso)
visualize_mols('2_cis_di', cis_di_mol_list)

trans_di = full_df.loc[atf.find_trans_di(full_df)]
trans_di_iso = give_unique_react(trans_di)
trans_di_mol_list = create_mol_list(trans_di_iso)
visualize_mols('3_trans_di', trans_di_mol_list)

tri = full_df.loc[atf.find_tri(full_df)]
tri_iso = give_unique_react(tri)
tri_mol_list = create_mol_list(tri_iso)
visualize_mols('4_tri', tri_mol_list)

tetra = full_df.loc[atf.find_tetra(full_df)]
tetra_iso = give_unique_react(tetra)
tetra_mol_list = create_mol_list(tetra_iso)
visualize_mols('5_tetra', tetra_mol_list)