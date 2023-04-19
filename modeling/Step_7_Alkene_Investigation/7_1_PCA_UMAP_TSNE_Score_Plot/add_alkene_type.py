import numpy as np
import pandas as pd
from sklearn import feature_selection as fs
from sklearn import preprocessing as pp
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn import decomposition as dr
from sklearn import metrics
from scipy.spatial.distance import cdist
import pickle
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.PropertyMol import PropertyMol
import os
from sklearn.pipeline import Pipeline
from glob import glob
from bidict import bidict
import warnings
from pprint import pprint

def create_mol_list(react_df: pd.DataFrame):
    id = react_df['Reactant ID'].values
    smiles_str = react_df['Reactant SMILES'].values
    alkene_type = react_df['Olefin Type'].values


    mol_list = list()

    for i, smiles, _type in zip(id, smiles_str, alkene_type):
        if i == 'react_56':
            mol = Chem.MolFromSmiles(smiles)
            mol.SetProp("_Name", i)
            mol.SetProp("_Alkene_Type", _type)
            mol_list.append(mol)

    return mol_list


def give_unique_react(df: pd.DataFrame):
    df_iso = df[['Reactant ID', 'Reactant SMILES', 'Olefin Type']]
    return df_iso.drop_duplicates()

db_df = pd.read_csv('p8_reduced_database_column_update_1001.csv')
alkene_iso = give_unique_react(db_df)
print(alkene_iso)
can_alkene_smiles = alkene_iso['Reactant SMILES'].values
alkene_mol_list = create_mol_list(alkene_iso)
alkene_mol_dict = bidict({mol.GetProp("_Name") : mol for mol in alkene_mol_list})
alkene_mol_type_dict = dict()
for mol in alkene_mol_list:

    alkene_mol_type_dict[mol.GetProp("_Name")] = mol.GetProp("_Alkene_Type")
    # if mol.GetProp("_Name") == 'react_56':
    #     print(mol.GetProp("_Alkene_Type"))
    #     raise ValueError()

    if mol.GetProp("_Name") == 'react_56':
        print(mol.GetProp("_Name"))
        print(mol.GetProp("_Alkene_Type"))
# alkene_mol_type_dict = {mol.GetProp("_Name") : mol.GetProp("_Alkene_Type") for mol in alkene_mol_list}

pprint(alkene_mol_type_dict)
raise ValueError()

full_df = pd.read_csv('Step_6_Full_Aligned_React_Desc_DF.csv', index_col=0)

new_alkene_type_ser = pd.Series(alkene_mol_type_dict, name='Olefin Type')
print(new_alkene_type_ser)
new_df = pd.concat([full_df, new_alkene_type_ser], axis=1)
print(new_df)
final_df = new_df.dropna()
print(final_df)
final_df.to_csv('Step_6_Full_Aligned_React_Desc_DF_w_Olefin_Type.csv')

# # print(new_df)
