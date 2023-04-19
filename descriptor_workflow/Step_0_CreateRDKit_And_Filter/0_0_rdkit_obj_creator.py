import pandas as pd 
import numpy as np
from pprint import pprint
import os
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.PropertyMol import PropertyMol
from atom_class_test_p3 import atom_class
import pickle
from datetime import date

today = date.today()

dt = today.strftime("%m_%d_%Y")

def sort_ids(s: str):
    '''This will correctly sort any type of reaction IDs'''
    _, b = s.split('_')
    return int(b)

def visualize_mols(name, mol_list, highlight_alkene=False):

    obj_name = name
    if len(mol_list) != 0:
        alkene_highlight_atom_list = list()
        alkene_highlight_bond_list = list()
        if highlight_alkene:
            for mol_object in mol_list:
                mol = atom_class(mol_object)
                alkene_boolean = np.array([True if v == '1' else False for v in mol.mol.GetProp("_Alkene")])
                isolated_carbon_atoms_idx = [int(i) for i in np.where(alkene_boolean)[0]]
                alkene_highlight_atom_list.append(isolated_carbon_atoms_idx)
                #This only highlights one bond bc i'm too lazy to change the script
                alkene_highlight_bond_list.append([mol.mol.GetBondBetweenAtoms(isolated_carbon_atoms_idx[0],isolated_carbon_atoms_idx[1]).GetIdx()])
        else: 
            alkene_highlight_atom_list = None
            alkene_highlight_bond_list = None

        _img = Draw.MolsToGridImage(mol_list, molsPerRow=5, subImgSize=(400,400), useSVG=True,returnPNG=False, highlightAtomLists=alkene_highlight_atom_list, highlightBondLists=alkene_highlight_bond_list,  legends=[i.GetProp("_Name") for i in mol_list], maxMols=20000)
        with open(f'{obj_name}.svg', 'w') as f:
            f.write(_img.data)
    return f'{obj_name}.png'

full_df = pd.read_csv('desc_matrix_creation_tools/p8_reduced_database_column_update_1001.csv')

# print(full_df)
# raise ValueError()

#This re-orders the dataframe based on the correct title of the reactant name, and then resets the index to make it simple to write an ordered dictionary
react_argsort = np.vectorize(sort_ids)(full_df['Reactant ID']).argsort()
sort_react_df = full_df.iloc[react_argsort]
sort_react_df = sort_react_df.reset_index(drop=True)

#This dictionary MUST BE ORDERED TO CORRECTLY NAME
react_map = {sort_react_df['Reactant ID'][i] : sort_react_df['Reactant SMILES'][i] for i in sort_react_df.index}
#This gives a list of each Reactant ID in the order of the dictionary react_map
react_name = [i for i in react_map]

prod_map = {sort_react_df['Product ID'][i] : sort_react_df['Product SMILES'][i] for i in sort_react_df.index}
prod_name = [i for i in prod_map]

print(prod_map)
react_mol_list = list()
prod_mol_list = list()

for i, react_id in enumerate(react_map):
    mol_object = PropertyMol(Chem.MolFromSmiles(react_map[react_id]))
    mol_object.SetProp('_Name', f'{react_id}')
    react_mol_list.append(mol_object)  

for i, prod_id in enumerate(prod_map):
    mol_object = PropertyMol(Chem.MolFromSmiles(prod_map[prod_id]))
    mol_object.SetProp('_Name', f'{prod_id}')
    prod_mol_list.append(mol_object)  

print(len(react_mol_list))
print(len(prod_mol_list))

with open('SAD_774_reactants_1000_entries.pkl', 'wb') as f:
    pickle.dump(react_mol_list, f)
with open('SAD_979_products_1000_entries.pkl', 'wb') as f:
    pickle.dump(prod_mol_list, f)
