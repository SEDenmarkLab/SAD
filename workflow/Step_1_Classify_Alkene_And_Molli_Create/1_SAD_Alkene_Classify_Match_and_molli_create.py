import pandas as pd
import numpy as np
import pickle
from datetime import date
import yaml
from rdkit import Chem
from rdkit.Chem import AllChem as ac
from openbabel import openbabel as ob
import molli as ml

'''
This script uses the input csv file to help match the alkene type to the correct molecule object.
'''

today = date.today()

dt = today.strftime("%m_%d_%Y")

df = pd.read_csv('p8_reduced_database_column_update_1001.csv')

with open('3_1_2023_SAD_Step_0_Reactants_3_01_2023.pkl', 'rb') as f:
    react_mols = pickle.load(f)

for mol in react_mols:
    current_name = mol.GetProp("_Name")
    olefin_type_arr = df.query('`Reactant ID` == @current_name')['Olefin Type'].to_numpy()
    if olefin_type_arr.shape != (1,):
        assert all([True for i in olefin_type_arr if i == olefin_type_arr[0]]), 'rip'

    olefin_type = olefin_type_arr[0]
    mol.SetProp("_Alkene_Type",f'{olefin_type}')

with open(f'SAD_Alkene_Step_1_Reactants_w_Type_{dt}.pkl', 'wb') as f:
    pickle.dump(react_mols, f)

molli_mols = list()

for rdkit_mol in react_mols:
    name = rdkit_mol.GetProp("_Name")
    rdkit_mol_w_h = Chem.AddHs(rdkit_mol)
    ac.EmbedMolecule(rdkit_mol_w_h)
    ac.MMFFOptimizeMolecule(rdkit_mol_w_h)

    conv = ob.OBConversion()
    conv.SetInAndOutFormats("mol", 'mol2')
    obmol = ob.OBMol()
    conv.ReadString(obmol, Chem.MolToMolBlock(rdkit_mol_w_h))
    
    molli_mols.append(ml.Molecule.from_mol2(conv.WriteString(obmol)))

col = ml.Collection(f'SAD_Alkene_Step_1_Reactants_w_Type_{dt}_MMFF_unordered', molli_mols)

col.to_zip(f'{col.name}.zip')
