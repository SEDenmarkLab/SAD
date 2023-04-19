import pandas as pd 
import numpy as np
from pprint import pprint
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.PropertyMol import PropertyMol
from atom_class_test_p3 import atom_class
import pickle

with open('SAD_Alkene_p0_react_multiple_alkenes_03_01_2023.pkl', 'rb') as f:
    mult_alk = pickle.load(f)

not_connected_alkenes = list()
problematic_alkenes = list()
final_mol_object_list = list()

for react_mol in mult_alk:
    mol = atom_class(react_mol)
    mol_bool = np.full((len(react_mol.GetAtoms()),), fill_value=False)
    print(f'For reactant: {react_mol.GetProp("_Name")}')
    c1 = input('What is the value of the first carbon? ')
    c2 = input('What is the value of the second carbon? ')

    c1 = eval(c1)-1
    c2 = eval(c2)-1
    mol_bool[c1] = True
    mol_bool[c2] = True
    react_mol.SetProp("_Alkene", "".join('1' if v else '0' for v in mol_bool))
    original_array = np.array([True if v == '1' else False for v in react_mol.GetProp("_Alkene")])

    if all(original_array == mol_bool):
        #This tests to make sure alkenes are connected

        #This returns atom indices where bool array is true
        isolated_carbon_idx_np = mol.atoms_array[mol_bool]

        #This returns dictionary of atom index : atom object for the indices where the bool array was true
        isolated_carbon_atoms = [mol.atoms_dict[i] for i in np.where(mol_bool)[0]]

        carbon1_neighbor_atom_idx = list()
        carbon1 = isolated_carbon_atoms[0]
        carbon2 = isolated_carbon_atoms[1]

        carbon1_neighbor_atoms = carbon1.GetNeighbors()

        for neighbor in carbon1_neighbor_atoms:
            neighbor_idx = neighbor.GetIdx()
            carbon1_neighbor_atom_idx.append(neighbor_idx)

        if carbon2.GetIdx() in carbon1_neighbor_atom_idx:
            final_mol_object_list.append(react_mol)
        else:
            print(f'{react_mol.GetProp("_Name")} do not have carbons connecting, appended to problematic mol object list')
            problematic_alkenes.append(react_mol)
            not_connected_alkenes.append(react_mol)
    else:
        print(f'{react_mol.GetProp("_Name")} did not correctly return alkene boolean, appended to problematic mol object list')
        problematic_alkenes.append(react_mol)
    assert react_mol.HasProp("_Alkene"), f"react_mol doesn't have property"

with open('SAD_Alkene_p0_react_03_01_2023.pkl', 'rb') as f:
    original_isolated_alkene_list = pickle.load(f)

print(len(original_isolated_alkene_list))
original_isolated_alkene_list.extend(final_mol_object_list)
print(len(original_isolated_alkene_list))

with open('3_1_2023_SAD_Step_0_Reactants_3_01_2023.pkl', 'wb') as f:
    pickle.dump(original_isolated_alkene_list, f)