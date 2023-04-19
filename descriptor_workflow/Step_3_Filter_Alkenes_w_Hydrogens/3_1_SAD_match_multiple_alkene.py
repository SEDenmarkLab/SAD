import pandas as pd 
import numpy as np
from pprint import pprint
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.PropertyMol import PropertyMol
from atom_class_test_p3 import atom_class
import pickle
from matplotlib import pyplot as plt
from PIL import Image

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

        _img = Draw.MolsToGridImage(mol_list, molsPerRow=2, subImgSize=(400,400), useSVG=True,returnPNG=False, highlightAtomLists=alkene_highlight_atom_list, highlightBondLists=alkene_highlight_bond_list,  legends=[i.GetProp("_Name") for i in mol_list], maxMols=20000)
        with open(f'{obj_name}.svg', 'w') as f:
            f.write(_img.data)
    return f'{obj_name}.png'

full_df = pd.read_csv('p8_reduced_database_column_update_1001.csv')

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

react_prod_id_map = {sort_react_df['Reactant ID'][i]: sort_react_df['Product ID'][i] for i in sort_react_df.index}
# print(react_prod_id_map)
with open('SAD_Step_3_First_Alkene_Filter_multiple_alkenes_03_01_2023.pkl', 'rb') as f:
    mult_alk = pickle.load(f)

mult_alkene_visualize = list()

for react_mol in mult_alk:
    react_id = react_mol.GetProp("_Name")
    prod_id = react_prod_id_map[react_id]
    prod_mol = Chem.MolFromSmiles(prod_map[prod_id])
    prod_mol.SetProp("_Name", prod_id)
    
    for atom in react_mol.GetAtoms():
        atom.SetProp("molAtomMapNumber", str(atom.GetIdx() + 1))

    for atom in prod_mol.GetAtoms():
        atom.SetProp("molAtomMapNumber", str(atom.GetIdx() + 1))

    # not_connected_alkenes = list()
    # problematic_alkenes = list()
    # final_mol_object_list = list()

    # testing_name_list = [react_id, prod_id]
    # testing_mol_list = [react_mol, prod_mol]
    # part_address = visualize_mols('example',testing_mol_list)
    # image = pyvips.Image.new_from_file(part_address,memory=False, dpi=300)
    # image.write_to_file(part_address + '.jpg')
    # ImageAddress = '/mnt/e/Dihydroxylation Project/Daily-Work-Folder/3-1-23'
    # ImageItself = Image.open(ImageAddress)
    # ImageNumpyFormat = np.asarray(ImageItself)
    # plt.imshow(ImageNumpyFormat)
    # plt.draw()
    # plt.pause(100) # pause how many seconds
    # plt.close()

    # mol = atom_class(react_mol)
    # mol_bool = np.full((len(react_mol.GetAtoms()),), fill_value=False)
    # print(f'For reactant: {react_mol.GetProp("_Name")}')
    # c1 = input('What is the value of the first carbon? ')
    # c2 = input('What is the value of the second carbon? ')

    # c1 = eval(c1)-1
    # c2 = eval(c2)-1
    # mol_bool[c1] = True
    # mol_bool[c2] = True
    # react_mol.SetProp("_Alkene", "".join('1' if v else '0' for v in mol_bool))
    # original_array = np.array([True if v == '1' else False for v in react_mol.GetProp("_Alkene")])

    # if all(original_array == mol_bool):
    #     #This tests to make sure alkenes are connected

    #     #This returns atom indices where bool array is true
    #     isolated_carbon_idx_np = mol.atoms_array[mol_bool]

    #     #This returns dictionary of atom index : atom object for the indices where the bool array was true
    #     isolated_carbon_atoms = [mol.atoms_dict[i] for i in np.where(mol_bool)[0]]

    #     carbon1_neighbor_atom_idx = list()
    #     carbon1 = isolated_carbon_atoms[0]
    #     carbon2 = isolated_carbon_atoms[1]

    #     carbon1_neighbor_atoms = carbon1.GetNeighbors()

    #     for neighbor in carbon1_neighbor_atoms:
    #         neighbor_idx = neighbor.GetIdx()
    #         carbon1_neighbor_atom_idx.append(neighbor_idx)

    #     if carbon2.GetIdx() in carbon1_neighbor_atom_idx:
    #         final_mol_object_list.append(react_mol)
    #     else:
    #         print(f'{react_mol.GetProp("_Name")} do not have carbons connecting, appended to problematic mol object list')
    #         problematic_alkenes.append(react_mol)
    #         not_connected_alkenes.append(react_mol)
    # else:
    #     print(f'{react_mol.GetProp("_Name")} did not correctly return alkene boolean, appended to problematic mol object list')
    #     problematic_alkenes.append(react_mol)

    mult_alkene_visualize.extend([react_mol, prod_mol])

visualize_mols('multiple_alkenes_w_h', mult_alkene_visualize)

