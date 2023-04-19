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
                alkene_boolean = np.array([True if v == '1' else False for v in mol.mol.GetProp("_Alkene_w_H")])
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

final_mol_object_list = list()
removed_mol_names = list()
problematic_mol_object_list = list()
not_connected_alkenes = list()
multiple_alkenes = list()

#Check 0: All Clear! Remove structures with issues remaining in Full List
remain_mol_0 = list()

# reactant_issues_df = pd.read_excel('messed_up_reactants.xlsx')

# react_id_issues = reactant_issues_df['Reactant ID']

# for mol_object in mol_list:
#     mol = atom_class(mol_object)

#     if mol.mol.GetProp("_Name") in react_id_issues.values:
#         problematic_mol_object_list.append(mol_object)
#     else:
#         remain_mol_0.append(mol_object)

# print(f'Isolated {len(problematic_mol_object_list)} problematic/non-alkenes, {len(remain_mol_0)} remaining.')

# img_0_problematic_alkenes = Draw.MolsToGridImage(problematic_mol_object_list, molsPerRow=5, subImgSize=(100,100), useSVG=True, legends=[i.GetProp("_Name") for i in problematic_mol_object_list], maxMols=1000)
# open('image0_problematic_alkenes.svg', 'w').write(img_0_problematic_alkenes.data)
with open('SAD_Step_2_Canonical_RDKitMol_w_h_React_03_01_2023', 'rb') as f:
    remain_mol_0 = pickle.load(f)

#Check 1: All clear!
mol_object_list1 = list()
remain_mol_1 = list()

print('For Check #1')
for mol_object in remain_mol_0:
    mol_object.UpdatePropertyCache()
    Chem.SanitizeMol(mol_object)
    mol = atom_class(mol_object)
    mol_bool = mol.sp2_type() & mol.carbon_type()

    if np.count_nonzero(mol_bool) == 2:
        mol_object_list1.append(mol.mol)
        removed_mol_names.append(mol.mol.GetProp("_Name"))
        #Sets property "_Alkene_w_H" to be equal to numpy boolean array
        mol.mol.SetProp("_Alkene_w_H", "".join('1' if v else '0' for v in mol_bool))
        original_array = np.array([True if v == '1' else False for v in mol.mol.GetProp("_Alkene_w_H")])
        #Tests to make sure array property is set and returned
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
                final_mol_object_list.append(mol.mol)
            else:
                print(f'{mol_object.GetProp("_Name")} do not have carbons connecting, appended to problematic mol object list')
                problematic_mol_object_list.append(mol.mol)
                not_connected_alkenes.append(mol.mol)
                mol_object_list1.remove(mol.mol)
        else:
            print(f'{mol_object.GetProp("_Name")} did not correctly return alkene boolean, appended to problematic mol object list')
            mol_object_list1.remove(mol.mol)
            problematic_mol_object_list.append(mol.mol)
            
    else:
        remain_mol_1.append(mol.mol)

print(f'Isolated {len(mol_object_list1)} alkenes, {len(remain_mol_1)} remaining.')

visualize_mols(f'Filter1_{dt}',mol_object_list1,highlight_alkene=True)

# Check 2: All Clear!
mol_object_list2 = list()
remain_mol_2 = list()

print('For Check #2')
for mol_object in remain_mol_1:
    mol = atom_class(mol_object)
    
    mol_bool = mol.sp2_type() & mol.carbon_type() & mol.in_1_ring() & mol.het_neighbors_0() & ~mol.aromatic_type()
    if np.count_nonzero(mol_bool) == 2:
        mol_object_list2.append(mol.mol)
        removed_mol_names.append(mol.mol.GetProp("_Name"))
        #Sets property "_Alkene_w_H" to be equal to numpy boolean array
        mol.mol.SetProp("_Alkene_w_H", "".join('1' if v else '0' for v in mol_bool))
        original_array = np.array([True if v == '1' else False for v in mol.mol.GetProp("_Alkene_w_H")])
        #Tests to make sure array property is set and returned
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
                final_mol_object_list.append(mol.mol)
            else:
                print(f'{mol.mol.GetProp("_Name")} do not have carbons connecting, appended to problematic mol object list')
                problematic_mol_object_list.append(mol.mol)
                not_connected_alkenes.append(mol.mol)
                mol_object_list2.remove(mol.mol)
        else:
            print(f'{mol.mol.GetProp("_Name")} did not correctly return alkene boolean, appended to problematic mol object list')
            mol_object_list2.remove(mol.mol)
            problematic_mol_object_list.append(mol.mol)
            
    else:
        remain_mol_2.append(mol.mol)

print(f'Isolated {len(mol_object_list2)} alkenes, {len(remain_mol_2)} remaining.')

visualize_mols(f'Filter2_{dt}',mol_object_list2,highlight_alkene=True)


#Check 3: All Clear
mol_object_list3 = list()
remain_mol_3 = list()

print('For Check #3')
for mol_object in remain_mol_2:
    mol = atom_class(mol_object)
    
    mol_bool = mol.sp2_type() & mol.carbon_type() & mol.het_neighbors_0() & ~mol.aromatic_type()

    if np.count_nonzero(mol_bool) == 2:
        mol_object_list3.append(mol.mol)
        removed_mol_names.append(mol.mol.GetProp("_Name"))
        #Sets property "_Alkene_w_H" to be equal to numpy boolean array
        mol.mol.SetProp("_Alkene_w_H", "".join('1' if v else '0' for v in mol_bool))
        original_array = np.array([True if v == '1' else False for v in mol.mol.GetProp("_Alkene_w_H")])
        #Tests to make sure array property is set and returned
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
                final_mol_object_list.append(mol.mol)
            else:
                print(f'{mol.mol.GetProp("_Name")} do not have carbons connecting, appended to problematic mol object list')
                problematic_mol_object_list.append(mol.mol)
                not_connected_alkenes.append(mol.mol)
                mol_object_list3.remove(mol.mol)
        else:
            print(f'{mol.mol.GetProp("_Name")} did not correctly return alkene boolean, appended to problematic mol object list')
            mol_object_list3.remove(mol.mol)
            problematic_mol_object_list.append(mol.mol)
    else:
        remain_mol_3.append(mol.mol)
print(f'Isolated {len(mol_object_list3)} alkenes, {len(remain_mol_3)} remaining.')

visualize_mols(f'Filter3_{dt}',mol_object_list3,highlight_alkene=True)


#Check 4: All clear!
mol_object_list4 = list()
remain_mol_4 = list()

print('For Check #4')
for mol_object in remain_mol_3:
    mol = atom_class(mol_object)
    
    mol_bool = mol.sp2_type() & mol.carbon_type() & mol.het_neighbors_1() & ~mol.in_1_ring() & ~mol.aromatic_type() & ~mol.smarts_query('[$([CX3]=[OX1]),$([CX3+]-[OX1-])]')

    if np.count_nonzero(mol_bool) == 2:
        mol_object_list4.append(mol.mol)
        removed_mol_names.append(mol.mol.GetProp("_Name"))
        #Sets property "_Alkene_w_H" to be equal to numpy boolean array
        mol.mol.SetProp("_Alkene_w_H", "".join('1' if v else '0' for v in mol_bool))
        original_array = np.array([True if v == '1' else False for v in mol.mol.GetProp("_Alkene_w_H")])
        #Tests to make sure array property is set and returned
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
                final_mol_object_list.append(mol.mol)
            else:
                print(f'{mol.mol.GetProp("_Name")} do not have carbons connecting, appended to problematic mol object list')
                problematic_mol_object_list.append(mol.mol)
                not_connected_alkenes.append(mol.mol)
                mol_object_list4.remove(mol.mol)
        else:
            print(f'{mol.mol.GetProp("_Name")} did not correctly return alkene boolean, appended to problematic mol object list')
            mol_object_list4.remove(mol.mol)
            problematic_mol_object_list.append(mol.mol)
    else:
        remain_mol_4.append(mol.mol)
print(f'Isolated {len(mol_object_list4)} alkenes, {len(remain_mol_4)} remaining.')

visualize_mols(f'Filter4_{dt}',mol_object_list4,highlight_alkene=True)

#Check 5: All clear!
mol_object_list5 = list()
remain_mol_5 = list()

print('For Check #5')
for mol_object in remain_mol_4:
    mol = atom_class(mol_object)
    
    mol_bool = mol.sp2_type() & mol.carbon_type() & mol.het_neighbors_1() & mol.in_1_ring() & ~mol.aromatic_type() & ~mol.smarts_query('[$([CX3]=[OX1]),$([CX3+]-[OX1-])]')

    if np.count_nonzero(mol_bool) == 2:
        mol_object_list5.append(mol.mol)
        removed_mol_names.append(mol.mol.GetProp("_Name"))
        #Sets property "_Alkene_w_H" to be equal to numpy boolean array
        mol.mol.SetProp("_Alkene_w_H", "".join('1' if v else '0' for v in mol_bool))
        original_array = np.array([True if v == '1' else False for v in mol.mol.GetProp("_Alkene_w_H")])
        #Tests to make sure array property is set and returned
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
                final_mol_object_list.append(mol.mol)
            else:
                print(f'{mol.mol.GetProp("_Name")} do not have carbons connecting, appended to problematic mol object list')
                problematic_mol_object_list.append(mol.mol)
                not_connected_alkenes.append(mol.mol)
                mol_object_list5.remove(mol.mol)
        else:
            print(f'{mol.mol.GetProp("_Name")} did not correctly return alkene boolean, appended to problematic mol object list')
            mol_object_list5.remove(mol.mol)
            problematic_mol_object_list.append(mol.mol)
    else:
        remain_mol_5.append(mol.mol)
print(f'Isolated {len(mol_object_list5)} alkenes, {len(remain_mol_5)} remaining.')

visualize_mols(f'Filter5_{dt}',mol_object_list5,highlight_alkene=True)

#Check 6: All clear!
mol_object_list6 = list()
remain_mol_6 = list()

print('For Check #6')
for mol_object in remain_mol_5:
    mol = atom_class(mol_object)
    
    mol_bool = mol.sp2_type() & mol.carbon_type() & ~mol.het_neighbors_3() & mol.in_1_ring() & ~mol.aromatic_type() & ~mol.smarts_query('[$([CX3]=[OX1]),$([CX3+]-[OX1-])]')

    if np.count_nonzero(mol_bool) == 2:
        mol_object_list6.append(mol.mol)
        removed_mol_names.append(mol.mol.GetProp("_Name"))
        #Sets property "_Alkene_w_H" to be equal to numpy boolean array
        mol.mol.SetProp("_Alkene_w_H", "".join('1' if v else '0' for v in mol_bool))
        original_array = np.array([True if v == '1' else False for v in mol.mol.GetProp("_Alkene_w_H")])
        #Tests to make sure array property is set and returned
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
                final_mol_object_list.append(mol.mol)
            else:
                print(f'{mol.mol.GetProp("_Name")} do not have carbons connecting, appended to problematic mol object list')
                problematic_mol_object_list.append(mol.mol)
                not_connected_alkenes.append(mol.mol)
                mol_object_list6.remove(mol.mol)
        else:
            print(f'{mol.mol.GetProp("_Name")} did not correctly return alkene boolean, appended to problematic mol object list')
            mol_object_list6.remove(mol.mol)
            problematic_mol_object_list.append(mol.mol)
    else:
        remain_mol_6.append(mol.mol)
print(f'Isolated {len(mol_object_list6)} alkenes, {len(remain_mol_6)} remaining.')

visualize_mols(f'Filter6_{dt}',mol_object_list6,highlight_alkene=True)

#Check 7: All clear!
mol_object_list7 = list()
remain_mol_7 = list()

print('For Check #7')
for mol_object in remain_mol_6:
    mol = atom_class(mol_object)
    
    mol_bool = mol.sp2_type() & mol.carbon_type() & ~mol.het_neighbors_3() & ~mol.in_2_rings() & ~mol.aromatic_type() & ~mol.smarts_query('[$([CX3]=[OX1]),$([CX3+]-[OX1-])]')

    if np.count_nonzero(mol_bool) == 2:
        mol_object_list7.append(mol.mol)
        removed_mol_names.append(mol.mol.GetProp("_Name"))
        #Sets property "_Alkene_w_H" to be equal to numpy boolean array
        mol.mol.SetProp("_Alkene_w_H", "".join('1' if v else '0' for v in mol_bool))
        original_array = np.array([True if v == '1' else False for v in mol.mol.GetProp("_Alkene_w_H")])
        #Tests to make sure array property is set and returned
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
                final_mol_object_list.append(mol.mol)
            else:
                print(f'{mol.mol.GetProp("_Name")} do not have carbons connecting, appended to problematic mol object list')
                problematic_mol_object_list.append(mol.mol)
                not_connected_alkenes.append(mol.mol)
                mol_object_list7.remove(mol.mol)
        else:
            print(f'{mol.mol.GetProp("_Name")} did not correctly return alkene boolean, appended to problematic mol object list')
            mol_object_list7.remove(mol.mol)
            problematic_mol_object_list.append(mol.mol)
    else:
        remain_mol_7.append(mol.mol)
print(f'Isolated {len(mol_object_list7)} alkenes, {len(remain_mol_7)} remaining.')

visualize_mols(f'Filter7_{dt}',mol_object_list7,highlight_alkene=True)

#Check 8: All clear!
mol_object_list8 = list()
remain_mol_8 = list()

print('For Check #8')
for mol_object in remain_mol_7:
    mol = atom_class(mol_object)
    
    mol_bool = mol.sp2_type() & mol.carbon_type() & ~mol.in_2_rings() & ~mol.het_neighbors_3() & mol.aromatic_type() & mol.ring_size5() & ~mol.smarts_query('[$([CX3]=[OX1]),$([CX3+]-[OX1-])]') & ~mol.smarts_query('c1cscn1')

    if np.count_nonzero(mol_bool) == 2:
        mol_object_list8.append(mol.mol)
        removed_mol_names.append(mol.mol.GetProp("_Name"))
        #Sets property "_Alkene_w_H" to be equal to numpy boolean array
        mol.mol.SetProp("_Alkene_w_H", "".join('1' if v else '0' for v in mol_bool))
        original_array = np.array([True if v == '1' else False for v in mol.mol.GetProp("_Alkene_w_H")])
        #Tests to make sure array property is set and returned
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
                final_mol_object_list.append(mol.mol)
            else:
                print(f'{mol.mol.GetProp("_Name")} do not have carbons connecting, appended to problematic mol object list')
                problematic_mol_object_list.append(mol.mol)
                not_connected_alkenes.append(mol.mol)
                mol_object_list8.remove(mol.mol)
        else:
            print(f'{mol.mol.GetProp("_Name")} did not correctly return alkene boolean, appended to problematic mol object list')
            mol_object_list8.remove(mol.mol)
            problematic_mol_object_list.append(mol.mol)
    else:
        remain_mol_8.append(mol.mol)
print(f'Isolated {len(mol_object_list8)} alkenes, {len(remain_mol_8)} remaining.')

visualize_mols(f'Filter8_{dt}',mol_object_list8,highlight_alkene=True)

#Check 9: All Clear!
mol_object_list9 = list()
remain_mol_9 = list()

print('For Check #9')
for mol_object in remain_mol_8:
    mol = atom_class(mol_object)
    
    mol_bool = mol.sp2_type() & mol.carbon_type() & ~mol.in_2_rings() & ~mol.het_neighbors_3() & mol.smarts_query('[NX3][CX3]=[CX3]') & ~mol.smarts_query('[$([CX3]=[OX1]),$([CX3+]-[OX1-])]')

    if np.count_nonzero(mol_bool) == 2:
        mol_object_list9.append(mol.mol)
        removed_mol_names.append(mol.mol.GetProp("_Name"))
        #Sets property "_Alkene_w_H" to be equal to numpy boolean array
        mol.mol.SetProp("_Alkene_w_H", "".join('1' if v else '0' for v in mol_bool))
        original_array = np.array([True if v == '1' else False for v in mol.mol.GetProp("_Alkene_w_H")])
        #Tests to make sure array property is set and returned
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
                final_mol_object_list.append(mol.mol)
            else:
                print(f'{mol.mol.GetProp("_Name")} do not have carbons connecting, appended to problematic mol object list')
                problematic_mol_object_list.append(mol.mol)
                not_connected_alkenes.append(mol.mol)
                mol_object_list9.remove(mol.mol)
        else:
            print(f'{mol.mol.GetProp("_Name")} did not correctly return alkene boolean, appended to problematic mol object list')
            mol_object_list9.remove(mol.mol)
            problematic_mol_object_list.append(mol.mol)
    else:
        remain_mol_9.append(mol.mol)
print(f'Isolated {len(mol_object_list9)} alkenes, {len(remain_mol_9)} remaining.')

visualize_mols(f'Filter9_{dt}',mol_object_list9,highlight_alkene=True)

#Check 10: All Clear! Grabs any remaining indole type system
mol_object_list10 = list()
remain_mol_10 = list()

print('For Check #10')
for mol_object in remain_mol_9:
    mol = atom_class(mol_object)
    
    mol_bool = mol.sp2_type() & mol.carbon_type() & mol.smarts_query('[nX3H][cX3]([CX4])[cX3]([CX4])')

    if np.count_nonzero(mol_bool) == 2:
        mol_object_list10.append(mol.mol)
        removed_mol_names.append(mol.mol.GetProp("_Name"))
        #Sets property "_Alkene_w_H" to be equal to numpy boolean array
        mol.mol.SetProp("_Alkene_w_H", "".join('1' if v else '0' for v in mol_bool))
        original_array = np.array([True if v == '1' else False for v in mol.mol.GetProp("_Alkene_w_H")])
        #Tests to make sure array property is set and returned
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
                final_mol_object_list.append(mol.mol)
            else:
                print(f'{mol.mol.GetProp("_Name")} do not have carbons connecting, appended to problematic mol object list')
                problematic_mol_object_list.append(mol.mol)
                not_connected_alkenes.append(mol.mol)
                mol_object_list10.remove(mol.mol)
        else:
            print(f'{mol.mol.GetProp("_Name")} did not correctly return alkene boolean, appended to problematic mol object list')
            mol_object_list10.remove(mol.mol)
            problematic_mol_object_list.append(mol.mol)
    else:
        remain_mol_10.append(mol.mol)
print(f'Isolated {len(mol_object_list10)} alkenes, {len(remain_mol_10)} remaining.')

visualize_mols(f'Filter10_{dt}',mol_object_list10,highlight_alkene=True)

#Check 11: All Clear! Grabs any homo-allylic alcohol
mol_object_list11 = list()
remain_mol_11 = list()

print('For Check #11')
for mol_object in remain_mol_10:
    #This is a hard code as this is the only one outstanding:
    if mol_object.GetProp("_Name") == 'react_620':
        remain_mol_11.append(mol_object)
        continue

    mol = atom_class(mol_object)
    
    mol_bool = (mol.sp2_type() & mol.carbon_type() & mol.smarts_query('[CX3]=[CX3][CX4][OH]') & ~mol.smarts_query('C=CC=C'))
    if np.count_nonzero(mol_bool) == 2:
        mol_object_list11.append(mol_object)
        removed_mol_names.append(mol_object.GetProp("_Name"))
        #Sets property "_Alkene_w_H" to be equal to numpy boolean array
        mol.mol.SetProp("_Alkene_w_H", "".join('1' if v else '0' for v in mol_bool))
        original_array = np.array([True if v == '1' else False for v in mol.mol.GetProp("_Alkene_w_H")])
        #Tests to make sure array property is set and returned
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
                final_mol_object_list.append(mol.mol)
            else:
                print(f'{mol.mol.GetProp("_Name")} do not have carbons connecting, appended to problematic mol object list')
                problematic_mol_object_list.append(mol.mol)
                not_connected_alkenes.append(mol.mol)
                mol_object_list11.remove(mol.mol)
        else:
            print(f'{mol.mol.GetProp("_Name")} did not correctly return alkene boolean, appended to problematic mol object list')
            mol_object_list11.remove(mol.mol)
            problematic_mol_object_list.append(mol.mol)
    else:
        remain_mol_11.append(mol.mol)
print(f'Isolated {len(mol_object_list11)} alkenes, {len(remain_mol_11)} remaining.')

visualize_mols(f'Filter11_{dt}',mol_object_list11,highlight_alkene=True)

#Check 12: All clear! Grabs any oxazolidinone alkene
mol_object_list12 = list()
remain_mol_12 = list()

print('For Check #12')
for mol_object in remain_mol_11:
    mol = atom_class(mol_object)
    
    mol_bool = mol.sp2_type() & mol.carbon_type() & ~mol.het_neighbors_3() & mol.smarts_query('c1[nH]c(=O)oc1')

    if np.count_nonzero(mol_bool) == 2:
        mol_object_list12.append(mol.mol)
        removed_mol_names.append(mol.mol.GetProp("_Name"))
        #Sets property "_Alkene_w_H" to be equal to numpy boolean array
        mol.mol.SetProp("_Alkene_w_H", "".join('1' if v else '0' for v in mol_bool))
        original_array = np.array([True if v == '1' else False for v in mol.mol.GetProp("_Alkene_w_H")])
        #Tests to make sure array property is set and returned
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
                final_mol_object_list.append(mol.mol)
            else:
                print(f'{mol.mol.GetProp("_Name")} do not have carbons connecting, appended to problematic mol object list')
                problematic_mol_object_list.append(mol.mol)
                not_connected_alkenes.append(mol.mol)
                mol_object_list12.remove(mol.mol)
        else:
            print(f'{mol_object.GetProp("_Name")} did not correctly return alkene boolean, appended to problematic mol object list')
            mol_object_list12.remove(mol.mol)
            problematic_mol_object_list.append(mol.mol)
    else:
        remain_mol_12.append(mol.mol)
print(f'Isolated {len(mol_object_list12)} alkenes, {len(remain_mol_12)} remaining.')

visualize_mols(f'Filter12_{dt}',mol_object_list12,highlight_alkene=True)

#Check 13: All clear! Grabs lactene-one
mol_object_list13 = list()
remain_mol_13 = list()

print('For Check #13')

for mol_object in remain_mol_12:
    mol = atom_class(mol_object)
    
    mol_bool = mol.sp2_type() & mol.carbon_type() & ~mol.het_neighbors_2() & mol.smarts_query('O=c(o)cc')

    if np.count_nonzero(mol_bool) == 2:
        mol_object_list13.append(mol.mol)
        removed_mol_names.append(mol.mol.GetProp("_Name"))
        #Sets property "_Alkene_w_H" to be equal to numpy boolean array
        mol.mol.SetProp("_Alkene_w_H", "".join('1' if v else '0' for v in mol_bool))
        original_array = np.array([True if v == '1' else False for v in mol.mol.GetProp("_Alkene_w_H")])
        #Tests to make sure array property is set and returned
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
                final_mol_object_list.append(mol.mol)
            else:
                print(f'{mol.mol.GetProp("_Name")} do not have carbons connecting, appended to problematic mol object list')
                problematic_mol_object_list.append(mol.mol)
                not_connected_alkenes.append(mol.mol)
                mol_object_list13.remove(mol.mol)
        else:
            print(f'{mol.mol.GetProp("_Name")} did not correctly return alkene boolean, appended to problematic mol object list')
            mol_object_list13.remove(mol.mol)
            problematic_mol_object_list.append(mol.mol)
    else:
        remain_mol_13.append(mol_object)
print(f'Isolated {len(mol_object_list13)} alkenes, {len(remain_mol_13)} remaining.')

visualize_mols(f'Filter13_{dt}',mol_object_list13,highlight_alkene=True)

visualize_mols(f'Filter13_Remaining_{dt}', remain_mol_13, highlight_alkene=False)

####Stores alkenes that need to be directly specified (multiple sites of reactivity)
mol_object_list14 = list()
remain_mol_14 = list()

print('For Check #14')

for mol_object in remain_mol_13:
    mol = atom_class(mol_object)
    
    mol_bool = mol.sp2_type() & mol.carbon_type() & ~mol.aromatic_type()
    if np.count_nonzero(mol_bool) > 2:
        mol_object_list14.append(mol.mol)
        removed_mol_names.append(mol.mol.GetProp("_Name"))
        #Sets property "_Alkene_w_H" to be equal to numpy boolean array
        mol.mol.SetProp("_Alkene_w_H", "".join('1' if v else '0' for v in mol_bool))
        original_array = np.array([True if v == '1' else False for v in mol.mol.GetProp("_Alkene_w_H")])
        #Tests to make sure array property is set and returned
        if all(original_array == mol_bool):
            print(f'{mol.mol.GetProp("_Name")} has multiple alkenes')
            multiple_alkenes.append(mol.mol)
            mol_object_list14.remove(mol.mol)
        else:
            print(f'{mol.mol.GetProp("_Name")} did not correctly return alkene boolean, appended to problematic mol object list')
            mol_object_list14.remove(mol.mol)
            problematic_mol_object_list.append(mol.mol)
    else:
        remain_mol_14.append(mol_object)

print(f'Isolated {len(mol_object_list14)} alkenes, {len(remain_mol_14)} remaining.\n')

visualize_mols(f'Filter14_Multiple_Alkenes_{dt}',mol_object_list13,highlight_alkene=True)

####Needed if not PropertyMol not already assigned####
# for i, j in enumerate(final_mol_object_list):
#     pm = PropertyMol(final_mol_object_list[i])
#     final_mol_object_list[i] = pm

with open(f'SAD_Step_3_First_Alkene_Filter_{dt}.pkl', 'wb') as f:
    pickle.dump(final_mol_object_list, f)
print(f'Saved final mol object list as "SAD_Step_3_First_Alkene_Filter_{dt}.pkl"')

if len(not_connected_alkenes) != 0:

    with open(f'SAD_Step_3_First_Alkene_Filter_not_connected_{dt}.pkl', 'wb') as f:
        pickle.dump(not_connected_alkenes, f)
    print(f'Saved not connected alkenes as "SAD_Step_3_First_Alkene_Filter_not_connected_{dt}.pkl"')

if len(multiple_alkenes) != 0:

    with open(f'SAD_Step_3_First_Alkene_Filter_multiple_alkenes_{dt}.pkl', 'wb') as f:
        pickle.dump(multiple_alkenes, f)
    print(f'Saved not connected alkenes as "SAD_Step_3_First_Alkene_Filter_multiple_alkenes_{dt}.pkl"')

if len(problematic_mol_object_list) != 0:

    with open(f'SAD_Step_3_First_Alkene_Filter_problematic_alkenes_{dt}.pkl', 'wb') as f:
        pickle.dump(problematic_mol_object_list, f)
    print(f'Saved problematic alkenes as "SAD_Step_3_First_Alkene_Filter_problematic_alkenes_{dt}.pkl"')

print(
f'''
There were {len(problematic_mol_object_list)} problematic alkenes
There were {len(final_mol_object_list)} alkenes isolated
There were {len(multiple_alkenes)} alkenes with multiple alkenes still present
There were {len(not_connected_alkenes)} unconnected carbons in the isolated alkene carbons
'''
)



