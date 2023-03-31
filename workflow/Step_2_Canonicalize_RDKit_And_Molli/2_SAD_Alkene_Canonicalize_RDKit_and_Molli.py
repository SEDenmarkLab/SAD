import molli as ml
from glob import glob 
import pickle
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.PropertyMol import PropertyMol
from rdkit import DataStructs
from rdkit.Chem import rdCIPLabeler
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import IPythonConsole
import time
import numpy as np
from datetime import date

# molli_mol_list = [ml.Molecule.from_mol2(file) for file in glob('./react_mol2s/*')]

# ml.Collection.to_zip(ml.Collection('SAD_Alkene_p0_react_w_type_10_28_2022',molli_mol_list),'SAD_Alkene_p0_react_w_type_10_28_2022.zip')

def can_mol_order(rdkit_mol):
    '''
    This a function tries to match the indexes of the canonicalized smiles string/molecular graph to a Molli Molecule object.
    Any inputs to this function will AUTOMATICALLY ADD HYDROGENS (make them explicit) to the RDKit mol object. This function returns 3 objects:

    1. Canonical RDKit Mol Object with Hydrogens and all maintained properties from the original rdkit mol
    2. A List for reordering the Atom Indices after canonicalization

    This third portion has failed unfortunately for unknown reasons (i.e. it is not generating the bond reordering property, this needs additional investigation)
    ~~3. A list for reordering the Bond Indices after canonicalization~~

    Important Notes:
    - It will only have "_Kekulize_Issue" if the initial object had this property set (i.e. if it ran into an issue in the in initial instantiation)
    - The canonical rdkit mol object will have the "Canonical SMILES with hydrogens" available as the property: "_Canonical_SMILES_w_H"
    - There may be some properties missing as the PropertyCache is not being updated on the new canonicalized mol object, so consider using rdkit_mol.UpdatePropertyCache() if you want to continue using the mol object
    '''

    #This is here to deal with any smiles strings or mol objects that do not get assigned hydrogens
    new_rdkit_mol = Chem.AddHs(rdkit_mol)

    #### This statement is necessary to generate the mol.GetPropertyName "_smilesAtomOutputOrder" and"_smilesBondOutputOrder"######
    Chem.MolToSmiles(new_rdkit_mol, canonical=True)
    # new_rdkit_mol.UpdatePropertyCache()
    # print(Chem.MolToSmiles(rdkit_mol, canonical=True))

    #The smiles output order is actually a string of the form "[0,1,2,3,...,12,]", so it requires a start at 1 and end at -2!
    #For some reason, in this current iteration, the SMILES Bond Output Order is not being generated as a property, but it seems to be because the bonds are already in the correct order
    # print(list(new_rdkit_mol.GetPropNames(includePrivate=True, includeComputed=True)))
    can_atom_reorder = list(map(int, new_rdkit_mol.GetProp("_smilesAtomOutputOrder")[1:-2].split(",")))
    if new_rdkit_mol.HasProp("_smilesBondOutputOrder"):
        canonical_bond_reorder_list = list(map(int, new_rdkit_mol.GetProp("_smilesBondOutputOrder")[1:-2].split(",")))
    else:
        canonical_bond_reorder_list = list(i for i in range(len(new_rdkit_mol.GetBonds())))
    can_smiles_w_h = Chem.MolToSmiles(new_rdkit_mol, canonical=True)

    # # # #Allows maintaining of hydrogens when Mol object is created
    Chem.SmilesParserParams.removeHs = False
    can_mol_w_h = PropertyMol(Chem.MolFromSmiles(can_smiles_w_h, Chem.SmilesParserParams.removeHs))

    #Helps new rdkit object maintain original properties of rdkit mol put in
    # Certain odd molecules result in some odd calculated properties, so this part is remaining commented out for now
    # can_mol_w_h.UpdatePropertyCache()

    # #Helps new rdkit object maintain original properties of rdkit mol put in
    # all_props_original_rdkit_mol = list(rdkit_mol.GetPropNames())
    # for prop in all_props_original_rdkit_mol:
    #     if not can_mol_w_h.HasProp(prop):
    #         print(prop)
    #         print(rdkit_mol.GetProp(prop))
    #         can_mol_w_h.SetProp(prop, rdkit_mol.GetProp(prop))

    can_mol_w_h.SetProp("_Name", rdkit_mol.GetProp("_Name"))
    can_mol_w_h.SetProp("_Alkene", rdkit_mol.GetProp("_Alkene"))
    can_mol_w_h.SetProp("_Canonical_SMILES_w_H", f'{can_smiles_w_h}')

    return can_mol_w_h, can_atom_reorder, canonical_bond_reorder_list

def reorder_molecule(molli_mol_object:ml.Molecule, canonical_rdkit_mol_w_h, canonical_atom_reorder_list, canonical_bond_reorder_list):
    '''
    This is a function that utilizes the outputs of new_mol_order to reorder an existing molecule
    '''

    #This reorders the atoms of the molecule object
    molli_atoms_arr = np.array(molli_mol_object.atoms)
    fixed_atom_order_list = molli_atoms_arr[canonical_atom_reorder_list].tolist()
    molli_mol_object.atoms = fixed_atom_order_list

    #This reorders the bonds of the molecule object
    molli_obj_bonds_arr = np.array(molli_mol_object.bonds)
    fixed_bond_order_list = molli_obj_bonds_arr[canonical_bond_reorder_list].tolist()
    molli_mol_object.bonds = fixed_bond_order_list

    #This fixes the geometry of the molecule object
    molli_mol_object.geom.coord = molli_mol_object.geom.coord[canonical_atom_reorder_list]

    #This checks to see if the new rdkit atom order in the canonical smiles matches the new molli order of atoms
    canonical_rdkit_atoms_list = canonical_rdkit_mol_w_h.GetAtoms()
    canonical_rdkit_atom_symbols_list = np.array([x.GetSymbol() for x in canonical_rdkit_atoms_list])
    canonical_rdkit_bonds_list = canonical_rdkit_mol_w_h.GetBonds()
                                             
    new_molli_symbol_list = np.array([x.symbol for x in molli_mol_object.atoms])
    equal_check = np.array_equal(canonical_rdkit_atom_symbols_list,new_molli_symbol_list)


    assert equal_check, f'Array of rdkit atoms: {canonical_rdkit_atom_symbols_list} is not equal to array of molli atoms: {new_molli_symbol_list}'

    return molli_mol_object

today = date.today()

dt = today.strftime("%m_%d_%Y")

collect = ml.Collection.from_zip('SAD_Alkene_Step_1_Reactants_w_Type_03_01_2023_MMFF_unordered.zip')

with open('SAD_Alkene_Step_1_Reactants_w_Type_03_01_2023.pkl', 'rb') as f:
    react_mols = pickle.load(f)

react_mol_dict = {mol.GetProp("_Name"):mol for mol in react_mols}

reordered_molli_list = list()
can_rdkit_mol_w_h_list = list()
multiple_alkenes = list()

val = 0

for molli_mol in collect:
    can_rdkit_mol_w_h, can_rdkit_atom_w_h, can_rdkit_bond_w_h = can_mol_order(react_mol_dict[molli_mol.name])
    reorder_molli_mol = reorder_molecule(molli_mol, can_rdkit_mol_w_h, can_rdkit_atom_w_h, can_rdkit_bond_w_h)
    reordered_molli_list.append(reorder_molli_mol)
    can_rdkit_mol_w_h_list.append(can_rdkit_mol_w_h)


for i, j in enumerate(can_rdkit_mol_w_h_list):
    pm = PropertyMol(can_rdkit_mol_w_h_list[i])
    can_rdkit_mol_w_h_list[i] = pm

ml.Collection.to_zip(ml.Collection(f'SAD_Step_2_Canonical_Molli_React_{dt}',reordered_molli_list), f'SAD_Step_2_Canonical_Molli_React_{dt}.zip')

with open(f"SAD_Step_2_Canonical_RDKitMol_w_h_React_{dt}", 'wb') as f:
    pickle.dump(can_rdkit_mol_w_h_list, f)
