from rdkit import Chem
from rdkit.Chem.PropertyMol import PropertyMol
from openbabel import openbabel as ob
import numpy as np
import pickle
from typing import Dict
from glob import glob
from rd_atom_filter import rdkit_atom_filter

def load_obmol(fname, input_ext: str = 'xyz') -> ob.OBMol:
    '''
    This function takes any file and creates an openbabel style mol format
    '''

    conv = ob.OBConversion()
    obmol = ob.OBMol()
    conv.SetInFormat(input_ext)
    conv.ReadFile(obmol, fname)

    obmol.ConnectTheDots()
    obmol.PerceiveBondOrders()

    return obmol

def obmol_to_mol2(obmol: ob.OBMol):
    '''
    This returns a basic mol2 block
    '''
    conv = ob.OBConversion()
    conv.SetOutFormat('mol2')

    return conv.WriteString(obmol)

def obmol_to_xyz(obmol: ob.OBMol):
    '''
    This returns a basic xyz block
    '''

    conv = ob.OBConversion()
    conv.SetOutFormat('xyz')

    return conv.WriteString(obmol)

def canonicalize_rdkit_mol(rdkit_mol, removeHs=False) -> PropertyMol:
    '''
    Returns canonicalized RDKit mol generated from a canonicalized RDKit SMILES string. 
    
    Can indicate whether hydrogens should be removed from mol object or not.
    '''
    can_smiles = Chem.MolToSmiles(rdkit_mol, canonical=True)
    Chem.SmilesParserParams.removeHs = removeHs
    can_rdkit_mol = Chem.MolFromSmiles(can_smiles, Chem.SmilesParserParams.removeHs)
    can_rdkit_mol.UpdatePropertyCache()
    
    return can_rdkit_mol

def create_rdkit_mol(obmol:ob.OBMol, mol_name: str, removeHs=False) -> Dict[ob.OBMol,PropertyMol]:
    '''
    Uses mol2 generated from openbabel's implementation of mol2 generation.
    '''

    try:
        rdkit_mol = PropertyMol(Chem.MolFromMol2Block(obmol_to_mol2(obmol), removeHs=removeHs))
        rdkit_mol.SetProp("_Name", mol_name)
    except:
        rdkit_mol = PropertyMol(Chem.MolFromMol2Block(obmol_to_mol2(obmol), removeHs=removeHs, sanitize=False))
        rdkit_mol.SetProp("_Name", mol_name)
        rdkit_mol.SetProp("_Kekulize_Issue","1")

    return {obmol: rdkit_mol}

def can_mol_order(rdkit_mol):
    '''
    This a function tries to match the indexes of the canonicalized smiles string/molecular graph to a Molli Molecule object.
    Any inputs to this function will AUTOMATICALLY ADD HYDROGENS (make them explicit) to the RDKit mol object. This function returns 3 objects:

    1. Canonical RDKit Mol Object with Hydrogens and all maintained properties from the original rdkit mol
    2. A List for reordering the Atom Indices after canonicalization
    3. A list for reordering the Bond Indices after canonicalization

    Important Notes:
    - It will only have "_Kekulize_Issue" if the initial object had this property set (i.e. if it ran into an issue in the in initial instantiation)
    - The canonical rdkit mol object will have the "Canonical SMILES with hydrogens" available as the property: "_Canonical_SMILES_w_H"
    - There may be some properties missing as the PropertyCache is not being updated on the new canonicalized mol object, so consider using rdkit_mol.UpdatePropertyCache() if you want to continue using the mol object
    '''

    #This is here to deal with any smiles strings or mol objects that do not get assigned hydrogens
    new_rdkit_mol = Chem.AddHs(rdkit_mol)

    #### This statement is necessary to generate the mol.GetPropertyName "_smilesAtomOutputOrder" and"_smilesBondOutputOrder"######
    Chem.MolToSmiles(new_rdkit_mol, canonical=True)

    #The smiles output order is actually a string of the form "[0,1,2,3,...,12,]", so it requires a start at 1 and end at -2!
    # print(list(new_rdkit_mol.GetPropNames(includePrivate=True, includeComputed=True)))
    can_atom_reorder = list(map(int, new_rdkit_mol.GetProp("_smilesAtomOutputOrder")[1:-2].split(",")))
    canonical_bond_reorder_list = list(map(int, new_rdkit_mol.GetProp("_smilesBondOutputOrder")[1:-2].split(",")))
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
    can_mol_w_h.SetProp("_Canonical_SMILES_w_H", f'{can_smiles_w_h}')

    return can_mol_w_h, can_atom_reorder, canonical_bond_reorder_list

def reorder_molecule(obmol:ob.OBMol, can_rdkit_mol_w_h, can_atom_reorder:list, can_bond_reorder:list):
    '''
    This is a function that utilizes the outputs of new_mol_order to reorder an existing molecule.
    Currently done in place on the original molli_mol object.
    '''
    can_obmol = ob.OBMol()

    #This reorders the atoms of the molecule object
    old_atom_arr = np.array([obmol.GetAtomById(i) for i in range(obmol.NumAtoms())])
    fixed_atom_order_list = old_atom_arr[can_atom_reorder].tolist()

    for a in fixed_atom_order_list:
        oba: ob.OBAtom = can_obmol.NewAtom()
        oba.SetAtomicNum(a.GetAtomicNum())
        oba.SetVector(a.GetX(), a.GetY(), a.GetZ())

    #This reorders the bonds of the molecule object
    old_bond_arr = np.array([obmol.GetBondById(i) for i in range(obmol.NumBonds())])
    fixed_bond_order_list = old_bond_arr[can_bond_reorder].tolist()

    for b in fixed_bond_order_list:
        i1: ob.OBAtom = b.GetBeginAtomIdx()
        i2: ob.OBAtom = b.GetEndAtomIdx()
        order = b.GetBondOrder()
        obb: ob.OBBond = can_obmol.AddBond(i1 - 1, i2 - 1, order)


    #This checks to see if the new rdkit atom order in the canonical smiles matches the new molli order of atoms
    can_rdkit_atoms = can_rdkit_mol_w_h.GetAtoms()
    can_rdkit_atom_elem = np.array([x.GetSymbol() for x in can_rdkit_atoms])

    #This gets all of the symbols in the new "canonical" obmol object
    can_obmol_atom_list = [can_obmol.GetAtomById(i) for i in range(can_obmol.NumAtoms())]
    new_obmol_elem = np.array([ob.GetSymbol(a.GetAtomicNum()) for a in can_obmol_atom_list])

    equal_check = np.array_equal(can_rdkit_atom_elem,new_obmol_elem)

    assert equal_check, f'Array of rdkit atoms: {can_rdkit_atom_elem} is not equal to array of molli atoms: {new_obmol_elem}'

    return {can_obmol:can_rdkit_mol_w_h}

# all_mol2 = glob('*.mol2')

# for file in all_mol2:
#     name = file.split('.mol2')[0]
#     obmol = load_obmol(file, 'mol2')
#     with open(f'{name}.xyz', 'w') as f:
#         f.write(obmol_to_xyz(obmol))

all_xyz_files = glob('*.xyz')

final_rdkit_mol_list = list()

for file in all_xyz_files:
    name = file.split('.xyz')[0]
    
    example_obmol = load_obmol(file, 'xyz')

    obmol_rdkit_dict = create_rdkit_mol(
        obmol=example_obmol,
        mol_name=name,
        removeHs=False,
    )

    # can_rdkit_object = canonicalize_rdkit_mol(rdkit_mol = obmol_rdkit_dict[example_obmol], removeHs=False)

    for obmol, rdkit_mol in obmol_rdkit_dict.items():
        if rdkit_mol.HasProp("_Kekulize_Issue"):
            print('There was a kekulization issue for this molecule')


    rd_can_mol, atom_reorder, bond_reorder = can_mol_order(obmol_rdkit_dict[example_obmol])

    obmol_can_rdkit_dict = reorder_molecule(
        obmol=example_obmol,
        can_rdkit_mol_w_h = rd_can_mol,
        can_atom_reorder = atom_reorder,
        can_bond_reorder = bond_reorder
    )

    for can_obmol, can_rdkit_mol in obmol_can_rdkit_dict.items():
        can_obmol_atom_list = [can_obmol.GetAtomById(i) for i in range(can_obmol.NumAtoms())]
        can_obmol_elem_list = [ob.GetSymbol(a.GetAtomicNum()) for a in can_obmol_atom_list]
        print(f'For object "{can_rdkit_mol.GetProp("_Name")}')
        print(f'Canoncial RDKit SMILES:')
        print(Chem.MolToSmiles(can_rdkit_mol, canonical=False))
        print(f'Canonical obmol:')
        print(can_obmol_elem_list)
        print()

        with open(f'canonical_{name}.xyz', 'w') as f:
            f.write(obmol_to_xyz(can_obmol))
        
        final_rdkit_mol_list.append(can_rdkit_mol)


#This can be used to serialize and return lists of rdkit objects 
with open('final_can_rdkit_mol_list.pkl', 'wb') as f:
    pickle.dump(final_rdkit_mol_list, f)

with open('final_can_rdkit_mol_list.pkl', 'rb') as f:
    final_rdkit_mol_list = pickle.load(f)

#This is my own creation for allowing me to identify different substructures/query different properties of the molecule
for mol in final_rdkit_mol_list:
    af_mol = rdkit_atom_filter(mol)
    print(mol.GetProp("_Name"))
    atom_bool_test = (af_mol.aromatic_type())
    print(atom_bool_test)
    
