import molli as ml
import pandas as pd 
import numpy as np
from pprint import pprint
import os
import pickle
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.PropertyMol import PropertyMol
import rdkit.Chem.AllChem as ac
from rdkit.Geometry import Point3D
from rdkit.Chem import PyMol
from atom_class_test_p3 import atom_class
import re
from itertools import combinations
from glob import glob
from openbabel import openbabel as ob

def place_dummy_atom(rdkit_frag_obj, rdkit_can_smiles_no_wildcard, num_atoms_added:int, smarts: str = '[K:1]', verbose=False):
    '''
    This function returns a carbon1 fragment RDKIT OBJECT list and carbon2 fragment RDKIT OBJECT list in this order with a smarts attached. For ESPmin/Max use the SMARTS: 
    "[C:2][N+](C)(C)C". For Any individual dummy atoms: Use "['atom':2]". THESE RDKIT OBJECTS HAVE NOT BEEN SANITIZED.
    '''
    rxn = ac.ReactionFromSmarts(f'[#0:1]-{rdkit_can_smiles_no_wildcard}>>{smarts}-{rdkit_can_smiles_no_wildcard}')

    res = rxn.RunReactants((rdkit_frag_obj, ))[0][0]
    Chem.SanitizeMol(res)

    old_mol_atoms = np.array(list(x.GetSymbol() for x in rdkit_frag_obj.GetAtoms()))[1:]
    new_mol_atoms = np.array(list(x.GetSymbol() for x in res.GetAtoms()))[num_atoms_added:]
    if verbose:
        print(f'The result is {res}')
        print(f'The old mol is {rdkit_frag_obj}, the new mol is {Chem.MolToSmiles(res, canonical=False)}')
        print(f'old atoms is = {old_mol_atoms} and new_atoms is {new_mol_atoms}')

    assert np.array_equal(old_mol_atoms, new_mol_atoms), f'The order of old_mol_atoms: {old_mol_atoms} is not equal to the order new_mol_atoms: {new_mol_atoms}'

    return res

with open('SAD_Step_4_All_Reactant_Frags_3_01_2023.pkl', 'rb') as f:
    mol_list = pickle.load(f)

molli_frags = list()
Chem.SmilesParserParams.removeHs = False

wildcard_smarts_replace = '[H:1]'
wildcard_replace_label = 'H'

# wildcard_smarts_replace = '[C:1][N+](C)(C)C'
# wildcard_replace_label = 'ESP'

rdkit_frag_wildcard_dictionary = dict()

for rdkit_mol in mol_list:
    rdkit_name = rdkit_mol.GetProp("_Name")
    all_mol_frags = list()
    
    c1_frags_can_str = rdkit_mol.GetProp("can_C1_Fragments_w_H").split(',')
    f = 0
    for smiles in c1_frags_can_str:
        if smiles != '':
            c1_frag = PropertyMol(Chem.MolFromSmiles(smiles, Chem.SmilesParserParams.removeHs))
            Chem.SanitizeMol(c1_frag)
            c1_frag.SetProp("_Name", f'{rdkit_name}_c1_frag_{f}')
            f += 1
            all_mol_frags.append((smiles, c1_frag))

    c2_frags_can_str = rdkit_mol.GetProp("can_C2_Fragments_w_H").split(',')
    f = 0
    for smiles in c2_frags_can_str:
        if smiles != '':
            c2_frag = PropertyMol(Chem.MolFromSmiles(smiles, Chem.SmilesParserParams.removeHs))
            Chem.SanitizeMol(c2_frag)
            c2_frag.SetProp("_Name", f'{rdkit_name}_c2_frag_{f}')
            f += 1
            all_mol_frags.append((smiles,c2_frag))

    rdkit_frag_wildcard_dictionary[rdkit_name] = all_mol_frags  
    
    #This checks to make sure the canonical rdkit string/mol object directly matches the order of the Molli mol object
    for can_rdkit_smiles_w_wildcard, can_rdkit_mol in all_mol_frags:

        full_wildcard = re.match(r'\[\d{1,3}(?=\*)\*\]|^\*', can_rdkit_smiles_w_wildcard).group()
        frag_no_wildcard = can_rdkit_smiles_w_wildcard.replace(full_wildcard,'')

        mol_name = can_rdkit_mol.GetProp("_Name")
        # raise ValueError()

        mini_smarts_mol = Chem.MolFromSmarts(wildcard_smarts_replace)
        num_added_atoms = len(list(x.GetSymbol() for x in mini_smarts_mol.GetAtoms()))

        modified_rdkit_frag_obj = place_dummy_atom(rdkit_frag_obj=can_rdkit_mol, rdkit_can_smiles_no_wildcard=frag_no_wildcard, num_atoms_added=num_added_atoms, smarts=wildcard_smarts_replace)
        modified_rdkit_frag_mol_w_h = Chem.AddHs(modified_rdkit_frag_obj, addCoords=True)
        ac.EmbedMolecule(modified_rdkit_frag_mol_w_h)
        # modified_rdkit_frag_mol_w_h = Chem.AddHs(modified_rdkit_obj, addCoords=True)
        modified_rdkit_frag_mol_w_h.SetProp("_Name",f'{mol_name}_{wildcard_replace_label}')

        # try:
        ac.MMFFOptimizeMolecule(modified_rdkit_frag_mol_w_h)

        conv = ob.OBConversion()
        conv.SetInAndOutFormats("mol", 'mol2')
        obmol = ob.OBMol()
        obmol.SetTitle(can_rdkit_mol.GetProp("_Name"))
        conv.ReadString(obmol, Chem.MolToMolBlock(modified_rdkit_frag_mol_w_h))
        molli_frags.append(ml.Molecule.from_mol2(conv.WriteString(obmol), name=can_rdkit_mol.GetProp("_Name")))

print(len(molli_frags))

with open(f'SAD_Step_5_All_RDKit_Frags_Name_Frag_Dict_3_01_2023.pkl', 'wb') as f:
    pickle.dump(rdkit_frag_wildcard_dictionary, f)

col = ml.Collection(f'SAD_Step_5_All_Molli_Frags_{wildcard_replace_label}_3_01_2023',molli_frags)
col.to_zip(f'{col.name}.zip')