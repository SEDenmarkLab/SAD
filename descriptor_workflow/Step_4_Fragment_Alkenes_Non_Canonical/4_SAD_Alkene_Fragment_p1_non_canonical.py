import pandas as pd 
import numpy as np
from pprint import pprint
import pickle
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.PropertyMol import PropertyMol
from atom_class_test_p3 import atom_class
import re

def visualize_mols(name, mol_list, highlight_alkene=False):
    '''
    Simple visualization script
    '''
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
        # _img = Draw.MolsToGridImage(mol_list, molsPerRow=5, subImgSize=(400,400), useSVG=True,returnPNG=False, highlightAtomLists=alkene_highlight_atom_list, highlightBondLists=alkene_highlight_bond_list, maxMols=20000)

        _img = Draw.MolsToGridImage(mol_list, molsPerRow=5, subImgSize=(400,400), useSVG=True,returnPNG=False, highlightAtomLists=alkene_highlight_atom_list, highlightBondLists=alkene_highlight_bond_list, legends=[i.GetProp("_Name") for i in mol_list], maxMols=20000)
        with open(f'{obj_name}.svg', 'w') as f:
            f.write(_img.data)
    return f'{obj_name}.png'

def remove_alkene_frag(carbon_atom, carbon_dict):
    '''
    This function removes the alkene fragment from the carbon dictionaries with a re statement, either matching an indexed number or "*" at the beginning of the SMILES String
    '''
    final_dict = dict()
    print(carbon_dict)
    print(carbon_atom.GetIdx())
    for smiles in carbon_dict:
        # indices = re.findall(r'\d{1,3}(?=\*)|^\*', smiles)
        indices = re.findall(r'\[\d{1,3}(?=\*)\*\]|\*(?=\])', smiles)
        print(indices)
        if len(indices) == 2:
            pass
        elif len(indices) == 1:
            if str(carbon_atom.GetIdx()) == re.search(r'\d{1,3}(?=\*)',indices[0]).group():
                final_dict.update({smiles: carbon_dict[smiles]})
            elif (carbon_atom.GetIdx() == 0) and ('*' == indices[0]):
                final_dict.update({smiles: carbon_dict[smiles]})
        elif len(indices) == 0:
            pass
        else:
            raise ValueError('Something Weird is happening with indices')
    return final_dict

def fragment_same_ring(rwmol, original_atom, original_bond, connected_to_original_atom, other_alkene_atom, neighbor_dict, ring_tuple,sanitize_and_add_h):
    '''
    This looks to fragment alkenes in the same ring. If there is a C1 and C2 carbon, it will identify either the fragment within the same ring by creating a wildcard on the piece attached to the first carbon and then automatically filling in the newly available second carbon bond with an implicit hydrogen. This should be used when the alkene carbons are in the same ring. The else statement fragments it as though it were a branched alkene
    '''
    #This asks if atom2, which is connected to carbon1, is in a ring
    if connected_to_original_atom.IsInRing():
        for neighbor_atom_idx in neighbor_dict:
            if (neighbor_atom_idx in ring_tuple) and (connected_to_original_atom.GetIdx() in ring_tuple):
                rwmol.RemoveBond(other_alkene_atom.GetIdx(), neighbor_atom_idx)
        #The rwmol has reset the bond indices, so I need to find the bond between carbon1 (original_atom) and the other atom part of the same ring.
        new_bond = rwmol.GetBondBetweenAtoms(original_atom.GetIdx(), connected_to_original_atom.GetIdx())
        #This fragments to form wildcards at the original atom carbon
        fragment = Chem.FragmentOnBonds(rwmol, [new_bond.GetIdx()])
        if sanitize_and_add_h:
            Chem.SanitizeMol(fragment)
            fragment = Chem.AddHs(fragment)
        return fragment
    else:
        #This fragments as though it were a branched alkene
        fragment = Chem.FragmentOnBonds(rwmol,[original_bond.GetIdx()])
        if sanitize_and_add_h:
            Chem.SanitizeMol(fragment)
        return fragment

def fragment_different_ring(rwmol, original_atom, original_bond, connected_to_original_atom, original_neighbor_dict, sanitize_and_add_h):
    '''
    This looks for exocyclic alkenes where the alkene bridges two different rings. This can be used for alkenes where the carbons fall in different rings or non-cyclic alkenes.
    '''
    #This asks if the atom connected to the original carbon is in a ring
    if connected_to_original_atom.IsInRing():
        for neighbor_atom_idx in original_neighbor_dict:
            if (original_neighbor_dict[neighbor_atom_idx].IsInRing()) & (neighbor_atom_idx != connected_to_original_atom.GetIdx()):
                rwmol.RemoveBond(original_atom.GetIdx(), neighbor_atom_idx)
                break
        #The rwmol has reset the bond indices, so I need to find the bond between the original atom and the original atom connected to the original atom
        new_bond = rwmol.GetBondBetweenAtoms(original_atom.GetIdx(),connected_to_original_atom.GetIdx())
        #This fragments to form wildcards at the original_atom_carbon
        fragment = Chem.FragmentOnBonds(rwmol, [new_bond.GetIdx()])
        if sanitize_and_add_h:
            Chem.SanitizeMol(fragment)
            fragment = Chem.AddHs(fragment)

        return fragment
    else:
        fragment = Chem.FragmentOnBonds(rwmol,[original_bond.GetIdx()])
        if sanitize_and_add_h:
            Chem.SanitizeMol(fragment)

        return fragment

def ring_alkene_fragment_same_ring(rdkit_mol: Chem.Mol, ring_tuple, carbon1: Chem.Atom, carbon2, carbon1_bond_list: list, carbon2_bond_list, c1_neighbor_dict: dict, c2_neighbor_dict: dict, sanitize_and_add_h):
    '''
    This fragments alkenes when they are in the same ring. This functions by fragmenting at one carbon leaving a wildcard, and fragmenting at the other end leaving an implicit hydrogen that can be filled in.
    '''
    carbon1_frag_dict = dict()
    carbon2_frag_dict = dict()

    for bond in carbon1_bond_list:
        rwmol = Chem.RWMol(rdkit_mol)
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        if atom1.GetIdx() == carbon1.GetIdx():
            fragment = fragment_same_ring(rwmol=rwmol, original_atom=carbon1, original_bond=bond,connected_to_original_atom=atom2, other_alkene_atom=carbon2,neighbor_dict=c2_neighbor_dict,ring_tuple=ring_tuple, sanitize_and_add_h=sanitize_and_add_h)

        elif atom2.GetIdx() == carbon1.GetIdx():
            fragment = fragment_same_ring(rwmol=rwmol, original_atom=carbon1, original_bond=bond,connected_to_original_atom=atom1,other_alkene_atom=carbon2,neighbor_dict=c2_neighbor_dict, ring_tuple=ring_tuple, sanitize_and_add_h=sanitize_and_add_h)

        frags1_mols = Chem.GetMolFrags(fragment, asMols=True)

        frags1_smiles_map = {Chem.MolToSmiles(rdkit_mol,canonical=False): rdkit_mol for rdkit_mol in frags1_mols}

        carbon1_frag_dict.update(frags1_smiles_map)

    final_carbon1_frag_dict = remove_alkene_frag(carbon1, carbon1_frag_dict)

    for bond in carbon2_bond_list:
        rwmol = Chem.RWMol(rdkit_mol)
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        if atom1.GetIdx() == carbon2.GetIdx():
            fragment = fragment_same_ring(rwmol=rwmol, original_atom=carbon2, original_bond=bond,connected_to_original_atom=atom2, other_alkene_atom=carbon1, neighbor_dict=c1_neighbor_dict,ring_tuple=ring_tuple, sanitize_and_add_h=sanitize_and_add_h)

        elif atom2.GetIdx() == carbon2.GetIdx():
            fragment = fragment_same_ring(rwmol=rwmol, original_atom=carbon2, original_bond=bond,connected_to_original_atom=atom1, other_alkene_atom=carbon1, neighbor_dict=c1_neighbor_dict,ring_tuple=ring_tuple, sanitize_and_add_h=sanitize_and_add_h)

        frags2_mols = Chem.GetMolFrags(fragment, asMols=True)
        
        frags2_smiles_map = {Chem.MolToSmiles(rdkit_mol,canonical=False): rdkit_mol for rdkit_mol in frags2_mols}

        carbon2_frag_dict.update(frags2_smiles_map)

    final_carbon2_frag_dict = remove_alkene_frag(carbon2, carbon2_frag_dict)

    return final_carbon1_frag_dict, final_carbon2_frag_dict

def ring_alkene_fragment_different_ring(rdkit_mol: Chem.Mol, carbon1: Chem.Atom, carbon2, carbon1_bond_list: list, carbon2_bond_list, c1_neighbor_dict: dict, c2_neighbor_dict: dict, sanitize_and_add_h):
    '''
    This function is used to deal with alkenes where the carbons are in separate rings, or only one carbon is in a ring. If an exocyclic alkene, this will fragment at one end of the alkene carbon and leave a wildcard and then on the other end 
    of the same alkene carbon, it will fragment and fill in with an implicit hydrogen. This can also function if there are no alkenes in rings. 
    '''
    carbon1_frag_dict = dict()
    carbon2_frag_dict = dict()

    for bond in carbon1_bond_list:
        rwmol = Chem.RWMol(rdkit_mol)
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        if atom1.GetIdx() == carbon1.GetIdx():
            fragment = fragment_different_ring(rwmol=rwmol,original_atom=carbon1,original_bond=bond,connected_to_original_atom=atom2,original_neighbor_dict=c1_neighbor_dict, sanitize_and_add_h=sanitize_and_add_h)

        elif atom2.GetIdx() == carbon1.GetIdx():
            fragment = fragment_different_ring(rwmol=rwmol,original_atom=carbon1,connected_to_original_atom=atom1,original_bond=bond,original_neighbor_dict=c1_neighbor_dict, sanitize_and_add_h=sanitize_and_add_h)

        frags1_mols = Chem.GetMolFrags(fragment, asMols=True)

        frags1_smiles_map = {Chem.MolToSmiles(rdkit_mol,canonical=False): rdkit_mol for rdkit_mol in frags1_mols}

        carbon1_frag_dict.update(frags1_smiles_map)
    
    final_carbon1_frag_dict = remove_alkene_frag(carbon1, carbon1_frag_dict)

    for bond in carbon2_bond_list:
        rwmol = Chem.RWMol(rdkit_mol)
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        if atom1.GetIdx() == carbon2.GetIdx():
            fragment = fragment_different_ring(rwmol=rwmol,original_atom=carbon2,original_bond=bond,connected_to_original_atom=atom2,original_neighbor_dict=c2_neighbor_dict, sanitize_and_add_h=sanitize_and_add_h)

        elif atom2.GetIdx() == carbon2.GetIdx():
            fragment = fragment_different_ring(rwmol=rwmol,original_atom=carbon2,original_bond=bond,connected_to_original_atom=atom1,original_neighbor_dict=c2_neighbor_dict, sanitize_and_add_h=sanitize_and_add_h)
    
        frags2_mols = Chem.GetMolFrags(fragment, asMols=True)

        frags2_smiles_map = {Chem.MolToSmiles(rdkit_mol,canonical=False): rdkit_mol for rdkit_mol in frags2_mols}

        carbon2_frag_dict.update(frags2_smiles_map)
    
    final_carbon2_frag_dict = remove_alkene_frag(carbon2, carbon2_frag_dict)

    return final_carbon1_frag_dict, final_carbon2_frag_dict

def fragment_w_rdkit_wildcard(rdkit_mol, alkene_property, sanitize_and_add_h=True, verbose=True):
    '''
    This is an all-encompassing function that can fragment any type of alkene with any geometry if it is given an rdkit object with an alkene boolean array indicating where the two carbons are. This will return the dictionary for the first carbon,
    a dictionary for the second carbon, and the original rdkit mol. This is designed to deal with branched alkenes and cyclic alkenes fundamentally differently. For branched alkenes, it will return a fragment with a wildcard labeled with the atom index of 
    the carbon it was detached from. For cyclic alkenes, it will fragment at one side of the alkene leaving a wildcard, and at the other attachment point of the ring (whether the same carbon (exocyclic alkenes) or different carbons (endocyclic alkenes)
    it will automatically populate an implicit hydrogen (explicit if sanitize_and_add_h is marked as True).
    '''
    
    mol = atom_class(rdkit_mol)
    alkene_boolean = np.array([True if v == '1' else False for v in mol.mol.GetProp(alkene_property)])

    #This returns the indexes for the two carbon atoms in an arbitrary order
    isolated_carbon_idx_np = mol.atoms_array[alkene_boolean]

    assert isolated_carbon_idx_np.shape[0] == 2, 'The isolated atoms should have a maximum of 2 atom indexes'

    #Represents bond between alkene carbons
    c1c2_bond = mol.mol.GetBondBetweenAtoms(carbon1_idx := int(isolated_carbon_idx_np[0]),carbon2_idx := int(isolated_carbon_idx_np[1]))

    ring_alkene_bool = alkene_boolean & mol.in_ring()

    #Simple properties of the first carbon
    carbon1 = mol.atoms_dict[carbon1_idx]
    carbon1_all_bonds = carbon1.GetBonds()

    #Simple properties of the second carbon
    carbon2 = mol.atoms_dict[carbon2_idx]
    carbon2_all_bonds = carbon2.GetBonds()

    #Neighbor Dictionaries
    c1_neighbor = carbon1.GetNeighbors()
    c1_neighbor_dict = {atom.GetIdx(): atom for atom in c1_neighbor if carbon2_idx != atom.GetIdx()}
    c2_neighbor = carbon2.GetNeighbors()
    c2_neighbor_dict = {atom.GetIdx(): atom for atom in c2_neighbor if carbon1_idx != atom.GetIdx()}

    #This filters out the alkene carbon-carbon bond, and makes sure only carbon and non-hydrogen bonds remain
    carbon1_temp1 = [bond for bond in carbon1_all_bonds if (bond.GetIdx() != c1c2_bond.GetIdx())]
    carbon1_temp2 = [bond for bond in carbon1_temp1 if (bond.GetBeginAtom().GetSymbol() != 'H') and (bond.GetEndAtom().GetSymbol() != 'H')]
    carbon1_bonds = [bond for bond in carbon1_temp2 if (bond.GetBeginAtom().GetSymbol() != 'D') and (bond.GetEndAtom().GetSymbol() != 'D')]

    #This filters out the alkene carbon-carbon bond, and makes sure only carbon-(non-hydrogen) bonds remain
    carbon2_temp1 = [bond for bond in carbon2_all_bonds if bond.GetIdx() != c1c2_bond.GetIdx()]
    carbon2_temp2 = [bond for bond in carbon2_temp1 if (bond.GetBeginAtom().GetSymbol() != 'H') and (bond.GetEndAtom().GetSymbol() != 'H')]
    carbon2_bonds = [bond for bond in carbon2_temp2 if (bond.GetBeginAtom().GetSymbol() != 'D') and (bond.GetEndAtom().GetSymbol() != 'D')]
    if verbose:
        print(f'the alkene name is {mol.mol.GetProp("_Name")}')
        print(f'the alkene boolean is {mol.mol.GetProp(alkene_property)}')
        print(f'{len(carbon1_bonds)} are how many carbon1 bonds')
        print(f'{len(carbon2_bonds)} are how many carbon2 bonds')
        print(f'The canonical_smiles {rdkit_mol.GetProp("_Canonical_SMILES_w_H")}')

    #This function is meant for any type of endocyclic alkene (includes cis-disubstituted, tri, tetra, and cyclohexylidene type alkenes)
    if np.count_nonzero(ring_alkene_bool) == 2:
        rings = mol.mol.GetRingInfo().AtomRings()
        for ring_tuple in rings:

            #Both alkene carbons are in the same ring
            if (carbon1_idx in ring_tuple) and (carbon2_idx in ring_tuple):

                carbon1_frag_dict,carbon2_frag_dict = ring_alkene_fragment_same_ring(rdkit_mol=mol.mol, ring_tuple=ring_tuple, carbon1=carbon1,carbon2=carbon2,carbon1_bond_list=carbon1_bonds,carbon2_bond_list=carbon2_bonds, c1_neighbor_dict=c1_neighbor_dict,c2_neighbor_dict=c2_neighbor_dict,sanitize_and_add_h=sanitize_and_add_h)

                return carbon1_frag_dict, carbon2_frag_dict, mol.mol

            #Checks if an index is or is not in the same ring (since both are noted as being in a ring, this should only need to be checked once)
            elif (carbon1_idx in ring_tuple) and (carbon2_idx not in ring_tuple):
                carbon1_frag_dict, carbon2_frag_dict = ring_alkene_fragment_different_ring(rdkit_mol=mol.mol,  carbon1=carbon1, carbon2=carbon2, carbon1_bond_list=carbon1_bonds,carbon2_bond_list=carbon2_bonds, c1_neighbor_dict=c1_neighbor_dict,c2_neighbor_dict=c2_neighbor_dict,sanitize_and_add_h=sanitize_and_add_h)
                
                return carbon1_frag_dict, carbon2_frag_dict, mol.mol
    #This deals with exocyclic alkenes
    elif np.count_nonzero(ring_alkene_bool) == 1:
        isolated_ring_atoms = [mol.atoms_dict[i] for i in np.where(ring_alkene_bool)[0]]
        ring_atom_idx = [atom.GetIdx() for atom in isolated_ring_atoms]
        #Confirms it should be an exocyclic alkene in both cases where either carbon1 or carbon2 are in different rings
        if (carbon1_idx in ring_atom_idx) & (carbon2_idx not in ring_atom_idx) | (carbon1_idx not in ring_atom_idx) & (carbon2_idx in ring_atom_idx):
            carbon1_frag_dict, carbon2_frag_dict = ring_alkene_fragment_different_ring(rdkit_mol=mol.mol, carbon1=carbon1, carbon2=carbon2, carbon1_bond_list=carbon1_bonds,carbon2_bond_list=carbon2_bonds, c1_neighbor_dict=c1_neighbor_dict,c2_neighbor_dict=c2_neighbor_dict,sanitize_and_add_h=sanitize_and_add_h)

            return carbon1_frag_dict, carbon2_frag_dict, mol.mol
    #This deals with branched alkenes
    elif np.count_nonzero(ring_alkene_bool) == 0:
        carbon1_frag_dict, carbon2_frag_dict = ring_alkene_fragment_different_ring(rdkit_mol=mol.mol, carbon1=carbon1, carbon2=carbon2, carbon1_bond_list=carbon1_bonds,carbon2_bond_list=carbon2_bonds, c1_neighbor_dict=c1_neighbor_dict,c2_neighbor_dict=c2_neighbor_dict,sanitize_and_add_h=sanitize_and_add_h)

        return carbon1_frag_dict, carbon2_frag_dict, mol.mol

with open('SAD_Step_3_All_Alkene_Filter_3_01_2023.pkl', 'rb') as f:
    mol_list = pickle.load(f)

# other_mol_list = list()
# for mol in mol_list:
#     if mol.GetProp("_Name") == 'react_256':
#         other_mol_list.append(mol)

# mol_list = other_mol_list
updated_mol_list = list()
x=0
for i in mol_list:
    carbon1_dict, carbon2_dict, rdkit_mol = fragment_w_rdkit_wildcard(i, alkene_property='_Alkene_w_H', sanitize_and_add_h=False, verbose=True)
    print(f'carbon1_dict is {carbon1_dict}')
    print(f'carbon2_dict is {carbon2_dict}')
    carbon1_dict_can_frag_smiles = list()
    carbon1_dict_can_atom_orders = list()
    # carbon1_dict_can_bond_orders = list()
    for key in carbon1_dict.keys():
        carbon1_dict_can_frag_smiles.append(Chem.MolToSmiles(carbon1_dict[key],canonical=True))
        carbon1_dict_can_atom_orders.append(carbon1_dict[key].GetProp("_smilesAtomOutputOrder"))
        # carbon1_dict_can_bond_orders.append(carbon1_dict[key].GetProp("_smilesBondOutputOrder"))
        # print(f'atoms are now {carbon1_dict[key].GetProp("_smilesAtomOutputOrder")}')
        # print(f'atoms are now {list(map(int, carbon1_dict[key].GetProp("_smilesAtomOutputOrder")[1:-2].split(",")))}')
        # # print(f'bonds are now {list(map(int, carbon1_dict[key].GetProp("_smilesBondOutputOrder")[1:-2].split(",")))}')

    rdkit_mol.SetProp("can_C1_Fragments_w_H", ','.join(carbon1_dict_can_frag_smiles))
    rdkit_mol.SetProp("non_can_C1_Fragments_w_H",','.join(list(carbon1_dict.keys())))
    rdkit_mol.SetProp("non_can_C1_Fragments_w_H_can_atom_orders", '/'.join(carbon1_dict_can_atom_orders))
    # rdkit_mol.SetProp("non_can_C1_Fragments_w_H_can_bond_orders", '/'.join(carbon1_dict_can_bond_orders))

    carbon2_dict_can_frag_smiles = list()
    carbon2_dict_can_atom_orders = list()
    # carbon2_dict_can_bond_orders = list()
    for key in carbon2_dict.keys():
        carbon2_dict_can_frag_smiles.append(Chem.MolToSmiles(carbon2_dict[key],canonical=True))
        carbon2_dict_can_atom_orders.append(carbon2_dict[key].GetProp("_smilesAtomOutputOrder"))
        # carbon2_dict_can_bond_orders.append(carbon2_dict[key].GetProp("_smilesBondOutputOrder"))

    rdkit_mol.SetProp("can_C2_Fragments_w_H", ','.join(carbon2_dict_can_frag_smiles))
    rdkit_mol.SetProp("non_can_C2_Fragments_w_H",','.join(list(carbon2_dict.keys())))
    rdkit_mol.SetProp("non_can_C2_Fragments_w_H_can_atom_orders", '/'.join(carbon2_dict_can_atom_orders))
    # rdkit_mol.SetProp("non_can_C2_Fragments_w_H_can_bond_orders", '/'.join(carbon2_dict_can_bond_orders))
    updated_mol_list.append(rdkit_mol)

with open('SAD_Step_4_All_Reactant_Frags_3_01_2023.pkl', 'wb') as f:
    pickle.dump(updated_mol_list, f)
