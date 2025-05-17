import molli as ml
from collections import deque
from molli.external.rdkit import atom_filter as mrd_af
from rdkit import Chem
import numpy as np
import pickle
from tqdm import tqdm
import pandas as pd
from pprint import pprint
import networkx as nx


def new_match(diol: ml.Molecule, pattern: ml.Connectivity, max_iter):
    nx_source = diol.to_nxgraph()
    nx_pattern = pattern.to_nxgraph()

    matcher = nx.isomorphism.GraphMatcher(
        nx_source,
        nx_pattern,
        node_match=ml.Connectivity._node_match,
        edge_match=ml.Connectivity._edge_match,
    )

    for i,iso in enumerate(matcher.subgraph_isomorphisms_iter()):
        if i == max_iter:
            break
        yield {v: k for k, v in iso.items()}
        
def find_matches(diol: ml.Molecule, pattern: ml.Connectivity, max_iter: int):
    '''This function iterates through all the possible subgraph isomorphisms

    Parameters
    ----------
    diol : ml.Molecule
        Diol to parse
    pattern : ml.Connectivity
        Substructure associated with the diol
    max_iter : int
        Maximum number of iterations

    Yields
    ------
    frozenset
        An immutable form of a set
    '''

    mappings = new_match(
        diol,
        pattern,
        max_iter=max_iter
    )

    atom_idx = {a: i for i, a in enumerate(diol.atoms)}

    for mapping in mappings:
        yield frozenset(atom_idx[mapping[x]] for x in pattern.atoms)

def find_arb_diol_q1q4_atoms(diol, _diol_c0, _diol_c1):
    diol_q1q4_atoms = list()
    # print(_diol_c1 in diol.atoms)
    for atom in diol.connected_atoms(_diol_c0): 
        # print(atom)
        #Ignores other diol carbon
        if atom == _diol_c1:
            # print('C1')
            continue
        else:
            #Tests if it's the diol
            add_atom = 1
            if atom.element == ml.Element.O:
                #Check connected atoms to determine if it's hydroxyl
                for con_atom in diol.connected_atoms(atom):
                    # print(f'Con Atom: {con_atom}')
                    if con_atom.element == ml.Element.H:
                        add_atom = 0
                    elif con_atom == _diol_c0:
                        continue
            # print(f'Add Atom!: {add_atom}')
            #If it's determined to be non-hydroxyl, it will add the atom.
            if add_atom:
                diol_q1q4_atoms.append(atom)
    
    assert len(diol_q1q4_atoms) == 2, f'There are an incorrect number of atoms found!: {diol_q1q4_atoms}'

    return diol_q1q4_atoms

def run_match(diol_alk_dict, diols, ref_mlib, diol_mlib, mlib_q1q4_connect, mlib_q1q4, final_diol_mlib):
    
    sym_mols = list()
    no_q1_match_mols = list()
    sym_q1q4 = list()
    with ref_mlib.reading(), diol_mlib.reading(), mlib_q1q4_connect.reading(), mlib_q1q4.reading(), final_diol_mlib.writing():
        for rd_diol in tqdm(diols):
            diol_name = rd_diol.GetProp("_Name")
            
            #Retrieves diol and its location
            ml_diol = diol_mlib[diol_name]
            diol_location = [i for i,x in enumerate(ml_diol.attrib["_Diol_w_H"]) if x == "1"]
            assert len(diol_location) == 2, f'More than two carbons marked for diols! {diol_location}'


            alk_name = diol_alk_dict[diol_name]
            #Retrieves alkene and associated quadrant values
            alk_mlmol = ref_mlib[alk_name]
            alk_c0,alk_c1 = [alk_mlmol.get_atom(x) for x in alk_mlmol.attrib['C Order']]
            alk_c0_idx, alk_c1_idx = [alk_mlmol.get_atom_index(x) for x in [alk_c0,alk_c1]]

            alk_q_atoms = [alk_mlmol.get_atom(x) for x in alk_mlmol.attrib['Q Order']]
            alk_q1a,alk_q2a,alk_q3a,alk_q4a = alk_q_atoms
            alk_q1a_idx,alk_q2a_idx,alk_q3a_idx,alk_q4a_idx = [alk_mlmol.get_atom_index(x) for x in alk_q_atoms]

            connect_mlmol=mlib_q1q4_connect[f'{alk_name}_Q1Q4']
            q1_mlmol = mlib_q1q4[f'{alk_name}_Q1']
            q4_mlmol = mlib_q1q4[f'{alk_name}_Q4']

            #Finds unique subgraph isomorphisms
            diol_q1q4_sub_search = {x for x in find_matches(ml_diol, connect_mlmol, max_iter=max_iter)}

            #If there are multiple matches, it has failed.
            if len(diol_q1q4_sub_search) != 1:
                sym_mols.append((alk_name, diol_name))
                continue
                assert len(diol_q1q4_sub_search) == 1, f'diol_q1q4_sub_search has multiple matches! {diol_q1q4_sub}'
            else:
                
                #Isolates the frozenset corresponding to the Q1Q4 match
                diol_q1q4_sub_idx, = diol_q1q4_sub_search
                x = 0
                #Finds the two carbons of the diol associated with the alkene
                for i in diol_location:
                    if i in diol_q1q4_sub_idx:
                        x += 1
                        # print(f'C0 = {i}')
                        diol_c0 = ml_diol.get_atom(i)
                        diol_c0.attrib['C0'] = i
                    else:
                        x += 1
                        # print(f'C1 = {i}')
                        diol_c1 = ml_diol.get_atom(i)
                        diol_c1.attrib['C1'] = i

                assert x == 2, f'There were not two alkene carbon atoms found! {x} atoms found'

                diol_q1q4_atoms = find_arb_diol_q1q4_atoms(ml_diol, diol_c0, diol_c1)

                #Searches for the atoms associated with Q1
                diol_q1_sub_search = {x for x in find_matches(ml_diol, q1_mlmol, max_iter=max_iter)}

                #If no match occurred, it finds this
                if len(diol_q1_sub_search) != 1:
                    no_q1_match_mols.append((alk_name,diol_name))
                    continue
                    assert len(diol_q1_sub_search) == 1, f'diol_q1_sub_search has multiple matches! {diol_q1_sub_search}'

                else:
                    diol_q1q4_sub = ml.Substructure(ml_diol, diol_q1q4_sub_idx)
                    
                    diol_q1_sub_idx, = diol_q1_sub_search
                    diol_q1_sub = ml.Substructure(ml_diol, diol_q1_sub_idx)

                    #IF the Q1 match worked, the Q4 index can be isolated with the symmetric difference between Q1 and Q4 substructures
                    isolate_q4_sub_idx = list(diol_q1q4_sub_idx.symmetric_difference(diol_q1_sub_idx))
                    isolate_q4_sub_idx.remove(ml_diol.get_atom_index(diol_c0))
                    if len(isolate_q4_sub_idx) == 0:
                        sym_q1q4.append((alk_name,diol_name))
                        continue
                    # print(isolate_q4_sub_idx)
                    x = 0
                    
                    #Assigns the Q1/Q4 indices
                    for atom in diol_q1q4_atoms:
                        atom_idx = ml_diol.get_atom_index(atom)
                        if atom_idx in isolate_q4_sub_idx:
                            x += 1
                            # print(f'Q4 = {atom_idx}')
                            diol_q4 = atom
                            diol_q4.attrib['Q'] = 4
                        else:
                            x += 1
                            # print(f'Q1 = {atom_idx}')
                            diol_q1 = atom
                            diol_q1.attrib['Q'] = 1

                    assert x == 2, f'There were not two quadrant atoms found! {x} atoms found'
                    
                    #Assigns the quadrant orders
                    ml_diol.attrib['C Order'] = [ml_diol.get_atom_index(x) for x in [diol_c0,diol_c1]]
                    ml_diol.attrib['Q1Q4 Order'] = [ml_diol.get_atom_index(x) for x in [diol_q1,diol_q4]]

                    final_diol_mlib[diol_name] = ml_diol

    print(f'{len(sym_mols)} with symmetry')
    print(f'{len(no_q1_match_mols)} with no q matches')
    print(f'{len(sym_q1q4)} symmetrical q1q4')

max_iter=10000

DB_df = pd.read_csv("SAD_Database.csv")

diol_alk_dict = dict(DB_df[["Product ID", "Reactant ID"]].values)

with open("4_Diol_w_H_Filter.pkl", "rb") as f:
    diols = pickle.load(f)

ref_BFSVol_mlib = ml.MoleculeLibrary("6_7_Realign_3BFSVol.mlib")
diol_BFSVol_mlib = ml.MoleculeLibrary("4_DB_Diols_w_H.mlib")
BFSVol_mlib_q1q4_connect = ml.MoleculeLibrary("5_1_q1q4_connect_3BFSVol.mlib")
BFSVol_mlib_q1q4 = ml.MoleculeLibrary("5_1_q1q4_frags_3BFSVol.mlib")
final_diol_BFSVol_mlib = ml.MoleculeLibrary(
    f"5_2_Diol_Q1Q4_3BFSVol_Assign_{max_iter}iter.mlib", readonly=False, overwrite=True
)

run_match(
    diol_alk_dict=diol_alk_dict,
    diols=diols,
    ref_mlib=ref_BFSVol_mlib, 
    diol_mlib=diol_BFSVol_mlib,
    mlib_q1q4_connect=BFSVol_mlib_q1q4_connect,
    mlib_q1q4=BFSVol_mlib_q1q4,
    final_diol_mlib=final_diol_BFSVol_mlib)

ref_maxvol_mlib = ml.MoleculeLibrary("6_7_Realign_MaxVol.mlib")
diol_maxvol_mlib = ml.MoleculeLibrary("4_DB_Diols_w_H.mlib")
maxvol_mlib_q1q4_connect = ml.MoleculeLibrary("5_1_q1q4_connect_MaxVol.mlib")
maxvol_mlib_q1q4 = ml.MoleculeLibrary("5_1_q1q4_frags_MaxVol.mlib")
final_diol_maxvol_mlib = ml.MoleculeLibrary(
    f"5_2_Diol_Q1Q4_MaxVol_Assign_{max_iter}iter.mlib", readonly=False, overwrite=True
)

run_match(
    diol_alk_dict=diol_alk_dict,
    diols=diols,
    ref_mlib=ref_maxvol_mlib, 
    diol_mlib=diol_maxvol_mlib,
    mlib_q1q4_connect=maxvol_mlib_q1q4_connect,
    mlib_q1q4=maxvol_mlib_q1q4,
    final_diol_mlib=final_diol_maxvol_mlib)