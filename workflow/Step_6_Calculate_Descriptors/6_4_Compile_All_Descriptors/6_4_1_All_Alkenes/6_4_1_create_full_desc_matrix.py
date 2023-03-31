import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.PropertyMol import PropertyMol
import pickle
import molli as ml
from pprint import pprint
from alkene_class import alkene_recognize as akr
from rdkit.Chem import AllChem as ac
from atom_class_test_p3 import atom_class
import yaml
from bidict import bidict
from collections import deque

def _bfs_single(q: deque, visited: set, alk_mol:akr):
    '''
    Built from Alex Shved's Molli 1.0 molli.chem.bond implementation
    '''
    start, dist = q.pop()
    for a in alk_mol.neighbor_idx_list(atom_idx=start):
        if a not in visited:
            yield (a, dist+1)
            visited.add(a)
            q.appendleft((a, dist + 1))

def yield_bfsd(alk_mol: akr, start_idx:int, no_visit_c1: int, no_visit_c2):
    '''
    Built from Alex Shved's Molli 1.0 molli.chem.bond implementation.
    This will return a generator object
    '''
    _sa = start_idx
    
    visited = set((_sa, no_visit_c1, no_visit_c2,))
    q = deque([(_sa, 0)])
    while q:
        yield from _bfs_single(q,visited, alk_mol)

def create_sterimol_series(sub_name:str, sterimol_esp_df, num):
    sterimol_esp_val = sterimol_esp_df.loc[sub_name]
    sterimol_esp_val.index = [f'{idx}_{num}' for idx in sterimol_esp_val.index]
    return sterimol_esp_val

def create_sterimol_test_series(sub_name, sterimol_esp_df):
    return sterimol_esp_df.loc[sub_name]

def test_cis_trans(alk_mol, coords, sub_0_idx, high_vol_c1, low_vol_c2):
    '''
    This tests if the first idx of the neighbor list is cis or trans to the first substituent.
    This is used when test1 and test2 have not been defined yet
    '''
    sub_0_coords = coords[sub_0_idx]
    high_vol_c1_coords = coords[high_vol_c1]

    v1_vec = high_vol_c1_coords-sub_0_coords

    low_vol_idx0, low_vol_idx1 = alk_mol.neighbor_idx_list(atom_idx=low_vol_c2, atom_idx_not_included=high_vol_c1)

    test1_idx = low_vol_idx0
    test2_idx = low_vol_idx1

    test1_coords = coords[test1_idx]
    test2_coords = coords[test2_idx]

    v2_vec = test2_coords-test1_coords

    '''
    v1 = vector from largest substituent first idx to high_vol_c1_idx 
    v2 = vector from a test1 -> test 2               
    '''
    if np.sign(np.dot(v1_vec,v2_vec)) == 1:
        test1_orientation = 'cis'
    else:
        test1_orientation = 'trans'
    
    return test1_orientation, test1_idx, test2_idx

def test_cis_trans_known_idx(alk_mol, coords, sub_0_idx, low_vol_idx0, low_vol_idx1, high_vol_c1):
    '''
    This tests if the a vector drawn from implies cis or trans geometry:

    v1 = vector from largest substituent first idx to high_vol_c1_idx 
    v2 = vector from a test1 -> test 2    
    '''
    sub_0_coords = coords[sub_0_idx]
    high_vol_c1_coords = coords[high_vol_c1]

    v1_vec = high_vol_c1_coords-sub_0_coords

    test1_idx = low_vol_idx0
    test2_idx = low_vol_idx1

    test1_coords = coords[test1_idx]
    test2_coords = coords[test2_idx]

    v2_vec = test2_coords-test1_coords

    '''
    v1 = vector from largest substituent first idx to high_vol_c1_idx 
    v2 = vector from a test1 -> test 2               
    '''
    if np.sign(np.dot(v1_vec,v2_vec)) == 1:
        test1_orientation = 'cis'
    else:
        test1_orientation = 'trans'
    
    return test1_orientation, test1_idx, test2_idx


def bfs_test_idx(
        alk_mol:akr, 
        frag_tuples, 
        test1_sub_connect, 
        test1_atom_idx, 
        tuple_idx0, 
        test2_sub_connect, 
        test2_atom_idx,
        tuple_idx1, 
        high_vol_c1, 
        low_vol_c2):
    '''
    This does a bfs_test based on input indices.
    The goal of this code is to force an ordering that is reproducible for mol atoms. 
    This will take an input set of fragment tuples, test fragment names and connectivity,
    and then it will match the fragment to the index it is associated with. There is a 3-layered test for this:

    The first test is confirming that lists are the same length
    
    The second test is confirming the lists contain the same symbols

    The third and final test is confirming if there are aromatic atoms that need to be differentiated.

    This is currently enough to differentiate all the fragments that I have.

    '''

    assert test1_sub_connect == test2_sub_connect, f'Test1 and Test2 are connected to different carbons'

    test1_mol = Chem.MolFromSmiles(frag_tuples[tuple_idx0][0], Chem.SmilesParserParams.removeHs)
    test1_mol_atoms_sym = [atom.GetSymbol() for atom in test1_mol.GetAtoms()]
    test1_mol_atoms_sym_no_wildcard = [sym for sym in test1_mol_atoms_sym if sym != '*']
    test1_aromatic_atoms_no_wildcard = np.array([atom.GetIsAromatic() for i,atom in enumerate(test1_mol.GetAtoms()) if i != 0])
    test1_bfsd = list(yield_bfsd(
        alk_mol=alk_mol,
        start_idx=test1_atom_idx,
        no_visit_c1 = high_vol_c1,
        no_visit_c2 = low_vol_c2
    ))

    test2_bfsd = list(yield_bfsd(
        alk_mol=alk_mol,
        start_idx=test2_atom_idx,
        no_visit_c1 = high_vol_c1,
        no_visit_c2 = low_vol_c2
    ))
    test2_mol = Chem.MolFromSmiles(frag_tuples[tuple_idx1][0], Chem.SmilesParserParams.removeHs)
    test2_mol_atoms_sym = [atom.GetSymbol() for atom in test2_mol.GetAtoms()]

    #This is for dealing with unique boundary cases in the graph that wouldn't be caught by assuming the lengths are the same
    if len(test1_bfsd) == len(test2_bfsd):
        #Gets atom idx numbers
        test1_idx_bfsd_list = [val[0] for val in test1_bfsd]
        test1_idx_bfsd_list.append(test1_atom_idx)
        test1_sym_bfsd_list = [alk_mol.atoms_dict[idx].GetSymbol() for idx in test1_idx_bfsd_list]
        test1_aromatic_bfsd_list = np.array([alk_mol.atoms_dict[idx].GetIsAromatic() for idx in test1_idx_bfsd_list])

        #This tests that the empirical formula of each frag is the same, if not, test1_sub and test2_sub names are flipped (i.e. fragments are incorrect)
        if not all(sym in test1_mol_atoms_sym_no_wildcard for sym in test1_sym_bfsd_list):
            test1_sub_name = frag_tuples[tuple_idx1][1].GetProp("_Name")
            test1_idx = test2_atom_idx
            test2_sub_name = frag_tuples[tuple_idx0][1].GetProp("_Name")
            test2_idx = test1_atom_idx
            print('Test1 and Test2 Reordered')
        #This tests that there are the same number of aromatic atoms                   
        elif np.count_nonzero(test1_aromatic_atoms_no_wildcard) != np.count_nonzero(test1_aromatic_bfsd_list):
            test1_sub_name = frag_tuples[tuple_idx1][1].GetProp("_Name")
            test1_idx = test2_atom_idx
            test2_sub_name = frag_tuples[tuple_idx0][1].GetProp("_Name")
            test2_idx = test1_atom_idx
            print('Test1 and Test2 Reordered')
        else:
            #This means they are already in the correct order
            test1_idx = test1_atom_idx
            test1_sub_name = frag_tuples[tuple_idx0][1].GetProp("_Name")
            test2_idx = test2_atom_idx
            test2_sub_name = frag_tuples[tuple_idx1][1].GetProp("_Name")

    else:
        #Test1/Test2 is in the correct order
        if len(test1_bfsd) == len(test1_mol_atoms_sym)-2:
            assert len(test2_bfsd) == len(test2_mol_atoms_sym)-2, f'Length of test2_bfsd ({len(test2_bfsd)}) != Length of test2_mol_atoms-2 ({len(test2_mol_atoms_sym-2)})'
            test1_idx = test1_atom_idx
            test1_sub_name = frag_tuples[tuple_idx0][1].GetProp("_Name")
            test2_idx = test2_atom_idx
            test2_sub_name = frag_tuples[tuple_idx1][1].GetProp("_Name")
        else:
            assert len(test1_bfsd) == len(test2_mol_atoms_sym)-2, f'Length of test1_bfsd ({len(test1_bfsd)}) != Length of test2_mol_atoms-2 ({len(test2_mol_atoms_sym-2)})'
            assert len(test2_bfsd) == len(test1_mol_atoms_sym)-2, f'Length of test2_bfsd ({len(test2_bfsd)}) != Length of test1_mol_atoms-2 ({len(test1_mol_atoms_sym-2)})'
            test1_idx = test2_atom_idx
            test1_sub_name = frag_tuples[tuple_idx1][1].GetProp("_Name")
            test2_idx = test1_atom_idx
            test2_sub_name = frag_tuples[tuple_idx0][1].GetProp("_Name")
            print('Test1 and Test2 Reordered')

    return test1_idx, test1_sub_name, test2_idx, test2_sub_name 

#alkene_col
full_mol_col = ml.Collection.from_zip('almost_all_alkenes_opt.zip')

#database
db_df = pd.read_csv('p8_reduced_database_column_update_1001.csv')

#yml ref

with open('database_fixed_update_1001.yaml') as f:
    yml = yaml.safe_load(f)
    yml_react = bidict(yml['react_id_smiles_map'])

#Note: This is actually the full mols with the fragments inside the overall rdkit mol object
with open('SAD_Step_4_All_Reactant_Frags_3_01_2023.pkl', 'rb') as f:
    frag_mol_list = pickle.load(f)

mol_dict = {mol.GetProp("_Name"): mol for mol in frag_mol_list}

#This is going to be used to assign the correct name to each of the fragments
with open('SAD_Step_5_All_RDKit_Frags_Name_Frag_Dict_3_01_2023.pkl', 'rb') as f:
    mol_frag_dict = pickle.load(f)

#This exists because the Alkene Type did not get transferred to each of the alkenes
for mol in frag_mol_list:
    current_name = mol.GetProp("_Name")
    olefin_type_arr = db_df.query('`Reactant ID` == @current_name')['Olefin Type'].to_numpy()
    if olefin_type_arr.shape != (1,):
        assert all([True for i in olefin_type_arr if i == olefin_type_arr[0]]), 'rip'

    olefin_type = olefin_type_arr[0]
    mol.SetProp("_Alkene_Type",f'{olefin_type}')

#nbo_dictionary
with open('Step_6_1000_Entry_NBO_almost_all_alkenes_dict.pkl', 'rb') as f:
    nbo_dict = pickle.load(f)
# pprint(nbo_dict)
# print()
#RDF dictionary corresponding to 
with open(f'Step_6_Alkene_RDF_dict.pkl', 'rb') as f:
    alkene_rdf_dict = pickle.load(f)

#dummy hydrogen values
dummy_df = pd.read_csv('hydrogen_test.csv', index_col=0)
#fragment_sterimol dictionary (built from ESP structures and hydrogen dummy frags)
frag_sterimol_esp = pd.read_csv('Step_6_1000_Entry_Sterimol_ESP_Frag_desc_df.csv', index_col=0)
sterimol_esp_idx = frag_sterimol_esp.index

molli_frags = list()

problem_mols = ['react_25', 'react_28', 'react_223', 'react_580', 'react_582']

full_df = pd.DataFrame()

# problematic_tuples = list()
# pprint(nbo_dict.keys())

##This is the massive chunk of code that assembles the reactant descriptor dictionary##
for name,frag_tuples in mol_frag_dict.items():
    react_smiles = yml_react.inverse[name]
    print(name)
    # print(react_smiles)
    if name in problem_mols:
        continue
    alk_mol = akr(mol_dict[name])
    Chem.SmilesParserParams.removeHs = False
    # print(Chem.MolToSmiles(alk_mol.rdkit_alkene_mol, Chem.SmilesParserParams.removeHs))
    c1 = alk_mol.c1_idx
    c2 = alk_mol.c2_idx

    if alk_mol.alkene_type == 'mono':
        '''
        Alignment will look like this:
        Sub | H
        ----------
        H  | H

        '''
        #Asserts that there is only one value in the list for a monosubstituted alkene
        assert len(frag_tuples) == 1, f'Mono Substituted has != 1 frag = {frag_tuples}'
        #Finds the name of the fragment for utilizing the dictionary
        sub_name = frag_tuples[0][1].GetProp("_Name")
        sub_connect = sub_name.split('_')[2]
        #Retrieves the series for the sterimol and esp descriptors for this alkene
        sterimol_esp_val_0 = create_sterimol_series(sub_name, frag_sterimol_esp, num=0)
        sterimol_esp_val_1 = create_sterimol_series('hydrogen_test', dummy_df, num=1)
        sterimol_esp_val_2 = create_sterimol_series('hydrogen_test', dummy_df, num=2)
        sterimol_esp_val_3 = create_sterimol_series('hydrogen_test', dummy_df, num=3)

        #Matches the nbo dictionary for the various values
        try:
            orb_homo_lumo, nat_charge_dict, pert_final_list, nbo_orb_final_list = nbo_dict[alk_mol.mol_name]
        except:
            alk_mol.mol_name = f'{alk_mol.mol_name}_opt_freq_conf_6'
            orb_homo_lumo, nat_charge_dict, pert_final_list, nbo_orb_final_list = nbo_dict[alk_mol.mol_name]
        # print(nat_charge_dict)
        homo = orb_homo_lumo[0]
        lumo = orb_homo_lumo[1]
        rdf_dict = alkene_rdf_dict[alk_mol.mol_name]

        if sub_connect == 'c1':
            nat_charge_1 = nat_charge_dict[c1]
            rdf_1_ser    = rdf_dict[0]
            rdf_1_ser.index = [f'{idx}_0' for idx in rdf_1_ser.index]
            nat_charge_2 = nat_charge_dict[c2]
            rdf_2_ser        = rdf_dict[1]
            rdf_2_ser.index = [f'{idx}_1' for idx in rdf_2_ser.index]
        else:
            nat_charge_1 = nat_charge_dict[c2]
            rdf_1_ser    = rdf_dict[1]
            rdf_1_ser.index = [f'{idx}_0' for idx in rdf_1_ser.index]
            nat_charge_2 = nat_charge_dict[c1]
            rdf_2_ser        = rdf_dict[0]
            rdf_2_ser.index = [f'{idx}_1' for idx in rdf_2_ser.index]
        pert_vals = list()
        for (indices, value) in pert_final_list:
            if (c1 and c2 in indices):
                pert_vals.append(value)
        pert_arr = np.array(pert_vals)
        sort_pert_arr = pert_arr[np.argsort(pert_arr)[::-1]]
        top_4_pert_vals = sort_pert_arr[0:4]

        assert len(pert_vals) >= 3, f'pert_vals is {pert_vals}'

        for (bd_type, indices, bd_order, energy) in nbo_orb_final_list:
            if (bd_type == 'BD') and (bd_order == '2'):
                pi_orb_energy = energy
            elif (bd_type == 'BD*') and (bd_order == '2'):
                anti_pi_orb_energy = energy


        add_ser = {
            'homo': homo, 
            'lumo': lumo, 
            'nat_charge_1': nat_charge_1, 
            'nat_charge_2': nat_charge_2, 
            'pert_1': top_4_pert_vals[0], 
            'pert_2': top_4_pert_vals[1], 
            'pert_3': top_4_pert_vals[2],
            'pi_orb': pi_orb_energy, 
            'pi*_orb': anti_pi_orb_energy,
            }
        nbo_ser = pd.Series(add_ser)
        concat_df = pd.concat([sterimol_esp_val_0, sterimol_esp_val_1, sterimol_esp_val_2, sterimol_esp_val_3, nbo_ser, rdf_1_ser, rdf_2_ser], axis=0).to_frame(name = name)
        fix_df = concat_df.transpose()
        full_df = pd.concat([full_df, fix_df], axis=0)

    if alk_mol.alkene_type == 'gem_di':
        '''
        Alignment will look like this based on a reorder by highest volume:
        Sub_0 | H
        ----------
        Sub_1  | H

        '''

        #Asserts that there are 2 value in the list for a monosubstituted alkene
        assert len(frag_tuples) == 2, f'frag_tuples = {frag_tuples}'
        #Finds the name of the fragment for utilizing the dictionary
        test_sub_name_0 = frag_tuples[0][1].GetProp("_Name")
        test_sub_0_connect = test_sub_name_0.split('_')[2]
        test_sub_name_1 = frag_tuples[1][1].GetProp("_Name")
        test_sub_1_connect = test_sub_name_1.split('_')[2]
        assert test_sub_0_connect == test_sub_1_connect, f'sub_0 not conneted to sub_1 for gem_di: sub_0 = {test_sub_0_connect}, sub_1 = {test_sub_1_connect}'
        #Retrieves the series for the sterimol and esp descriptors for this alkene
        test_sterimol_esp_val_0 = create_sterimol_test_series(test_sub_name_0, frag_sterimol_esp)
        test_sterimol_esp_val_1 = create_sterimol_test_series(test_sub_name_1, frag_sterimol_esp)

        test_sterimol_vol_0 = test_sterimol_esp_val_0['vol']
        test_sterimol_vol_1 = test_sterimol_esp_val_1['vol']

        #Current order is correct (i.e. sub_0 = sub_0)
        if test_sterimol_vol_0 >= test_sterimol_vol_1:
            sub_name_0 = test_sub_name_0
            sub_0_connect = test_sub_0_connect
            sub_name_1 = test_sub_name_1
            sub_1_connect = test_sub_1_connect

        #Current order needs to be inverted (i.e. sub_0 = sub_1)
        else:
            sub_name_0 = test_sub_name_1
            sub_0_connect = test_sub_1_connect
            sub_name_1 = test_sub_name_0
            sub_1_connect = test_sub_0_connect
            
        sterimol_esp_val_0 = create_sterimol_series(sub_name_0, frag_sterimol_esp, num=0)
        sterimol_esp_val_1 = create_sterimol_series('hydrogen_test', dummy_df, num=1)
        sterimol_esp_val_2 = create_sterimol_series('hydrogen_test', dummy_df, num=2)
        sterimol_esp_val_3 = create_sterimol_series(sub_name_1, frag_sterimol_esp, num=3)

        try:
            orb_homo_lumo, nat_charge_dict, pert_final_list, nbo_orb_final_list = nbo_dict[alk_mol.mol_name]
        except:
            alk_mol.mol_name = f'{alk_mol.mol_name}_opt_freq_conf_6'
            orb_homo_lumo, nat_charge_dict, pert_final_list, nbo_orb_final_list = nbo_dict[alk_mol.mol_name]
        # print(nat_charge_dict)
        homo = orb_homo_lumo[0]
        lumo = orb_homo_lumo[1]
        rdf_dict = alkene_rdf_dict[alk_mol.mol_name]
        #Current order is correct (i.e. sub_0 = sub_0)
        if sub_0_connect == 'c1':
            nat_charge_1 = nat_charge_dict[c1]
            rdf_1_ser    = rdf_dict[0]
            rdf_1_ser.index = [f'{idx}_0' for idx in rdf_1_ser.index]
            nat_charge_2 = nat_charge_dict[c2]
            rdf_2_ser        = rdf_dict[1]
            rdf_2_ser.index = [f'{idx}_1' for idx in rdf_2_ser.index]
        #Current order needs to be inverted (i.e. sub_0 = sub_1)
        else:
            nat_charge_1 = nat_charge_dict[c2]
            rdf_1_ser    = rdf_dict[1]
            rdf_1_ser.index = [f'{idx}_0' for idx in rdf_1_ser.index]
            nat_charge_2 = nat_charge_dict[c1]
            rdf_2_ser        = rdf_dict[0]
            rdf_2_ser.index = [f'{idx}_1' for idx in rdf_2_ser.index]

        pert_vals = list()
        for (indices, value) in pert_final_list:
            if (c1 and c2 in indices):
                pert_vals.append(value)
        pert_arr = np.array(pert_vals)
        sort_pert_arr = pert_arr[np.argsort(pert_arr)[::-1]]
        top_4_pert_vals = sort_pert_arr[0:4]

        assert len(pert_vals) >= 3, f'pert_vals is {pert_vals}'

        for (bd_type, indices, bd_order, energy) in nbo_orb_final_list:
            if (bd_type == 'BD') and (bd_order == '2'):
                pi_orb_energy = energy
            elif (bd_type == 'BD*') and (bd_order == '2'):
                anti_pi_orb_energy = energy


        add_ser = {
            'homo': homo, 
            'lumo': lumo, 
            'nat_charge_1': nat_charge_1, 
            'nat_charge_2': nat_charge_2, 
            'pert_1': top_4_pert_vals[0], 
            'pert_2': top_4_pert_vals[1], 
            'pert_3': top_4_pert_vals[2],
            'pi_orb': pi_orb_energy, 
            'pi*_orb': anti_pi_orb_energy,
            }
        nbo_ser = pd.Series(add_ser)
        concat_df = pd.concat([sterimol_esp_val_0, sterimol_esp_val_1, sterimol_esp_val_2, sterimol_esp_val_3, nbo_ser, rdf_1_ser, rdf_2_ser], axis=0).to_frame(name = name)
        fix_df = concat_df.transpose()        
        full_df = pd.concat([full_df, fix_df], axis=0)

    if alk_mol.alkene_type == 'cis_di':
        '''
        Alignment will look like this based on a reorder by highest volume:
        Sub_0 | Sub_1
        ----------
           H  |   H

        '''
        if len(frag_tuples) < 2:
            atom_mol = atom_class(Chem.MolFromSmiles(react_smiles))
            #This checks to see if this has a deuterium in it, as it would not count as a fragment in my current workflow
            #At the moment, they are getting directly replaced with the dummy hydrogen\
            #Necessarily, the greater volume is going to be any non-Deuterium connection
            if np.count_nonzero(atom_mol.smarts_query('[2H]')) != 0:
                print(atom_mol.smarts_query('[2H]'))
                sub_name_0 = frag_tuples[0][1].GetProp("_Name")
                sub_name_1 = 'Deuterium'
                sub_0_connect = sub_name_0.split('_')[2]
                sterimol_esp_val_0 = create_sterimol_series(sub_name_0, frag_sterimol_esp, num=0)
                sterimol_esp_val_1 = create_sterimol_series('hydrogen_test', dummy_df, num=1)
                sterimol_esp_val_2 = create_sterimol_series('hydrogen_test', dummy_df, num=2)
                sterimol_esp_val_3 = create_sterimol_series('hydrogen_test', dummy_df, num=3)

                if sub_0_connect == 'c1':
                    sub_1_connect = 'c2'
                else:
                    sub_1_connect = 'c1'
            else:
                raise ValueError(f'react_smiles are {react_smiles}')
        else:
            #Asserts that there are 2 value in the list for a monosubstituted alkene
            assert len(frag_tuples) == 2, f'frag_tuples = {frag_tuples}'
            #Finds the name of the fragment for utilizing the dictionary
            test_sub_name_0 = frag_tuples[0][1].GetProp("_Name")
            test_sub_0_connect = test_sub_name_0.split('_')[2]
            test_sub_name_1 = frag_tuples[1][1].GetProp("_Name")
            test_sub_1_connect = test_sub_name_1.split('_')[2]
            assert test_sub_0_connect != test_sub_1_connect, f'sub_0 connected to sub_1 for cis_di: sub_0 = {test_sub_0_connect}, sub_1 = {test_sub_1_connect}'
            #Retrieves the series for the sterimol and esp descriptors for this alkene
            test_sterimol_esp_val_0 = create_sterimol_test_series(test_sub_name_0, frag_sterimol_esp)
            test_sterimol_esp_val_1 = create_sterimol_test_series(test_sub_name_1, frag_sterimol_esp)

            test_sterimol_vol_0 = test_sterimol_esp_val_0['vol']
            test_sterimol_vol_1 = test_sterimol_esp_val_1['vol']

            #Current order is correct (i.e. sub_0 = sub_0)
            if test_sterimol_vol_0 >= test_sterimol_vol_1:
                sub_name_0 = test_sub_name_0
                sub_0_connect = test_sub_0_connect
                sub_name_1 = test_sub_name_1
                sub_1_connect = test_sub_1_connect

            #Current order needs to be inverted (i.e. sub_0 = sub_1)
            else:
                sub_name_0 = test_sub_name_1
                sub_0_connect = test_sub_1_connect
                sub_name_1 = test_sub_name_0
                sub_1_connect = test_sub_0_connect

            sterimol_esp_val_0 = create_sterimol_series(sub_name_0, frag_sterimol_esp, num=0)
            sterimol_esp_val_1 = create_sterimol_series(sub_name_1, frag_sterimol_esp, num=1)
            sterimol_esp_val_2 = create_sterimol_series('hydrogen_test', dummy_df, num=2)
            sterimol_esp_val_3 = create_sterimol_series('hydrogen_test', dummy_df, num=3)
        try:
            orb_homo_lumo, nat_charge_dict, pert_final_list, nbo_orb_final_list = nbo_dict[alk_mol.mol_name]
        except:
            alk_mol.mol_name = f'{alk_mol.mol_name}_opt_freq_conf_6'
            orb_homo_lumo, nat_charge_dict, pert_final_list, nbo_orb_final_list = nbo_dict[alk_mol.mol_name]
        homo = orb_homo_lumo[0]
        lumo = orb_homo_lumo[1]
        rdf_dict = alkene_rdf_dict[alk_mol.mol_name]
        #Current order is correct (i.e. sub_0 = sub_0)

        #Note, this works because c1 is arbitrarily always the first value in the dictionary, the if else statement reverses it if necessary
        if sub_0_connect == 'c1':
            nat_charge_1 = nat_charge_dict[c1]
            rdf_1_ser    = rdf_dict[0]
            rdf_1_ser.index = [f'{idx}_0' for idx in rdf_1_ser.index]
            nat_charge_2 = nat_charge_dict[c2]
            rdf_2_ser        = rdf_dict[1]
            rdf_2_ser.index = [f'{idx}_1' for idx in rdf_2_ser.index]
        #Current order needs to be inverted (i.e. sub_0 = sub_1)
        else:
            nat_charge_1 = nat_charge_dict[c2]
            rdf_1_ser    = rdf_dict[1]
            rdf_1_ser.index = [f'{idx}_0' for idx in rdf_1_ser.index]
            nat_charge_2 = nat_charge_dict[c1]
            rdf_2_ser        = rdf_dict[0]
            rdf_2_ser.index = [f'{idx}_1' for idx in rdf_2_ser.index]

        pert_vals = list()
        for (indices, value) in pert_final_list:
            if (c1 and c2 in indices):
                pert_vals.append(value)
        pert_arr = np.array(pert_vals)
        sort_pert_arr = pert_arr[np.argsort(pert_arr)[::-1]]
        top_4_pert_vals = sort_pert_arr[0:4]

        assert len(pert_vals) >= 3, f'pert_vals is {pert_vals}'

        for (bd_type, indices, bd_order, energy) in nbo_orb_final_list:
            if (bd_type == 'BD') and (bd_order == '2'):
                pi_orb_energy = energy
            elif (bd_type == 'BD*') and (bd_order == '2'):
                anti_pi_orb_energy = energy


        add_ser = {
            'homo': homo, 
            'lumo': lumo, 
            'nat_charge_1': nat_charge_1, 
            'nat_charge_2': nat_charge_2, 
            'pert_1': top_4_pert_vals[0], 
            'pert_2': top_4_pert_vals[1], 
            'pert_3': top_4_pert_vals[2],
            'pi_orb': pi_orb_energy, 
            'pi*_orb': anti_pi_orb_energy,
            }
        

        nbo_ser = pd.Series(add_ser)
        
        concat_df = pd.concat([sterimol_esp_val_0, sterimol_esp_val_1, sterimol_esp_val_2, sterimol_esp_val_3, nbo_ser, rdf_1_ser, rdf_2_ser], axis=0).to_frame(name = name)
        fix_df = concat_df.transpose()        
        full_df = pd.concat([full_df, fix_df], axis=0)

    if alk_mol.alkene_type == 'trans_di':
        '''
        Alignment will look like this based on a reorder by highest volume:
        Sub_0 |   H
        ----------
           H  |  Sub_1

        '''
        #Asserts that there is only one value in the list for a monosubstituted alkene
        if len(frag_tuples) < 2:
            atom_mol = atom_class(Chem.MolFromSmiles(react_smiles))
            #This checks to see if this has a deuterium in it, as it would not count as a fragment in my current workflow
            #At the moment, they are getting directly replaced with the dummy hydrogen\
            #Necessarily, the greater volume is going to be any non-Deuterium connection
            if np.count_nonzero(atom_mol.smarts_query('[2H]')) != 0:
                print(atom_mol.smarts_query('[2H]'))
                sub_name_0 = frag_tuples[0][1].GetProp("_Name")
                sub_name_1 = 'Deuterium'
                sub_0_connect = sub_name_0.split('_')[2]
                sterimol_esp_val_0 = create_sterimol_series(sub_name_0, frag_sterimol_esp, num=0)
                sterimol_esp_val_1 = create_sterimol_series('hydrogen_test', dummy_df, num=1)
                sterimol_esp_val_2 = create_sterimol_series('hydrogen_test', dummy_df, num=2)
                sterimol_esp_val_3 = create_sterimol_series('hydrogen_test', dummy_df, num=3)

                if sub_0_connect == 'c1':
                    sub_1_connect = 'c2'
                else:
                    sub_1_connect = 'c1'
            else:
                raise ValueError(f'react_smiles are {react_smiles}')
        else:
            #Asserts that there are 2 value in the list for a monosubstituted alkene
            assert len(frag_tuples) == 2, f'frag_tuples = {frag_tuples}'
            #Finds the name of the fragment for utilizing the dictionary
            test_sub_name_0 = frag_tuples[0][1].GetProp("_Name")
            test_sub_0_connect = test_sub_name_0.split('_')[2]
            test_sub_name_1 = frag_tuples[1][1].GetProp("_Name")
            test_sub_1_connect = test_sub_name_1.split('_')[2]
            assert test_sub_0_connect != test_sub_1_connect, f'sub_0 connected to sub_1 for trans_di: sub_0 = {test_sub_0_connect}, sub_1 = {test_sub_1_connect}'
            #Retrieves the series for the sterimol and esp descriptors for this alkene
            test_sterimol_esp_val_0 = create_sterimol_test_series(test_sub_name_0, frag_sterimol_esp)
            test_sterimol_esp_val_1 = create_sterimol_test_series(test_sub_name_1, frag_sterimol_esp)

            test_sterimol_vol_0 = test_sterimol_esp_val_0['vol']
            test_sterimol_vol_1 = test_sterimol_esp_val_1['vol']

            #Current order is correct (i.e. sub_0 = sub_0)
            if test_sterimol_vol_0 >= test_sterimol_vol_1:
                sub_name_0 = test_sub_name_0
                sub_0_connect = test_sub_0_connect
                sub_name_1 = test_sub_name_1
                sub_1_connect = test_sub_1_connect


            #Current order needs to be inverted (i.e. sub_0 = sub_1)
            else:
                sub_name_0 = test_sub_name_1
                sub_0_connect = test_sub_1_connect
                sub_name_1 = test_sub_name_0
                sub_1_connect = test_sub_0_connect

            sterimol_esp_val_0 = create_sterimol_series(sub_name_0, frag_sterimol_esp, num=0)
            sterimol_esp_val_1 = create_sterimol_series('hydrogen_test', dummy_df, num=1)
            sterimol_esp_val_2 = create_sterimol_series(sub_name_1, frag_sterimol_esp, num=2)
            sterimol_esp_val_3 = create_sterimol_series('hydrogen_test', dummy_df, num=3)
        try:
            orb_homo_lumo, nat_charge_dict, pert_final_list, nbo_orb_final_list = nbo_dict[alk_mol.mol_name]
        except:
            alk_mol.mol_name = f'{alk_mol.mol_name}_opt_freq_conf_6'
            orb_homo_lumo, nat_charge_dict, pert_final_list, nbo_orb_final_list = nbo_dict[alk_mol.mol_name]
        # print(nat_charge_dict)
        homo = orb_homo_lumo[0]
        lumo = orb_homo_lumo[1]
        rdf_dict = alkene_rdf_dict[alk_mol.mol_name]
        if sub_0_connect == 'c1':
            nat_charge_1 = nat_charge_dict[c1]
            rdf_1_ser    = rdf_dict[0]
            rdf_1_ser.index = [f'{idx}_0' for idx in rdf_1_ser.index]
            nat_charge_2 = nat_charge_dict[c2]
            rdf_2_ser        = rdf_dict[1]
            rdf_2_ser.index = [f'{idx}_1' for idx in rdf_2_ser.index]
        #Current order needs to be inverted (i.e. sub_0 = sub_1)
        
        else:
            nat_charge_1 = nat_charge_dict[c2]
            rdf_1_ser    = rdf_dict[1]
            rdf_1_ser.index = [f'{idx}_0' for idx in rdf_1_ser.index]
            nat_charge_2 = nat_charge_dict[c1]
            rdf_2_ser        = rdf_dict[0]
            rdf_2_ser.index = [f'{idx}_1' for idx in rdf_2_ser.index]
        pert_vals = list()
        for (indices, value) in pert_final_list:
            if (c1 and c2 in indices):
                pert_vals.append(value)
        pert_arr = np.array(pert_vals)
        sort_pert_arr = pert_arr[np.argsort(pert_arr)[::-1]]
        top_4_pert_vals = sort_pert_arr[0:4]

        assert len(pert_vals) >= 3, f'pert_vals is {pert_vals}'

        for (bd_type, indices, bd_order, energy) in nbo_orb_final_list:
            if (bd_type == 'BD') and (bd_order == '2'):
                pi_orb_energy = energy
            elif (bd_type == 'BD*') and (bd_order == '2'):
                anti_pi_orb_energy = energy


        add_ser = {
            'homo': homo, 
            'lumo': lumo, 
            'nat_charge_1': nat_charge_1, 
            'nat_charge_2': nat_charge_2, 
            'pert_1': top_4_pert_vals[0], 
            'pert_2': top_4_pert_vals[1], 
            'pert_3': top_4_pert_vals[2],
            'pi_orb': pi_orb_energy, 
            'pi*_orb': anti_pi_orb_energy,
            }
        nbo_ser = pd.Series(add_ser)
        concat_df = pd.concat([sterimol_esp_val_0, sterimol_esp_val_1, sterimol_esp_val_2, sterimol_esp_val_3, nbo_ser, rdf_1_ser, rdf_2_ser], axis=0).to_frame(name = name)
        fix_df = concat_df.transpose()        
        full_df = pd.concat([full_df, fix_df], axis=0)

    if alk_mol.alkene_type == 'tri':
        if len(frag_tuples) != 3:
            #This if statement exists to deal with graph isomorphisms, i.e. a relic of Step 4 needing to be redone without dictionaries
            assert len(frag_tuples) == 2, f"There are less than 2 fragments, which shouldn't be possible, {frag_tuples}"
            test_sub_name_0 = frag_tuples[0][1].GetProp("_Name")
            test_sub_0_connect = test_sub_name_0.split('_')[2]
            test_sub_name_0 = frag_tuples[1][1].GetProp("_Name")
            test_sub_1_connect = test_sub_name_0.split('_')[2]

            #Retrieves the series for the sterimol and esp descriptors for this alkene
            test_sterimol_esp_val_0 = create_sterimol_test_series(test_sub_name_0, frag_sterimol_esp)
            test_sterimol_esp_val_1 = create_sterimol_test_series(test_sub_name_1, frag_sterimol_esp)
            sterimol_vals = np.array([test_sterimol_esp_val_0['vol'], test_sterimol_esp_val_1['vol']])

            max_vol_idx = np.where(sterimol_vals == np.max(sterimol_vals))[0][0]
            remaining_idx = np.where(sterimol_vals != np.max(sterimol_vals))[0]
            assert np.shape(remaining_idx)[0] == 1, f'remaining index is {remaining_idx}'
            #Defines 
            sub_name_0 = frag_tuples[max_vol_idx][1].GetProp("_Name")
            sub_0_connect = sub_name_0.split('_')[2]
            sterimol_esp_val_0 = create_sterimol_series(sub_name_0, frag_sterimol_esp, num=0)

            sub_name_1 = frag_tuples[remaining_idx[0]][1].GetProp("_Name")
            sub_1_connect = sub_name_1.split('_')[2]
            sterimol_esp_val_1 = create_sterimol_series(sub_name_1, frag_sterimol_esp, num=1)

            assert sub_0_connect != sub_1_connect, f'sub_0 is connected to {sub_0_connect} and sub_1 is connected to {sub_1_connect}!'
            
            if sub_0_connect == 'c1':
                high_vol_c1 = c1
                low_vol_c2 = c2
            else:
                high_vol_c1 = c2
                low_vol_c2 = c1
            
            #If hydrogen is in neighbor of the high volume c1, this implies the other carbon has the isomorphism.
            if 'H' in alk_mol.neighbor_symbol_list(high_vol_c1):
                '''
                H is connected gem_disubstituted, while the other substituents are unclear
                Sub0 | sub1
                ----------
                  H  | sub1
                '''
                sub_name_2 = sub_name_1
                sub_2_connect = sub_1_connect
                sterimol_esp_val_2 = create_sterimol_series(sub_name_2, frag_sterimol_esp, num=2)
                sterimol_esp_val_3 = create_sterimol_series('hydrogen_test', dummy_df, num=3)
            else:
                '''
                H is connected gem_disubstituted, but by symmetry rules, the two alkenes are interchangeable, It makes sense to match
                The sharpless mnemonic such that that the hydrogen is in the bottom orientation:
                

                Sub0 | sub1
                ----------
                sub0  | H

                or 

                Sub0 | H
                ----------
                sub0  | sub1
                '''
                sub_name_2 = sub_name_0
                sub_2_connect = sub_0_connect
                sterimol_esp_val_2 = create_sterimol_series('hydrogen_test', dummy_df, num=2)
                sterimol_esp_val_3 = create_sterimol_series(sub_name_2, frag_sterimol_esp, num=3)
            
        else:
            ac.Compute2DCoords(alk_mol.rdkit_alkene_mol)
            c = alk_mol.rdkit_alkene_mol.GetConformer()
            coords = c.GetPositions()
            test_sub_name_0 = frag_tuples[0][1].GetProp("_Name")
            test_sub_0_connect = test_sub_name_0.split('_')[2]
            test_sub_name_1 = frag_tuples[1][1].GetProp("_Name")
            test_sub_1_connect = test_sub_name_1.split('_')[2]
            test_sub_name_2 = frag_tuples[2][1].GetProp("_Name")
            test_sub_2_connect = test_sub_name_2.split('_')[2]

            #Retrieves the series for the sterimol and esp descriptors for this alkene
            test_sterimol_esp_val_0 = create_sterimol_test_series(test_sub_name_0, frag_sterimol_esp)
            test_sterimol_esp_val_1 = create_sterimol_test_series(test_sub_name_1, frag_sterimol_esp)
            test_sterimol_esp_val_2 = create_sterimol_test_series(test_sub_name_2, frag_sterimol_esp)

            sterimol_vals = np.array([test_sterimol_esp_val_0['vol'], test_sterimol_esp_val_1['vol'],test_sterimol_esp_val_2['vol']])
            val_names = np.array([test_sub_name_0, test_sub_name_1, test_sub_name_2])
            val_connect = np.array([test_sub_0_connect, test_sub_1_connect, test_sub_2_connect])
            
            max_vol_idx = np.where(sterimol_vals == np.max(sterimol_vals))[0][0]
            remaining_idx0, remaining_idx1 = np.where(sterimol_vals != np.max(sterimol_vals))[0]
            
            #Assigns the first quadrant as sub_name_0
            sub_name_0 = frag_tuples[max_vol_idx][1].GetProp("_Name")
            sub_0_connect = sub_name_0.split('_')[2]

            sterimol_esp_val_0 = create_sterimol_series(sub_name_0, frag_sterimol_esp, num=0)
            if sub_0_connect == 'c1':
                high_vol_c1 = c1
                low_vol_c2 = c2
            else:
                high_vol_c1 = c2
                low_vol_c2 = c1

            high_vol_c1_neighbor_idxs = alk_mol.neighbor_idx_list(
                atom_idx=high_vol_c1, 
                atom_idx_not_included=low_vol_c2
                )
            

            low_vol_c2_neighbor_idxs = alk_mol.neighbor_idx_list(
                atom_idx=low_vol_c2,
                atom_idx_not_included=high_vol_c1
            )

            assert len(high_vol_c1_neighbor_idxs) == 2, f'Length of sub_0_neighbor list is != 2 {high_vol_c1_neighbor_idxs}'
            
            #This is used to test
            sub0_test_idx0, sub0_test_idx1 = high_vol_c1_neighbor_idxs
            low_test_idx0, low_test_idx1 = low_vol_c2_neighbor_idxs
            
            test_sub_name_1 = val_names[remaining_idx0]
            test_sub_1_connect = val_connect[remaining_idx0]
            test_sub_name_2 = val_names[remaining_idx1]
            test_sub_2_connect = val_connect[remaining_idx1]

            if sub_0_connect == test_sub_1_connect:
                '''
                Test1 is connected to the same carbon of the highest volume substituent, Will ALWAYS have to be second with clockwise orientation:
                Sub0  | Test2/H
                ----------
                Test1 | Test2/H
                '''
                sterimol_esp_val_3 = create_sterimol_series(test_sub_name_1, frag_sterimol_esp, num=3)
                
                #This implies that remaining_idx1 is the test2 substrate
                sub_0_idx, _0_name, sub3_idx, _3_name = bfs_test_idx(
                    alk_mol=alk_mol, 
                    frag_tuples=frag_tuples,
                    test1_sub_connect = sub_0_connect,
                    test1_atom_idx = sub0_test_idx0,
                    tuple_idx0 = max_vol_idx,
                    test2_sub_connect = test_sub_1_connect,
                    test2_atom_idx = sub0_test_idx1,
                    tuple_idx1 = remaining_idx0,
                    high_vol_c1 = high_vol_c1,
                    low_vol_c2 = low_vol_c2
                    )

                #Now testing orientation of test1_idx and test2_idx
                test1_orientation, test1_idx, test2_idx = test_cis_trans(
                    alk_mol=alk_mol,
                    coords = coords,
                    sub_0_idx = sub_0_idx,
                    high_vol_c1 = high_vol_c1,
                    low_vol_c2 = low_vol_c2
                )

                assert (test1_idx != sub3_idx) and (test2_idx != sub3_idx), f'Test1 ({test1_idx}) and Test2 ({test2_idx}) should not be equal to sub3_idx {sub3_idx}'
                
                #If test1_idx = cis, the hydrogen should be in q2
                if test1_orientation == 'cis':
                    if alk_mol.atoms_dict[test1_idx].GetSymbol() == 'H':
                        sterimol_esp_val_1 = create_sterimol_series('hydrogen_test', dummy_df, num=1)
                        sterimol_esp_val_2 = create_sterimol_series(test_sub_name_2, frag_sterimol_esp, num=2)
                    else:
                        sterimol_esp_val_1 = create_sterimol_series(test_sub_name_2, frag_sterimol_esp, num=1)
                        sterimol_esp_val_2 = create_sterimol_series('hydrogen_test', dummy_df, num=2)

            elif sub_0_connect == test_sub_2_connect:
                '''
                Same rules as above, just Test2 is connected to the same carbon of the highest volume substituent
                '''

                sterimol_esp_val_3 = create_sterimol_series(test_sub_name_2, frag_sterimol_esp, num=3)
                
                #This implies that remaining_idx0 is the test2 substrate
                sub_0_idx, _0_name, sub3_idx, _3_name = bfs_test_idx(
                    alk_mol=alk_mol, 
                    frag_tuples=frag_tuples,
                    test1_sub_connect = sub_0_connect,
                    test1_atom_idx = sub0_test_idx0,
                    tuple_idx0 = max_vol_idx,
                    test2_sub_connect = test_sub_2_connect,
                    test2_atom_idx = sub0_test_idx1,
                    tuple_idx1 = remaining_idx1,
                    high_vol_c1 = high_vol_c1,
                    low_vol_c2 = low_vol_c2
                    )

                #Now testing orientation of test1_idx and test2_idx
                test1_orientation, test1_idx, test2_idx = test_cis_trans(
                    alk_mol=alk_mol,
                    coords = coords,
                    sub_0_idx = sub_0_idx,
                    high_vol_c1 = high_vol_c1,
                    low_vol_c2 = low_vol_c2
                )

                assert (test1_idx != sub3_idx) and (test2_idx != sub3_idx), f'Test1 ({test1_idx}) and Test2 ({test2_idx}) should not be equal to sub3_idx {sub3_idx}'
                
                #If test1_idx = cis, the hydrogen should be in q2
                if test1_orientation == 'cis':
                    if alk_mol.atoms_dict[test1_idx].GetSymbol() == 'H':
                        sterimol_esp_val_1 = create_sterimol_series('hydrogen_test', dummy_df, num=1)
                        sterimol_esp_val_2 = create_sterimol_series(test_sub_name_1, frag_sterimol_esp, num=2)
                    else:
                        sterimol_esp_val_1 = create_sterimol_series(test_sub_name_1, frag_sterimol_esp, num=1)
                        sterimol_esp_val_2 = create_sterimol_series('hydrogen_test', dummy_df, num=2)

            else:
                '''
                H is connected gem_disubstituted, while the other substituents are unclear
                Sub0 | Test2/Test3
                ----------
                  H  | Test2/Test3
                '''
                #This determines which carbon is the carbon where two fragments need to be differentiated
                #I now need to create a set of indices to draw vectors from:

                sterimol_esp_val_3 = create_sterimol_series('hydrogen_test', dummy_df, num=3)

                if alk_mol.atoms_dict[sub0_test_idx0].GetSymbol() != 'H':
                    sub_0_idx = sub0_test_idx0
                else:
                    sub_0_idx = sub0_test_idx1

                test1_idx, test1_sub_name, test2_idx, test2_sub_name = bfs_test_idx(
                    alk_mol=alk_mol,
                    frag_tuples=frag_tuples,
                    test1_sub_connect = test_sub_1_connect,
                    test1_atom_idx = low_test_idx0,
                    tuple_idx0=remaining_idx0,
                    test2_sub_connect = test_sub_2_connect,
                    test2_atom_idx=low_test_idx1,
                    tuple_idx1=remaining_idx1,
                    high_vol_c1=high_vol_c1,
                    low_vol_c2=low_vol_c2)
                
                test1_orientation, test1_idx, test2_idx = test_cis_trans_known_idx(
                    alk_mol=alk_mol,
                    coords = coords,
                    sub_0_idx = sub_0_idx,
                    low_vol_idx0=test1_idx,
                    low_vol_idx1=test2_idx,
                    high_vol_c1 = high_vol_c1
                )

                if test1_orientation == 'cis':
                    sterimol_esp_val_1 = create_sterimol_series(test1_sub_name, frag_sterimol_esp, num=1)
                    sterimol_esp_val_2 = create_sterimol_series(test2_sub_name, frag_sterimol_esp, num=2)
                else:
                    sterimol_esp_val_1 = create_sterimol_series(test2_sub_name, frag_sterimol_esp, num=1)
                    sterimol_esp_val_2 = create_sterimol_series(test1_sub_name, frag_sterimol_esp, num=2)
                    
        try:
            orb_homo_lumo, nat_charge_dict, pert_final_list, nbo_orb_final_list = nbo_dict[alk_mol.mol_name]
        except:
            alk_mol.mol_name = f'{alk_mol.mol_name}_opt_freq_conf_6'
            orb_homo_lumo, nat_charge_dict, pert_final_list, nbo_orb_final_list = nbo_dict[alk_mol.mol_name]
        # print(nat_charge_dict)
        homo = orb_homo_lumo[0]
        lumo = orb_homo_lumo[1]
        rdf_dict = alkene_rdf_dict[alk_mol.mol_name]
        if sub_0_connect == 'c1':
            nat_charge_1 = nat_charge_dict[c1]
            rdf_1_ser    = rdf_dict[0]
            rdf_1_ser.index = [f'{idx}_0' for idx in rdf_1_ser.index]
            nat_charge_2 = nat_charge_dict[c2]
            rdf_2_ser        = rdf_dict[1]
            rdf_2_ser.index = [f'{idx}_1' for idx in rdf_2_ser.index]
        #Current order needs to be inverted (i.e. sub_0 = sub_1)
        else:
            nat_charge_1 = nat_charge_dict[c2]
            rdf_1_ser    = rdf_dict[1]
            rdf_1_ser.index = [f'{idx}_0' for idx in rdf_1_ser.index]
            nat_charge_2 = nat_charge_dict[c1]
            rdf_2_ser        = rdf_dict[0]
            rdf_2_ser.index = [f'{idx}_1' for idx in rdf_2_ser.index]

        pert_vals = list()
        for (indices, value) in pert_final_list:
            if (c1 and c2 in indices):
                pert_vals.append(value)
        pert_arr = np.array(pert_vals)
        sort_pert_arr = pert_arr[np.argsort(pert_arr)[::-1]]
        top_4_pert_vals = sort_pert_arr[0:4]

        assert len(pert_vals) >= 3, f'pert_vals is {pert_vals}'

        for (bd_type, indices, bd_order, energy) in nbo_orb_final_list:
            if (bd_type == 'BD') and (bd_order == '2'):
                pi_orb_energy = energy
            elif (bd_type == 'BD*') and (bd_order == '2'):
                anti_pi_orb_energy = energy


        add_ser = {
            'homo': homo, 
            'lumo': lumo, 
            'nat_charge_1': nat_charge_1, 
            'nat_charge_2': nat_charge_2, 
            'pert_1': top_4_pert_vals[0], 
            'pert_2': top_4_pert_vals[1], 
            'pert_3': top_4_pert_vals[2],
            'pi_orb': pi_orb_energy, 
            'pi*_orb': anti_pi_orb_energy,
            }
        nbo_ser = pd.Series(add_ser)
        concat_df = pd.concat([sterimol_esp_val_0, sterimol_esp_val_1, sterimol_esp_val_2, sterimol_esp_val_3, nbo_ser, rdf_1_ser, rdf_2_ser], axis=0).to_frame(name = name)
        fix_df = concat_df.transpose()        
        full_df = pd.concat([full_df, fix_df], axis=0)

    if alk_mol.alkene_type == 'tetra':
        # continue
        if len(frag_tuples) != 4:
            assert len(frag_tuples) == 3, f'There are less than 3 fragments, which I have not implemented a method for'
            test_sub_name_0 = frag_tuples[0][1].GetProp("_Name")
            test_sub_0_connect = test_sub_name_0.split('_')[2]
            test_sub_name_1 = frag_tuples[1][1].GetProp("_Name")
            test_sub_1_connect = test_sub_name_1.split('_')[2]
            test_sub_name_2 = frag_tuples[2][1].GetProp("_Name")
            test_sub_2_connect = test_sub_name_2.split('_')[2]

            #Retrieves the series for the sterimol and esp descriptors for this alkene
            test_sterimol_esp_val_0 = create_sterimol_test_series(test_sub_name_0, frag_sterimol_esp)
            test_sterimol_esp_val_1 = create_sterimol_test_series(test_sub_name_1, frag_sterimol_esp)
            test_sterimol_esp_val_2 = create_sterimol_test_series(test_sub_name_2, frag_sterimol_esp)

            sterimol_vals = np.array([test_sterimol_esp_val_0['vol'], test_sterimol_esp_val_1['vol'],test_sterimol_esp_val_2['vol']])
            val_names = np.array([test_sub_name_0, test_sub_name_1, test_sub_name_2])
            val_connect = np.array([test_sub_0_connect, test_sub_1_connect, test_sub_2_connect])

            max_vol_idx = np.where(sterimol_vals == np.max(sterimol_vals))[0][0]
            remaining_idx_arr = np.where(sterimol_vals != np.max(sterimol_vals))[0]

            #Assigns the first quadrant as sub_name_0
            sub_name_0 = frag_tuples[max_vol_idx][1].GetProp("_Name")
            sub_0_connect = sub_name_0.split('_')[2]

            sterimol_esp_val_0 = create_sterimol_series(sub_name_0, frag_sterimol_esp, num=0)

            if sub_0_connect == 'c1':
                high_vol_c1 = c1
                low_vol_c2 = c2
            else:
                high_vol_c1 = c2
                low_vol_c2 = c1

            remaining_list = list()
            for idx in remaining_idx_arr:
                full_frag_name = frag_tuples[idx][1].GetProp("_Name")
                if sub_0_connect == full_frag_name.split('_')[2]:
                    sub_name_3 = full_frag_name
                    sterimol_esp_val_3 = create_sterimol_series(sub_name_3, frag_sterimol_esp, num=3)
                else:
                    remaining_list.append(idx)

            #This implies that sub_name_3 is connected to the same carbon as sub0, so the missing value is on the other carbon
            if len(remaining_list) == 1:
                sub_name_1 = frag_tuples[remaining_list[0]][1].GetProp("_Name")
                sterimol_esp_val_1 = create_sterimol_series(sub_name_1, frag_sterimol_esp, num=1)
                sterimol_esp_val_2 = create_sterimol_series(sub_name_1, frag_sterimol_esp, num=2)
            else:
                raise ValueError('Not Implemented a Check if the Isomorph is on the same carbon as the "High volume C1"')
          

            #This if statement exists to deal with graph isomorphisms, i.e. a relic of Step 4 needing to be redone without dictionaries
            # assert len(frag_tuples) == 2, f"There are less than 2 fragments, which shouldn't be possible, {frag_tuples}"
        else:
            ac.Compute2DCoords(alk_mol.rdkit_alkene_mol)
            c = alk_mol.rdkit_alkene_mol.GetConformer()
            coords = c.GetPositions()
            test_sub_name_0 = frag_tuples[0][1].GetProp("_Name")
            test_sub_0_connect = test_sub_name_0.split('_')[2]
            test_sub_name_1 = frag_tuples[1][1].GetProp("_Name")
            test_sub_1_connect = test_sub_name_1.split('_')[2]
            test_sub_name_2 = frag_tuples[2][1].GetProp("_Name")
            test_sub_2_connect = test_sub_name_2.split('_')[2]
            test_sub_name_3 = frag_tuples[3][1].GetProp("_Name")
            test_sub_3_connect = test_sub_name_3.split('_')[2]

            #Retrieves the series for the sterimol and esp descriptors for this alkene
            test_sterimol_esp_val_0 = create_sterimol_test_series(test_sub_name_0, frag_sterimol_esp)
            test_sterimol_esp_val_1 = create_sterimol_test_series(test_sub_name_1, frag_sterimol_esp)
            test_sterimol_esp_val_2 = create_sterimol_test_series(test_sub_name_2, frag_sterimol_esp)
            test_sterimol_esp_val_3 = create_sterimol_test_series(test_sub_name_3, frag_sterimol_esp)

            sterimol_vals = np.array([test_sterimol_esp_val_0['vol'], test_sterimol_esp_val_1['vol'],test_sterimol_esp_val_2['vol'], test_sterimol_esp_val_3['vol']])
            val_names = np.array([test_sub_name_0, test_sub_name_1, test_sub_name_2, test_sub_name_3])
            val_connect = np.array([test_sub_0_connect, test_sub_1_connect, test_sub_2_connect, test_sub_3_connect])

            max_vol_idx = np.where(sterimol_vals == np.max(sterimol_vals))[0][0]
            remaining_idx_arr = np.where(sterimol_vals != np.max(sterimol_vals))[0]

            sub_name_0 = frag_tuples[max_vol_idx][1].GetProp("_Name")
            sub_0_connect = sub_name_0.split('_')[2]
            sterimol_esp_val_0 = create_sterimol_series(sub_name_0, frag_sterimol_esp, num=0)

            if sub_0_connect == 'c1':
                high_vol_c1 = c1
                low_vol_c2 = c2
            else:
                high_vol_c1 = c2
                low_vol_c2 = c1


            #Defines the index of the fragment connected to the same carbon as sub_0
            remaining_list = list()
            for idx in remaining_idx_arr:
                full_frag_name = frag_tuples[idx][1].GetProp("_Name")
                if sub_0_connect == full_frag_name.split('_')[2]:
                    sub0_tuple_idx = idx
                    sub_name_3 = full_frag_name
                    sterimol_esp_val_3 = create_sterimol_series(sub_name_3, frag_sterimol_esp, num=3)
                else:
                    remaining_list.append(idx)

            assert len(remaining_list) == 2, f'Remaining List not correct! {remaining_list}'

            #These are going to be needed to figure out which fragment sub0 actually is
            sub0_test_idx0, sub0_test_idx1 = alk_mol.neighbor_idx_list(atom_idx=high_vol_c1, atom_idx_not_included=low_vol_c2)

            #Already determined that these two frags are connected, so test2 is just the same carbon as sub0
            sub_0_idx, _0_name, sub3_idx, _3_name = bfs_test_idx(
                alk_mol=alk_mol, 
                frag_tuples=frag_tuples,
                test1_sub_connect = sub_0_connect,
                test1_atom_idx = sub0_test_idx0,
                tuple_idx0 = max_vol_idx,
                test2_sub_connect = sub_0_connect,
                test2_atom_idx = sub0_test_idx1,
                tuple_idx1 = sub0_tuple_idx,
                high_vol_c1 = high_vol_c1,
                low_vol_c2 = low_vol_c2
                )

            low_test_idx0, low_test_idx1 = alk_mol.neighbor_idx_list(atom_idx=low_vol_c2, atom_idx_not_included=high_vol_c1)

            remaining_idx0, remaining_idx1 = remaining_list

            test_sub_1_connect = val_connect[remaining_idx0]
            test_sub_2_connect = val_connect[remaining_idx1]

            test1_idx, test1_sub_name, test2_idx, test2_sub_name = bfs_test_idx(
                alk_mol=alk_mol,
                frag_tuples=frag_tuples,
                test1_sub_connect = test_sub_1_connect,
                test1_atom_idx = low_test_idx0,
                tuple_idx0=remaining_idx0,
                test2_sub_connect = test_sub_2_connect,
                test2_atom_idx=low_test_idx1,
                tuple_idx1=remaining_idx1,
                high_vol_c1=high_vol_c1,
                low_vol_c2=low_vol_c2)
            
            test1_orientation, test1_idx, test2_idx = test_cis_trans_known_idx(
                alk_mol=alk_mol,
                coords = coords,
                sub_0_idx = sub_0_idx,
                low_vol_idx0=test1_idx,
                low_vol_idx1=test2_idx,
                high_vol_c1 = high_vol_c1,
            )

            if test1_orientation == 'cis':
                sterimol_esp_val_1 = create_sterimol_series(test1_sub_name, frag_sterimol_esp, num=1)
                sterimol_esp_val_2 = create_sterimol_series(test2_sub_name, frag_sterimol_esp, num=2)
            else:
                sterimol_esp_val_1 = create_sterimol_series(test2_sub_name, frag_sterimol_esp, num=1)
                sterimol_esp_val_2 = create_sterimol_series(test1_sub_name, frag_sterimol_esp, num=2)

        try:
            orb_homo_lumo, nat_charge_dict, pert_final_list, nbo_orb_final_list = nbo_dict[alk_mol.mol_name]
        except:
            alk_mol.mol_name = f'{alk_mol.mol_name}_opt_freq_conf_6'
            orb_homo_lumo, nat_charge_dict, pert_final_list, nbo_orb_final_list = nbo_dict[alk_mol.mol_name]
        # print(nat_charge_dict)
        homo = orb_homo_lumo[0]
        lumo = orb_homo_lumo[1]
        rdf_dict = alkene_rdf_dict[alk_mol.mol_name]
        if sub_0_connect == 'c1':
            nat_charge_1 = nat_charge_dict[c1]
            rdf_1_ser    = rdf_dict[0]
            rdf_1_ser.index = [f'{idx}_0' for idx in rdf_1_ser.index]
            nat_charge_2 = nat_charge_dict[c2]
            rdf_2_ser        = rdf_dict[1]
            rdf_2_ser.index = [f'{idx}_1' for idx in rdf_2_ser.index]
        #Current order needs to be inverted (i.e. sub_0 = sub_1)
        else:
            nat_charge_1 = nat_charge_dict[c2]
            rdf_1_ser    = rdf_dict[1]
            rdf_1_ser.index = [f'{idx}_0' for idx in rdf_1_ser.index]
            nat_charge_2 = nat_charge_dict[c1]
            rdf_2_ser        = rdf_dict[0]
            rdf_2_ser.index = [f'{idx}_1' for idx in rdf_2_ser.index]

        pert_vals = list()
        for (indices, value) in pert_final_list:
            if (c1 and c2 in indices):
                pert_vals.append(value)
        pert_arr = np.array(pert_vals)
        sort_pert_arr = pert_arr[np.argsort(pert_arr)[::-1]]
        top_4_pert_vals = sort_pert_arr[0:4]

        assert len(pert_vals) >= 3, f'pert_vals is {pert_vals}'

        for (bd_type, indices, bd_order, energy) in nbo_orb_final_list:
            if (bd_type == 'BD') and (bd_order == '2'):
                pi_orb_energy = energy
            elif (bd_type == 'BD*') and (bd_order == '2'):
                anti_pi_orb_energy = energy


        add_ser = {
            'homo': homo, 
            'lumo': lumo, 
            'nat_charge_1': nat_charge_1, 
            'nat_charge_2': nat_charge_2, 
            'pert_1': top_4_pert_vals[0], 
            'pert_2': top_4_pert_vals[1], 
            'pert_3': top_4_pert_vals[2],
            'pi_orb': pi_orb_energy, 
            'pi*_orb': anti_pi_orb_energy,
            }
        nbo_ser = pd.Series(add_ser)
        concat_df = pd.concat([sterimol_esp_val_0, sterimol_esp_val_1, sterimol_esp_val_2, sterimol_esp_val_3, nbo_ser, rdf_1_ser, rdf_2_ser], axis=0).to_frame(name = name)
        fix_df = concat_df.transpose()        
        full_df = pd.concat([full_df, fix_df], axis=0)

print(full_df)
full_df.to_csv('Step_6_Full_Aligned_React_Desc_DF.csv')