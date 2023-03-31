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

def create_sterimol_series(sub_name:'str', sterimol_esp_df, num):
    sterimol_esp_val = sterimol_esp_df.loc[sub_name]
    sterimol_esp_val.index = [f'{idx}_{num}' for idx in sterimol_esp_val.index]
    return sterimol_esp_val

def eval_frag_orientation():

    q2 = False
    q3 = False
    q4 = False



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

mono_df = pd.DataFrame()
gem_di_df = pd.DataFrame()
cis_di_df = pd.DataFrame()
trans_di_df = pd.DataFrame()
tri_df = pd.DataFrame()
tetra_df = pd.DataFrame()

problematic_tuples = list()
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
        #Asserts that there is only one value in the list for a monosubstituted alkene
        assert len(frag_tuples) == 1, f'frag_tuples = {frag_tuples}'
        #Finds the name of the fragment for utilizing the dictionary
        sub_name = frag_tuples[0][1].GetProp("_Name")
        sub_connect = sub_name.split('_')[2]
        #Retrieves the series for the sterimol and esp descriptors for this alkene
        sterimol_esp_vals = frag_sterimol_esp.loc[sub_name]
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
        concat_df = pd.concat([sterimol_esp_vals, nbo_ser, rdf_1_ser, rdf_2_ser], axis=0).to_frame(name = name)
        fix_df = concat_df.transpose()
        mono_df = pd.concat([mono_df, fix_df], axis=0)

    if alk_mol.alkene_type == 'gem_di':
        #Asserts that there is only one value in the list for a monosubstituted alkene
        assert len(frag_tuples) == 2, f'frag_tuples = {frag_tuples}'
        #Finds the name of the fragment for utilizing the dictionary
        sub_name_0 = frag_tuples[0][1].GetProp("_Name")
        sub_name_1 = frag_tuples[1][1].GetProp("_Name")
        sub_0_connect = sub_name_0.split('_')[2]
        sub_1_connect = sub_name_1.split('_')[2]
        assert sub_0_connect == sub_1_connect, f'sub_0 = {sub_0_connect}, sub_1 = {sub_1_connect}'
        #Retrieves the series for the sterimol and esp descriptors for this alkene
        sterimol_esp_val_0 = frag_sterimol_esp.loc[sub_name_0]
        sterimol_esp_val_1 = frag_sterimol_esp.loc[sub_name_1]

        sterimol_vol_0 = sterimol_esp_val_0['vol']
        sterimol_vol_1 = sterimol_esp_val_1['vol']

        #Current order is correct (i.e. sub_0 = sub_0)
        if sterimol_vol_0 >= sterimol_vol_1:
            sterimol_esp_val_0.index = [f'{idx}_0' for idx in sterimol_esp_val_0.index]
            sterimol_esp_val_1.index = [f'{idx}_1' for idx in sterimol_esp_val_1.index]
        #Current order needs to be inverted (i.e. sub_0 = sub_1)
        else:
            sub_name_0 = frag_tuples[1][1].GetProp("_Name")
            sterimol_esp_val_0 = frag_sterimol_esp.loc[sub_name_0]
            sterimol_esp_val_0.index = [f'{idx}_0' for idx in sterimol_esp_val_0.index]
            sub_name_1 = frag_tuples[0][1].GetProp("_Name")
            sterimol_esp_val_1 = frag_sterimol_esp.loc[sub_name_1]
            sterimol_esp_val_1.index = [f'{idx}_1' for idx in sterimol_esp_val_1.index]
        # print(sterimol_esp_val_0)
        # print(sterimol_esp_val_1)
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

        # print(rdf_dict[0])
        # print(rdf_dict[1])
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
        concat_df = pd.concat([sterimol_esp_val_0, sterimol_esp_val_1, nbo_ser, rdf_1_ser, rdf_2_ser], axis=0).to_frame(name = name)
        fix_df = concat_df.transpose()        
        gem_di_df = pd.concat([gem_di_df, fix_df], axis=0)

    if alk_mol.alkene_type == 'cis_di':
        if len(frag_tuples) < 2:
            atom_mol = atom_class(Chem.MolFromSmiles(react_smiles))
            if np.count_nonzero(atom_mol.smarts_query('[2H]')) != 0:
                print(atom_mol.smarts_query('[2H]'))
                sub_name_0 = frag_tuples[0][1].GetProp("_Name")
                sub_name_1 = 'Deuterium'
                sub_0_connect = sub_name_0.split('_')[2]
                sterimol_esp_val_0 = frag_sterimol_esp.loc[sub_name_0]
                sterimol_esp_val_0.index = [f'{idx}_0' for idx in sterimol_esp_val_0.index]
                sterimol_esp_val_1 = dummy_df.loc['hydrogen_test']
                sterimol_esp_val_1.index = [f'{idx}_1' for idx in sterimol_esp_val_1.index]

                if sub_0_connect == 'c1':
                    sub_1_connect = 'c2'
                elif sub_0_connect == 'c2':
                    sub_1_connect = 'c1'
            else:
                raise ValueError(f'react_smiles are {react_smiles}')
        else:
            # assert len(frag_tuples) == 2, f'frag_tuples = {frag_tuples}'
            #Finds the name of the fragment for utilizing the dictionary
            sub_name_0 = frag_tuples[0][1].GetProp("_Name")
            sub_name_1 = frag_tuples[1][1].GetProp("_Name")
            sub_0_connect = sub_name_0.split('_')[2]
            sub_1_connect = sub_name_1.split('_')[2]
            #Retrieves the series for the sterimol and esp descriptors for this alkene
            sterimol_esp_val_0 = frag_sterimol_esp.loc[sub_name_0]
            sterimol_esp_val_1 = frag_sterimol_esp.loc[sub_name_1]

            sterimol_vol_0 = sterimol_esp_val_0['vol']
            sterimol_vol_1 = sterimol_esp_val_1['vol']

        #Current order is correct (i.e. sub_0 = sub_0)
            if sterimol_vol_0 >= sterimol_vol_1:
                sterimol_esp_val_0.index = [f'{idx}_0' for idx in sterimol_esp_val_0.index]
                sterimol_esp_val_1.index = [f'{idx}_1' for idx in sterimol_esp_val_1.index]
            #Current order needs to be inverted (i.e. sub_0 = sub_1)
            else:
                #Renames what is considered the "first" carbon (note: this will be used in for RDF and natural charge)
                sub_name_0 = frag_tuples[1][1].GetProp("_Name")
                sub_0_connect = sub_name_1.split('_')[2]
                sterimol_esp_val_0 = frag_sterimol_esp.loc[sub_name_0]
                sterimol_esp_val_0.index = [f'{idx}_0' for idx in sterimol_esp_val_0.index]
                sub_name_1 = frag_tuples[0][1].GetProp("_Name")
                sub_1_connect = sub_name_0.split('_')[2]
                sterimol_esp_val_1 = frag_sterimol_esp.loc[sub_name_1]
                sterimol_esp_val_1.index = [f'{idx}_1' for idx in sterimol_esp_val_1.index]
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
        
        concat_df = pd.concat([sterimol_esp_val_0, sterimol_esp_val_1, nbo_ser, rdf_1_ser, rdf_2_ser], axis=0).to_frame(name = name)
        fix_df = concat_df.transpose()        
        cis_di_df = pd.concat([cis_di_df, fix_df], axis=0)

    if alk_mol.alkene_type == 'trans_di':
        #Asserts that there is only one value in the list for a monosubstituted alkene
        if len(frag_tuples) < 2:
            atom_mol = atom_class(Chem.MolFromSmiles(react_smiles))
            if np.count_nonzero(atom_mol.smarts_query('[2H]')) != 0:
                sub_name_0 = frag_tuples[0][1].GetProp("_Name")
                sub_name_1 = 'Deuterium'
                sub_0_connect = sub_name_0.split('_')[2]
                sterimol_esp_val_0 = frag_sterimol_esp.loc[sub_name_0]
                sterimol_esp_val_0.index = [f'{idx}_0' for idx in sterimol_esp_val_0.index]
                sterimol_esp_val_1 = dummy_df.loc['hydrogen_test']
                sterimol_esp_val_1.index = [f'{idx}_1' for idx in sterimol_esp_val_1.index]
                if sub_0_connect == 'c1':
                    sub_1_connect = 'c2'
                elif sub_0_connect == 'c2':
                    sub_1_connect = 'c1'
            else:
                raise ValueError(f'react_smiles are {react_smiles}')

        else:
            # assert len(frag_tuples) == 2, f'frag_tuples = {frag_tuples}'
            #Finds the name of the fragment for utilizing the dictionary
            sub_name_0 = frag_tuples[0][1].GetProp("_Name")
            sub_name_1 = frag_tuples[1][1].GetProp("_Name")
            sub_0_connect = sub_name_0.split('_')[2]
            sub_1_connect = sub_name_1.split('_')[2]
            #Retrieves the series for the sterimol and esp descriptors for this alkene
            sterimol_esp_val_0 = frag_sterimol_esp.loc[sub_name_0]
            sterimol_esp_val_1 = frag_sterimol_esp.loc[sub_name_1]

            sterimol_vol_0 = sterimol_esp_val_0['vol']
            sterimol_vol_1 = sterimol_esp_val_1['vol']

        #Current order is correct (i.e. sub_0 = sub_0)
            if sterimol_vol_0 >= sterimol_vol_1:
                sterimol_esp_val_0.index = [f'{idx}_0' for idx in sterimol_esp_val_0.index]
                sterimol_esp_val_1.index = [f'{idx}_1' for idx in sterimol_esp_val_1.index]
            #Current order needs to be inverted (i.e. sub_0 = sub_1)
            else:
                #Renames what is considered the "first" carbon (note: this will be used in for RDF and natural charge)
                sub_name_0 = frag_tuples[1][1].GetProp("_Name")
                sub_0_connect = sub_name_1.split('_')[2]
                sterimol_esp_val_0 = frag_sterimol_esp.loc[sub_name_0]
                sterimol_esp_val_0.index = [f'{idx}_0' for idx in sterimol_esp_val_0.index]
                sub_name_1 = frag_tuples[0][1].GetProp("_Name")
                sub_1_connect = sub_name_0.split('_')[2]
                sterimol_esp_val_1 = frag_sterimol_esp.loc[sub_name_1]
                sterimol_esp_val_1.index = [f'{idx}_1' for idx in sterimol_esp_val_1.index]
        # print(sterimol_esp_val_0)
        # print(sterimol_esp_val_1)
        #Current order is correct (i.e. sub_0 = sub_0)
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
        concat_df = pd.concat([sterimol_esp_val_0, sterimol_esp_val_1, nbo_ser, rdf_1_ser, rdf_2_ser], axis=0).to_frame(name = name)
        fix_df = concat_df.transpose()        
        trans_di_df = pd.concat([trans_di_df, fix_df], axis=0)

    if alk_mol.alkene_type == 'tri':
        if len(frag_tuples) != 3:
            #This if statement exists to deal with graph isomorphisms, i.e. a relic of Step 4 needing to be redone without dictionaries
            assert len(frag_tuples) == 2, f"There are less than 2 fragments, which shouldn't be possible, {frag_tuples}"
            sub_name_0 = frag_tuples[0][1].GetProp("_Name")
            sub_0_connect = sub_name_0.split('_')[2]
            sub_name_1 = frag_tuples[1][1].GetProp("_Name")
            sub_1_connect = sub_name_1.split('_')[2]

            sterimol_esp_val_0 = frag_sterimol_esp.loc[sub_name_0]
            sterimol_esp_val_1 = frag_sterimol_esp.loc[sub_name_1]
            sterimol_vals = np.array([sterimol_esp_val_0['vol'], sterimol_esp_val_1['vol']])

            max_vol_idx = np.where(sterimol_vals == np.max(sterimol_vals))[0][0]
            remaining_idx = np.where(sterimol_vals != np.max(sterimol_vals))[0]
            assert np.shape(remaining_idx)[0] == 1, f'remaining index is {remaining_idx}'
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
                sub_name_2 = sub_name_1
                sub_2_connect = sub_1_connect
                sterimol_esp_val_2 = create_sterimol_series(sub_name_2, frag_sterimol_esp, num=2)
            else:
                sub_name_2 = sub_name_0
                sub_2_connect = sub_0_connect
                sterimol_esp_val_2 = create_sterimol_series(sub_name_2, frag_sterimol_esp, num=2)
            
        else:
            sub_name_0 = frag_tuples[0][1].GetProp("_Name")
            sub_name_1 = frag_tuples[1][1].GetProp("_Name")
            sub_name_2 = frag_tuples[2][1].GetProp("_Name")

            #Retrieves the series for the sterimol and esp descriptors for this alkene
            sterimol_esp_val_0 = frag_sterimol_esp.loc[sub_name_0]
            sterimol_esp_val_1 = frag_sterimol_esp.loc[sub_name_1]
            sterimol_esp_val_2 = frag_sterimol_esp.loc[sub_name_2]

            sterimol_vals = np.array([sterimol_esp_val_0['vol'], sterimol_esp_val_1['vol'],sterimol_esp_val_2['vol']])

            max_vol_idx = np.where(sterimol_vals == np.max(sterimol_vals))[0][0]
            remaining_idx0, remaining_idx1 = np.where(sterimol_vals != np.max(sterimol_vals))[0]
            # descending_sort_arr = np.argsort(sterimol_vals)[::-1]
            
            sub_name_0 = frag_tuples[max_vol_idx][1].GetProp("_Name")
            sub_0_connect = sub_name_0.split('_')[2]

            sterimol_esp_val_0 = create_sterimol_series(sub_name_0, frag_sterimol_esp, num=0)

            #This reorganizes the sterimol values in clockwise order (the first value should never be attached)
            # for idx in remaining_idx:

            test1_sub_name = frag_tuples[remaining_idx0][1].GetProp("_Name")
            test1_sub_connect = test1_sub_name.split('_')[2]
            test2_sub_name = frag_tuples[remaining_idx1][1].GetProp("_Name")
            test2_sub_connect = test2_sub_name.split('_')[2]
            if sub_0_connect == test1_sub_connect:
                '''
                Test1 is connected to the same carbon of the highest volume substituent, Will ALWAYS have to be second with clockwise orientation:
                Sub0  | Test2/H
                ----------
                Test1 | Test2/H
                '''
                sterimol_esp_val_1 = create_sterimol_series(test2_sub_name, frag_sterimol_esp, num=1)
                sterimol_esp_val_2 = create_sterimol_series(test1_sub_name, frag_sterimol_esp, num=2)
            elif sub_0_connect == test2_sub_connect:
                '''
                Same rules as above, just Test2 is connected to the same carbon of the highest volume substituent
                '''
                sterimol_esp_val_1 = create_sterimol_series(test1_sub_name, frag_sterimol_esp, num=1)
                sterimol_esp_val_2 = create_sterimol_series(test2_sub_name, frag_sterimol_esp, num=2)
            else:
                '''
                H is connected gem_disubstituted, while the other substituents are unclear
                Sub0 | Test2/Test3
                ----------
                  H  | Test2/Test3
                '''
                #This determines which carbon is the carbon where two fragments need to be differentiated
                #I now need to create a set of indices to draw vectors from:
                if sub_0_connect == 'c1':
                    high_vol_c1 = c1
                    low_vol_c2 = c2
                else:
                    high_vol_c1 = c2
                    low_vol_c2 = c1

                sub_0_neighbor_list = alk_mol.neighbor_idx_list(
                    atom_idx=high_vol_c1, 
                    atom_idx_not_included=low_vol_c2
                    )
                
                assert len(sub_0_neighbor_list) == 2, f'Length of sub_0_neighbor_list is not 2: {sub_0_neighbor_list}'
                
                if alk_mol.atoms_dict[sub_0_neighbor_list[0]].GetSymbol() != 'H':
                    sub_0_idx = sub_0_neighbor_list[0]
                else:
                    sub_0_idx = sub_0_neighbor_list[1]

                ac.Compute2DCoords(alk_mol.rdkit_alkene_mol)
                c = alk_mol.rdkit_alkene_mol.GetConformer()
                coords = c.GetPositions()

                sub_0_coords = coords[sub_0_idx]
                high_vol_c1_coords = coords[high_vol_c1]
                #Defines first vector for cis or trans substituent
                v1_vec = high_vol_c1_coords-sub_0_coords

                low_vol_c2_neighbor_list = alk_mol.neighbor_idx_list(
                    atom_idx=low_vol_c2, 
                    atom_idx_not_included=high_vol_c1
                    )

                assert len(low_vol_c2_neighbor_list) == 2, f'Length of sub_0_neighbor_list is not 2: {sub_0_neighbor_list}'

                test1_idx = low_vol_c2_neighbor_list[0]
                test2_idx = low_vol_c2_neighbor_list[1]

                test1_coords = coords[test1_idx]
                test2_coords = coords[test2_idx]

                v2_vec = test2_coords-test1_coords

                '''
                v1 = vector from largest substituent first idx to high_vol_c1_idx 
                v2 = vector from a test1 -> test 2               
                '''
                if np.sign(np.dot(v1_vec,v2_vec)) == 1:
                    test1 = 'cis'
                else:
                    test1 = 'trans'
                Chem.SmilesParserParams.removeHs = False
                test1_mol = Chem.MolFromSmiles(frag_tuples[remaining_idx0][0], Chem.SmilesParserParams.removeHs)
                test1_mol_atoms_sym = [atom.GetSymbol() for atom in test1_mol.GetAtoms()]
                test1_mol_atoms_sym_no_wildcard = [sym for sym in test1_mol_atoms_sym if sym != '*']
                test1_aromatic_atoms_no_wildcard = np.array([atom.GetIsAromatic() for i,atom in enumerate(test1_mol.GetAtoms()) if i != 0])
                test1_bfsd = list(yield_bfsd(
                    alk_mol=alk_mol,
                    start_idx=test1_idx,
                    no_visit_c1 = high_vol_c1,
                    no_visit_c2 = low_vol_c2
                ))

                test2_bfsd = list(yield_bfsd(
                    alk_mol=alk_mol,
                    start_idx=test2_idx,
                    no_visit_c1 = high_vol_c1,
                    no_visit_c2 = low_vol_c2
                ))
                test2_mol = Chem.MolFromSmiles(frag_tuples[remaining_idx1][0], Chem.SmilesParserParams.removeHs)
                test2_mol_atoms_sym = [atom.GetSymbol() for atom in test2_mol.GetAtoms()]

                #This is for dealing with unique boundary cases in the graph that wouldn't be caught by assuming the lengths are the same
                if len(test1_bfsd) == len(test2_bfsd):
                    #Gets atom idx numbers
                    test1_idx_list = [val[0] for val in test1_bfsd]
                    test1_idx_list.append(test1_idx)
                    test1_bfsd_sym_list = [alk_mol.atoms_dict[idx].GetSymbol() for idx in test1_idx_list]
                    test1_bfsd_aromatic_atom_arr = np.array([alk_mol.atoms_dict[idx].GetIsAromatic() for idx in test1_idx_list])

                    #This tests that the empirical formula of each frag is the same, if not, test1_sub and test2_sub names are flipped (i.e. fragments are incorrect)
                    if not all(sym in test1_mol_atoms_sym_no_wildcard for sym in test1_bfsd_sym_list):
                        test1_sub_name = frag_tuples[remaining_idx1][1].GetProp("_Name")
                        test2_sub_name = frag_tuples[remaining_idx0][1].GetProp("_Name")
                        print('Test1 and Test2 Reordered')
                    #This tests that there are the same number of aromatic atoms                   
                    if np.count_nonzero(test1_aromatic_atoms_no_wildcard) != np.count_nonzero(test1_bfsd_aromatic_atom_arr):
                        test1_sub_name = frag_tuples[remaining_idx1][1].GetProp("_Name")
                        test2_sub_name = frag_tuples[remaining_idx0][1].GetProp("_Name")
                        print('Test1 and Test2 Reordered')
                
                assert len(test1_bfsd) == len(test1_mol_atoms_sym)-2, f'Length of test1_bfsd ({len(test1_bfsd)}) != Length of test1_mol_atoms-2 ({len(test1_mol_atoms_sym-2)})'
                assert len(test2_bfsd) == len(test2_mol_atoms_sym)-2, f'Length of test2_bfsd ({len(test2_bfsd)}) != Length of test2_mol_atoms-2 ({len(test2_mol_atoms_sym-2)})'
                
                if test1 == 'cis':
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
        concat_df = pd.concat([sterimol_esp_val_0, sterimol_esp_val_1, sterimol_esp_val_2, nbo_ser, rdf_1_ser, rdf_2_ser], axis=0).to_frame(name = name)
        fix_df = concat_df.transpose()        
        tri_df = pd.concat([tri_df, fix_df], axis=0)
        # print(tri_df)
    if alk_mol.alkene_type == 'tetra':
        # continue
        if len(frag_tuples) != 4:
            assert len(frag_tuples) == 3, f'There are less than 3 fragments, which I have not implemented a method for'
            ac.Compute2DCoords(alk_mol.rdkit_alkene_mol)
            c = alk_mol.rdkit_alkene_mol.GetConformer()
            coords = c.GetPositions()
            sub_name_0 = frag_tuples[0][1].GetProp("_Name")
            sub_name_1 = frag_tuples[1][1].GetProp("_Name")
            sub_name_2 = frag_tuples[2][1].GetProp("_Name")

            #Retrieves the series for the sterimol and esp descriptors for this alkene
            sterimol_esp_val_0 = frag_sterimol_esp.loc[sub_name_0]
            sterimol_esp_val_1 = frag_sterimol_esp.loc[sub_name_1]
            sterimol_esp_val_2 = frag_sterimol_esp.loc[sub_name_2]

            sterimol_vals = np.array([sterimol_esp_val_0['vol'], sterimol_esp_val_1['vol'],sterimol_esp_val_2['vol']])

            max_vol_idx = np.where(sterimol_vals == np.max(sterimol_vals))[0][0]
            remaining_idx_arr = np.where(sterimol_vals != np.max(sterimol_vals))[0]

            sub_name_0 = frag_tuples[max_vol_idx][1].GetProp("_Name")
            sub_0_connect = sub_name_0.split('_')[2]
            print(f'sub_name is {sub_name_0}, smiles = {frag_tuples[max_vol_idx][0]}, connect = {sub_0_connect}')
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

            assert len(remaining_list) == 1, f'Remaining List in messed up frag tuples not correct! {remaining_list}'

            


            #This if statement exists to deal with graph isomorphisms, i.e. a relic of Step 4 needing to be redone without dictionaries
            # assert len(frag_tuples) == 2, f"There are less than 2 fragments, which shouldn't be possible, {frag_tuples}"
        else:
            ac.Compute2DCoords(alk_mol.rdkit_alkene_mol)
            c = alk_mol.rdkit_alkene_mol.GetConformer()
            coords = c.GetPositions()
            sub_name_0 = frag_tuples[0][1].GetProp("_Name")
            sub_name_1 = frag_tuples[1][1].GetProp("_Name")
            sub_name_2 = frag_tuples[2][1].GetProp("_Name")
            sub_name_3 = frag_tuples[3][1].GetProp("_Name")

            #Retrieves the series for the sterimol and esp descriptors for this alkene
            sterimol_esp_val_0 = frag_sterimol_esp.loc[sub_name_0]
            sterimol_esp_val_1 = frag_sterimol_esp.loc[sub_name_1]
            sterimol_esp_val_2 = frag_sterimol_esp.loc[sub_name_2]
            sterimol_esp_val_3 = frag_sterimol_esp.loc[sub_name_3]

            sterimol_vals = np.array([sterimol_esp_val_0['vol'], sterimol_esp_val_1['vol'],sterimol_esp_val_2['vol'], sterimol_esp_val_3['vol']])

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
                    sub_name_3 = full_frag_name
                    sterimol_esp_val_3 = create_sterimol_series(sub_name_3, frag_sterimol_esp, num=3)
                else:
                    remaining_list.append(idx)

            assert len(remaining_list) == 2, f'Remaining List not correct! {remaining_list}'

            #These are going to be needed to figure out which fragment sub0 actually is
            sub0_test_idx0, sub0_test_idx1 = alk_mol.neighbor_idx_list(atom_idx=high_vol_c1, atom_idx_not_included=low_vol_c2)

            Chem.SmilesParserParams.removeHs = False
            sub0_mol = Chem.MolFromSmiles(frag_tuples[max_vol_idx][0], Chem.SmilesParserParams.removeHs)
            sub0_mol_atoms_sym = [atom.GetSymbol() for atom in sub0_mol.GetAtoms()]
            sub_mol_atoms_sym_no_wildcard = [sym for sym in sub0_mol_atoms_sym if sym != '*']
            sub0_aromatic_atoms_no_wildcard = np.array([atom.GetIsAromatic() for i, atom in enumerate(sub0_mol.GetAtoms()) if i != 0])

            #This is used to figure out which fragment on the high volume carbon is the correct one
            find_sub0_test0_bfsd = list(yield_bfsd(
                alk_mol=alk_mol,
                start_idx=sub0_test_idx0,
                no_visit_c1 = high_vol_c1,
                no_visit_c2 = low_vol_c2
            ))

            find_sub0_test1_bfsd = list(yield_bfsd(
                alk_mol=alk_mol,
                start_idx=sub0_test_idx1,
                no_visit_c1 = high_vol_c1,
                no_visit_c2 = low_vol_c2
            ))
            if len(find_sub0_test0_bfsd) == len(find_sub0_test1_bfsd):
                #This tests to see if the indices associated for sub0 and test3 need to be switched
                sub0_test0_idx_list = [val[0] for val in find_sub0_test0_bfsd]
                sub0_test0_idx_list.append(sub0_test_idx0)
                sub0_test0_sym_list = [alk_mol.atoms_dict[idx].GetSymbol() for idx in sub0_test0_idx_list]
                sub0_test0_aromatic_atom_arr = [np.array([alk_mol.atoms_dict[idx].GetIsAromatic() for idx in sub0_test0_idx_list])]

                if not all(sym in sub_mol_atoms_sym_no_wildcard for sym in sub0_test0_sym_list):
                    sub_0_idx = sub0_test_idx1
                    test3_idx = sub0_test_idx0
                    print('Sub0 and Test3 Reordered')
                #This tests that there are the same number of aromatic atoms                   
                if np.count_nonzero(sub0_aromatic_atoms_no_wildcard) != np.count_nonzero(sub0_test0_aromatic_atom_arr):
                    sub_0_idx = sub0_test_idx1
                    test3_idx = sub0_test_idx0
                    print('Sub0 and Test3 Reordered')
            else:
                if len(find_sub0_test0_bfsd) == len(sub0_mol_atoms_sym)-2:
                    sub_0_idx = sub0_test_idx0
                    test3_idx = sub0_test_idx1
                else:
                    sub_0_idx = sub0_test_idx1
                    test3_idx = sub0_test_idx0
                
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
                test1 = 'cis'
            else:
                test1 = 'trans'

            remaining_idx0, remaining_idx1 = remaining_list

            test1_sub_name = frag_tuples[remaining_idx0][1].GetProp("_Name")
            test1_sub_connect = test1_sub_name.split('_')[2]
            test2_sub_name = frag_tuples[remaining_idx1][1].GetProp("_Name")
            test2_sub_connect = test2_sub_name.split('_')[2]

            test1_mol = Chem.MolFromSmiles(frag_tuples[remaining_idx0][0], Chem.SmilesParserParams.removeHs)
            test1_mol_atoms_sym = [atom.GetSymbol() for atom in test1_mol.GetAtoms()]
            test1_mol_atoms_sym_no_wildcard = [sym for sym in test1_mol_atoms_sym if sym != '*']
            test1_aromatic_atoms_no_wildcard = np.array([atom.GetIsAromatic() for i,atom in enumerate(test1_mol.GetAtoms()) if i != 0])
            test1_bfsd = list(yield_bfsd(
                alk_mol=alk_mol,
                start_idx=test1_idx,
                no_visit_c1 = high_vol_c1,
                no_visit_c2 = low_vol_c2
            ))

            test2_bfsd = list(yield_bfsd(
                alk_mol=alk_mol,
                start_idx=test2_idx,
                no_visit_c1 = high_vol_c1,
                no_visit_c2 = low_vol_c2
            ))
            test2_mol = Chem.MolFromSmiles(frag_tuples[remaining_idx1][0], Chem.SmilesParserParams.removeHs)
            test2_mol_atoms_sym = [atom.GetSymbol() for atom in test2_mol.GetAtoms()]

            #This is for dealing with unique boundary cases in the graph that wouldn't be caught by assuming the lengths are the same
            if len(test1_bfsd) == len(test2_bfsd):
                #Gets atom idx numbers
                test1_idx_list = [val[0] for val in test1_bfsd]
                test1_idx_list.append(test1_idx)
                test1_bfsd_sym_list = [alk_mol.atoms_dict[idx].GetSymbol() for idx in test1_idx_list]
                test1_bfsd_aromatic_atom_arr = np.array([alk_mol.atoms_dict[idx].GetIsAromatic() for idx in test1_idx_list])

                #This tests that the empirical formula of each frag is the same, if not, test1_sub and test2_sub names are flipped (i.e. fragments are incorrect)
                if not all(sym in test1_mol_atoms_sym_no_wildcard for sym in test1_bfsd_sym_list):
                    test1_sub_name = frag_tuples[remaining_idx1][1].GetProp("_Name")
                    test2_sub_name = frag_tuples[remaining_idx0][1].GetProp("_Name")
                    print('Test1 and Test2 Reordered')
                #This tests that there are the same number of aromatic atoms                   
                if np.count_nonzero(test1_aromatic_atoms_no_wildcard) != np.count_nonzero(test1_bfsd_aromatic_atom_arr):
                    test1_sub_name = frag_tuples[remaining_idx1][1].GetProp("_Name")
                    test2_sub_name = frag_tuples[remaining_idx0][1].GetProp("_Name")
                    print('Test1 and Test2 Reordered')
            
            assert len(test1_bfsd) == len(test1_mol_atoms_sym)-2, f'Length of test1_bfsd ({len(test1_bfsd)}) != Length of test1_mol_atoms-2 ({len(test1_mol_atoms_sym-2)})'
            assert len(test2_bfsd) == len(test2_mol_atoms_sym)-2, f'Length of test2_bfsd ({len(test2_bfsd)}) != Length of test2_mol_atoms-2 ({len(test2_mol_atoms_sym-2)})'
            
            if test1 == 'cis':
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
        tetra_df = pd.concat([tetra_df, fix_df], axis=0)

print(mono_df)
mono_df.to_csv(f'Step_6_Mono_Only_React_Desc_DF.csv')
print(gem_di_df)
gem_di_df.to_csv(f'Step_6_Gem_Di_Only_React_Desc_DF.csv')
print(cis_di_df)
cis_di_df.to_csv(f'Step_6_Cis_Di_Only_React_Desc_DF.csv')
print(trans_di_df)
trans_di_df.to_csv(f'Step_6_Trans_di_Only_React_Desc_DF.csv')
print(tri_df)
tri_df.to_csv(f'Step_6_Tri_Only_React_Desc_DF.csv')
print(tetra_df)
tetra_df.to_csv(f'Step_6_Tetra_Only_React_Desc_DF.csv')
