import molli as ml
# from openbabel import openbabel as ob
from alkene_class import alkene_recognize as akr
import pickle
import numpy as np
from math import sqrt
import pandas as pd
import yaml
import os

with open('vdw_radius.yml') as f:
    vdw_dict = yaml.safe_load(f)

def get_rdf(
        conf: ml.dtypes.CartesianGeometry,
        ref_idx: int,
        atom_arr: np.array,
        all_atom_prop_ser: pd.Series,
        inc_size: float,
        first_int: float,
        radial_scaling: int or None):
    
    """
    Takes coordinates for molecule, reference atom index, list of atom indices to compute for, and property series ordered by atom idx

    radial_scaling is an exponent for 1/(r^n) scaling the descriptors - whatever they may be
    
    Original Designed by Ian Rinehart: Still needs a bit of optimization, but I'm gonna work with it as is

    """
    al = list()
    bl = list()
    cl = list()
    dl = list()
    el = list()
    fl = list()
    gl = list()
    hl = list()
    il = list()
    jl = list()
    central_atom = conf.get_coord(ref_idx)
    # print(atom_idx_list)
    for x in range(atom_arr.shape[0]):
        point = conf.get_coord(x)
        dist = sqrt(((float(central_atom[0]) - float(point[0]))**2 + (float(central_atom[1]) - float(point[1]))**2 + (float(central_atom[2])-float(point[2]))**2))
        prop_val = all_atom_prop_ser[x]
        try:
            prop_val_ = float(prop_val)
        except Exception as e:
            print(e)
            prop_val_ = 4.1888 * vdw_dict[prop_val]**3
        const = first_int
        if radial_scaling==0 or radial_scaling == None: pass
        elif (isinstance(radial_scaling, int) and radial_scaling!=0): prop_val_ = prop_val_ / (dist**radial_scaling)
        else: raise ValueError('radial scaling exponent should be an integer or None')
        if dist <= const + inc_size:                                 al.append(prop_val_)
        elif dist > const + inc_size and dist <= const +inc_size*2:  bl.append(prop_val_)
        elif dist > const +inc_size*2 and dist <= const +inc_size*3: cl.append(prop_val_)
        elif dist > const +inc_size*3 and dist <= const +inc_size*4: dl.append(prop_val_)            
        elif dist > const +inc_size*4 and dist <= const +inc_size*5: el.append(prop_val_)
        elif dist > const +inc_size*5 and dist <= const +inc_size*6: fl.append(prop_val_)
        elif dist > const +inc_size*6 and dist <= const +inc_size*7: gl.append(prop_val_)
        elif dist > const +inc_size*7 and dist <= const +inc_size*8: hl.append(prop_val_)
        elif dist > const +inc_size*8 and dist <= const +inc_size*9: il.append(prop_val_)    
        elif dist > const +inc_size*9:                               jl.append(prop_val_)
    series_ = pd.Series([sum(al),sum(bl),sum(cl),sum(dl),sum(el),sum(fl),sum(gl),sum(hl),sum(il),sum(jl)],
        index = [f'{all_atom_prop_ser.name}_'+str(f+1) for f in range(10)]
    )
    '''
    print al
    print bl
    print cl
    print dl
    print el
    print fl
    print gl
    print hl
    print il
    print jl
    '''
    # print(series_)
    return series_

def retrieve_alkene_rdf_descriptors(
        molli_mol:ml.Molecule, 
        alk_mol: akr, 
        apd: dict, 
        prop_val = ['disp', 'pol', 'charge', 'covCN', 'f+', 'f-', 'f0', 'max_bond_order'],
        inc_size: float = 0.90,
        first_int: float = 1.80,
        radial_scaling: int or None = 0
        ):
    '''
    This takes calculates the rdf values for both c1 and c2 in whatever order they have been implemented.
    DO NOT ATTEMPT TO USE THIS WITH MULTIPLE PROPERTY VALUES, IT IS CURRENTLY ONLY IMPLEMENTED FOR ONLY ONE AT A TIME DESPITE THE INPUT.
    '''
    rdf_df = pd.DataFrame(index=['sphere_'+str(i) for i in range(10)])
    rdf_df.name = mol.name
    print(rdf_df.name)
    arb_c1 = alk_mol.c1_idx
    arb_c2 = alk_mol.c2_idx
    atom_arr = np.array(molli_mol.atoms)
    all_c1_series = list()
    all_c2_series = list()
    #Finds RDF for all conformers and creates list of series for concatenation
    for k,conf in enumerate(molli_mol.conformers):
        df = apd[k]
        #An extremely irritating line that has not been fixed because of xtb labeling
        #All this does is fix the atom index numbers to start at 0
        df.index = range(df.index.shape[0])
        df: pd.DataFrame
        for prop in prop_val:
            rdf_ser_arb_c1 = get_rdf(
                conf=conf,
                ref_idx=arb_c1,
                atom_arr=atom_arr,
                all_atom_prop_ser=df[prop],
                inc_size=inc_size,
                first_int=first_int,
                radial_scaling=radial_scaling
            )
            rdf_ser_arb_c1.name = f'{k}_c1'
            all_c1_series.append(rdf_ser_arb_c1)
            rdf_ser_arb_c2 = get_rdf(
                conf=conf,
                ref_idx=arb_c2,
                atom_arr=atom_arr,
                all_atom_prop_ser=df[prop],
                inc_size=inc_size,
                first_int=first_int,
                radial_scaling=radial_scaling
            )
            rdf_ser_arb_c2.name = f'{k}_c2'
            all_c2_series.append(rdf_ser_arb_c2)

    c1_concat_df = pd.concat(all_c1_series, axis=1)
    # print(c1_concat_df)
    c1_rdf_ser = c1_concat_df.mean(axis=1)
    c1_rdf_ser.name = f'c1_rdf'

    c2_concat_df = pd.concat(all_c2_series, axis=1)
    # print(c2_concat_df)
    c2_rdf_ser = c2_concat_df.mean(axis=1)
    c2_rdf_ser.name = f'c2_rdf'
    
    return c1_rdf_ser, c2_rdf_ser

col = ml.Collection.from_zip('almost_all_alkenes_opt_gfnff_conf.zip')
        
with open('SAD_Step_3_All_Alkene_Filter_3_01_2023.pkl', 'rb') as f:
    mol_list = pickle.load(f)

alk_mol_list = [akr(mol) for mol in mol_list]
alk_mol_molli_map = dict()

#This maps the molli mol object name to the correct alk_mol object
for mol in col:
    fixed_name = '_'.join(mol.name.split('_')[0:2])
    for alk_mol in alk_mol_list:
        if alk_mol.mol_name == fixed_name:
            alk_mol_molli_map[mol.name] = alk_mol

final_alkene_rdf_dict = dict()
problem = list()
for mol in col:
    if mol.name == 'react_58':
        print(mol.name)
        alk_mol = alk_mol_molli_map[mol.name]
        if not os.path.exists(f'./conf_props_pkl/{mol.name}_confprops.pkl'):
            problem.append(mol.name)
        else:
            with open(f'./conf_props_pkl/{mol.name}_confprops.pkl', 'rb') as f:
                conf_props_df_list = pickle.load(f)
            c1_rdf_ser, c2_rdf_ser = retrieve_alkene_rdf_descriptors(
                molli_mol=mol,
                alk_mol=alk_mol,
                apd = conf_props_df_list,
                prop_val=['disp'],
                inc_size=0.9,
                first_int=1.8,
                radial_scaling=0
            )
            print(c1_rdf_ser)
            print(c2_rdf_ser)
            final_alkene_rdf_dict[mol.name] = (c1_rdf_ser, c2_rdf_ser)
        raise ValueError()

with open('Step_6_Alkene_RDF_dict.pkl', 'wb') as f:
    pickle.dump(final_alkene_rdf_dict, f)

print(f'problematic mols are {problem}')