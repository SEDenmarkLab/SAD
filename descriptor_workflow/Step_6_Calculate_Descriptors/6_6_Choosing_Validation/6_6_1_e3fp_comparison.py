from e3fp.pipeline import fprints_from_smiles
from python_utilities.parallel import Parallelizer
from e3fp.config.params import default_params
from e3fp.fingerprint.fprint import add,mean
from e3fp.fingerprint.db import FingerprintDatabase
from e3fp.fingerprint.fprint import Fingerprint
from e3fp.fingerprint.db import concat
from e3fp.fingerprint.metrics import tanimoto, soergel, dice, cosine, pearson
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.PropertyMol import PropertyMol
from pprint import pprint
from bidict import bidict

def find_canonical_smiles(original_smiles: str):
    mol = Chem.MolFromSmiles(original_smiles)
    can_smiles = Chem.MolToSmiles(mol, canonical=True)
    return can_smiles

def visualize_similarity_mols(name, mol_list, similarity_type):

    obj_name = name
    if len(mol_list) != 0:
        _img = Draw.MolsToGridImage(mol_list, molsPerRow=5, subImgSize=(400,400), useSVG=True,returnPNG=False, highlightAtomLists=None, highlightBondLists=None,  legends=[f'{i.GetProp("_Name")}\nSimilarity = {i.GetProp(f"_{similarity_type}")}' for i in mol_list], maxMols=20000)
        with open(f'{obj_name}.svg', 'w') as f:
            f.write(_img.data)

def give_unique_react(df: pd.DataFrame):
    df_iso = df[['Reactant ID', 'Reactant SMILES']]
    return df_iso.drop_duplicates()

def create_mol_list(react_df: pd.DataFrame):
    id = react_df['Reactant ID'].values
    smiles_str = react_df['Reactant SMILES'].values

    mol_list = list()

    for i, smiles in zip(id, smiles_str):
        mol = Chem.MolFromSmiles(smiles)
        mol.SetProp("_Name", i)
        mol_list.append(mol)

    return mol_list

def make_unique_similarity_mols(input_fp, fold_db, mol_dict, similarity_type='Tanimoto', num_unique=30):
    '''
    This returns a list of unique RDKit Mol Objects with the property of "_Tanimoto" set.
    This list is then sliced based on the number of structures you'd like to include in the output list.

    Works with the following indices:

    -Tanimoto

    -Soergel

    -Dice

    -Cosine

    -Pearson
    '''

    print(f'Fingerprint Name = {input_fp.name}', f'{similarity_type} Similarity Index' )
    if similarity_type == 'Tanimoto':
        val_arr = tanimoto(input_fp, fold_db)
    elif similarity_type == 'Soergel':
        val_arr = soergel(input_fp, fold_db)
    elif similarity_type == 'Dice':
        val_arr = dice(input_fp, fold_db)
    elif similarity_type == 'Cosine':
        val_arr = cosine(input_fp, fold_db)
    elif similarity_type == 'Pearson':
        val_arr = pearson(input_fp, fold_db)
    else:
        raise NameError(f'Fingerprint Similarity Index Not Found: {similarity_type}')
    descending_sort_arr = np.argsort(val_arr)[0][::-1]
    descending_val_arr = val_arr[0][descending_sort_arr]

    fprint_name_arr = np.array(fold_db.fp_names)
    descending_fprint_name_arr = fprint_name_arr[descending_sort_arr]

    tanimoto_val_dict = dict()
    #Makes sure the maximum version of the fingerprint gets written to the dictionary first
    for name, val in zip(descending_fprint_name_arr,descending_val_arr):
        unique_name = name.split('_')[0]
        if unique_name not in tanimoto_val_dict.keys():
            tanimoto_val_dict[unique_name] = val

    unique_mol_list = list()
    for name, val in tanimoto_val_dict.items():
        rdkit_mol = mol_dict[name]
        rdkit_mol.SetProp(f"_{similarity_type}", f'{val}')
        unique_mol_list.append(rdkit_mol)

    return unique_mol_list[0:num_unique]

####This where certain inputs can be changed####
test_fprint_can_smiles = find_canonical_smiles('C=CC1=CC=C(C(F)(F)F)C=C1C(F)(F)F')
test_fprint_name = '2,4-bis(trifluoromethyl)-1-vinylbenzene'
similarity_type = 'Tanimoto'
num_close_mols = 50

'''
References for some Similarity Index
https://www.asprs.org/wp-content/uploads/2018/04/Baisantry_M.pdf

Works with the following Similarity Indexes:

Note: A and B are vectors

#Tanimoto - Lower Similarity Values
--> Tan(A,B) = N(A AND B)/(N_A + N_B - N(A and B))
--> In this formula, N refers to the number of on bits

#Soergel

#Dice

#Cosine
--> Cos(A,B) = (A*B)/(||A||*||B||)
--> Ignores 0-0 matches

#Pearson - Higher Similarity Values
--> How two sets of data fit on a straight line (always on the range of -1 to 1), 
--> Correlation of 0 means no linear relationship between the two objects
--> Corr(A,B) = cov(A,B)/(stdev(A)*stdev(B))

Currently the similarity indices seem to be producing very similar outputs.
'''

################################################




#Creating rdkit name dictionary matching e3fp
full_df = pd.read_csv('p8_reduced_database_column_update_1001.csv')

alkene_iso = give_unique_react(full_df)
can_alkene_smiles = alkene_iso['Reactant SMILES'].values
alkene_mol_list = create_mol_list(alkene_iso)
alkene_mol_dict = bidict({f'react{mol.GetProp("_Name").split("react_")[1]}' : mol for mol in alkene_mol_list})

#Used to check the database before creating fingerprints
if test_fprint_can_smiles in can_alkene_smiles:
    print(f'\nThis SMILES {test_fprint_can_smiles} in Database!!\n')
    print(f'{alkene_iso.iloc[np.where(can_alkene_smiles == test_fprint_can_smiles)[0][0]]}\n')
    raise ValueError()
else:
    #Generate FP
    # raise ValueError()
    test_fprint_res = fprints_from_smiles(
        smiles = test_fprint_can_smiles,
        name = test_fprint_name,
        fprint_params={'level':-1}
    )
    test_fprint_res_fold = [fprint.fold(1024) for fprint in test_fprint_res]

#Used to deal with the fingerprints
full_db = FingerprintDatabase.load('Step_6_E3FP_1000_Entry_DB.fps.bz2')

fold_db = full_db.fold(1024)

#This calculates the similarity index of the input fingerprint and then visualizes the mols associated with them
for fprint in test_fprint_res_fold:
    unique_mol_list = make_unique_similarity_mols(
        input_fp=fprint,
        fold_db = fold_db,
        mol_dict = alkene_mol_dict,
        similarity_type = similarity_type,
        num_unique = num_close_mols
    )

    visualize_similarity_mols(f'./fprint_testing/{fprint.name}_{similarity_type}_Test', unique_mol_list, similarity_type)
