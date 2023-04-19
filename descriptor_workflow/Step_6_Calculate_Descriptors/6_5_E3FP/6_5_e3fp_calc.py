from rdkit import Chem
from rdkit.Chem.PropertyMol import PropertyMol
import molli as ml
import pickle
from e3fp.pipeline import fprints_from_smiles
from python_utilities.parallel import Parallelizer
from e3fp.config.params import default_params
from e3fp.fingerprint.fprint import add,mean
from e3fp.fingerprint.db import FingerprintDatabase
from e3fp.fingerprint.fprint import Fingerprint
from e3fp.fingerprint.db import concat
import numpy as np
from pprint import pprint

# col = ml.Collection.from_zip('3H_success_mono_w_crest.zip')
# for mol in col:
#     with open(f'{mol.name}.mol2', 'w') as f:
#         f.writelines(mol.to_mol2())

    # raise ValueError()

# col = ml.Collection.from_zip('almost_all_alkenes.zip')

# rdkit_mol_list = list()
# for mol in col:
#     rdkit_mol = PropertyMol(Chem.MolFromMol2Block(mol.to_mol2(), removeHs=False))
#     rdkit_mol.SetProp("_Name", mol.name)
#     rdkit_mol_list.append(rdkit_mol)

with open('SAD_Step_3_All_Alkene_Filter_3_01_2023.pkl', 'rb') as f:
    rdkit_mol_list = pickle.load(f)

e3fp_dict = dict()

print(default_params.items('conformer_generation'))
# print(default_params.items(''))
j = 0


dbs = list()
for mol in rdkit_mol_list:
    can_smiles = Chem.MolToSmiles(mol, canonical=True)
    storage_name = ''.join(mol.GetProp("_Name").split('_'))
    print(storage_name)
    fprint_res = fprints_from_smiles(
        smiles = can_smiles,
        name = storage_name,
        fprint_params={'level':-1}
    )
    # print(fprint_res)
    # print(mean(fprint_res))
    # break
    # for fprint in fprint_res:
    # # print(len(fprint_res[0]))
    #     print(fprint.bit_count)
    #     print(fprint.density)
    #     fp_folded = fprint.fold(1024)
    #     print(fp_folded)
    #     print(fp_folded.bit_count)
    #     print(fp_folded.density)
    db = FingerprintDatabase(fp_type=Fingerprint, name=mol.GetProp("_Name"))
    db.add_fingerprints(fprint_res)
    # fold_db = db.fold(1024)
    # print(fold_db)
    dbs.append(db)
    # e3fp_dict[mol.GetProp("_Name")] = fprint_res
    # # print(mono_e3fp_dict)
    # j += 1
    # if j > 3:

merge_db = concat(dbs)

merge_db.save(fn = 'Step_6_E3FP_1000_Entry_DB.fps.bz2')
print(merge_db)
test = FingerprintDatabase.load('Step_6_E3FP_1000_Entry_DB.fps.bz2')
print(test)
