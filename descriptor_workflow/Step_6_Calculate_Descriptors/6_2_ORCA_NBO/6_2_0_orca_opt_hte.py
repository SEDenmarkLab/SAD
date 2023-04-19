import molli as ml
from rdkit import Chem
import pickle
from molli import Orca_Out_Recognize
import shutil
from openbabel import openbabel as ob
import os

molli_col = ml.Collection.from_zip('SAD_Step_2_Canonical_Molli_React_03_01_2023.zip')

orca = ml.ORCADriver("orca", scratch_dir='/scratch/blakeo2/SAD/freq_scratch', nprocs=16)
concur = ml.Concurrent(molli_col, backup_dir='./backup_dir', concurrent=6, update=30, timeout=None)
col_test_run = concur(orca.orca_basic_calc)(
    orca_path = '/opt/share/orca/5.0.3/orca',
    ram_setting = '5000',
    kohn_sham_type = 'rks',
    method = 'b3lyp',
    basis_set = 'def2-svp',
    calc_type = 'opt freq',
    addtl_settings = 'rijcosx def2/j nopop miniprint',
    charge = 0,
    spin_multiplicity = 1,
    )

rerun_name = list()
molli_mols = list()

for obj in col_test_run:
    with open(f'{obj.out_name}','r') as _out, open(obj.hess_file_name,'r') as _hess, open(obj.gbw_file_name, 'rb') as _gbw:
        orca_obj = Orca_Out_Recognize(
            name = obj.mol_name,
            output_file = ''.join(_out.readlines()),
            calc_type = obj.calc_type,
            hess_file = _hess,
            gbw_file = _gbw
        )
        xyz_block = orca_obj.final_xyz()

        imaginary_check = orca_obj.search_freqs(1)
        for val in imaginary_check.values():
            if val <= 0:
                if not os.path.exists('./full_redo'):
                    os.makedirs('./full_redo')
                print(imaginary_check)
                shutil.copy(obj.hess_file_name, './full_redo')
                rerun_name.append(orca_obj.name)
                # xyz_block = orca_obj.final_xyz()
            else:
                xyz_block = orca_obj.final_xyz()

                conv = ob.OBConversion()
                conv.SetInAndOutFormats('xyz','mol2')
                obmol = ob.OBMol()
                conv.ReadString(obmol, xyz_block)

                molli_mols.append(ml.Molecule.from_mol2(conv.WriteString(obmol)))


                # raise ValueError()

new_col = ml.Collection('original_opt_calc_1000_entries', molli_mols)
new_col.to_zip(f'{new_col.name}.zip')

print(f'There are {len(rerun_name)} mono substituted alkenes that need to be re-run, they are:')
print(rerun_name)
