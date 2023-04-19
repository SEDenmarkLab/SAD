import molli as ml
from rdkit import Chem
import pickle
from molli import Orca_Out_Recognize
import shutil
from openbabel import openbabel as ob

molli_col = ml.Collection.from_zip('1000_entry_needs_redo.zip')

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
        imaginary_check = orca_obj.search_freqs(1)
        ####Note: There are still a few structures that require a rerun (I believe about 29 structures of the 774, but for the purposes of continuing, I have not rerun them yet
        for val in imaginary_check.values():
            if val <= 0:
                rerun_name.append(orca_obj.name)

        xyz_block = orca_obj.final_xyz()

        conv = ob.OBConversion()
        conv.SetInAndOutFormats('xyz','mol2')
        obmol = ob.OBMol()
        conv.ReadString(obmol, xyz_block)

        molli_mols.append(ml.Molecule.from_mol2(conv.WriteString(obmol)))

print(f'There are {len(rerun_name)} alkenes that need to be re-run, they are:')
print(rerun_name)

new_col = ml.Collection('redo_opt_calc', molli_mols)
new_col.to_zip(f'{new_col.name}.zip')


#This combines the two collections to make all of them
col1 = ml.Collection.from_zip('original_opt_calc_1000_entries.zip')
col2 = ml.Collection.from_zip('redo_opt_calc.zip')

final_mol_list = list()

final_mol_list = col1.molecules

final_mol_list.extend(col2.molecules)

print(len(final_mol_list))

final_col = ml.Collection('almost_all_alkenes_opt', final_mol_list)

final_col.to_zip(f'{final_col.name}.zip')


####Note: This was used to see that of the remaining

