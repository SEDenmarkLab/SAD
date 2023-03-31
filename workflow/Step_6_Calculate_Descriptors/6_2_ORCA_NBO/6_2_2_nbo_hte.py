from glob import glob
import molli as ml
from pprint import pprint
import molli as ml
from rdkit import Chem
import pickle
from molli import Orca_Out_Recognize
from pprint import pprint


col = ml.Collection.from_zip('almost_all_alkenes_opt.zip')

orca = ml.ORCADriver("orca", scratch_dir='./scratch_dir', nprocs=1)
concur = ml.Concurrent(col, backup_dir='done_nbo', concurrent=6, update=30, timeout=None)
test = concur(orca.orca_basic_calc)(
    orca_path = '/opt/share/orca/5.0.3/orca',
    ram_setting = '5000',
    kohn_sham_type = 'rks',
    method = 'b3lyp',
    basis_set = 'def2-svp',
    calc_type = 'nbo',
    addtl_settings = 'rijcosx def2/j nopop miniprint',
    charge = 0,
    spin_multiplicity = 1,
    )

nbo_dict = dict()

for obj in test:
    with open(f'{obj.out_name}','r') as _out, open(obj.gbw_file_name, 'rb') as _gbw:
        orca_obj = Orca_Out_Recognize(
            name = obj.mol_name,
            output_file = ''.join(_out.readlines()),
            calc_type = obj.calc_type,
            hess_file = None,
            gbw_file = _gbw
        )
        if orca_obj.name == 'react_244':
            print(orca_obj.name)
            orb_homo_lumo, nat_charge_dict, pert_final_list, nbo_orb_final_list = orca_obj.nbo_parse_values()
            print(f'orb_homo_lumo')
            print(orb_homo_lumo)
            print(f'nat_charge_dict')
            print(nat_charge_dict)
            print('pert_final_list')
            pprint(pert_final_list)
            print('nbo_orb_final_list')
            pprint(nbo_orb_final_list)
        
    nbo_dict[orca_obj.name] = (orb_homo_lumo, nat_charge_dict, pert_final_list, nbo_orb_final_list)

with open('Step_6_1000_Entry_NBO_almost_all_alkenes_dict.pkl', 'wb') as f:
    pickle.dump(nbo_dict, f)

