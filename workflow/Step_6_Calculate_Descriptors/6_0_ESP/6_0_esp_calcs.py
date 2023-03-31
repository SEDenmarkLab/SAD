import molli as ml 
import numpy as np
import pickle

col = ml.Collection.from_zip('SAD_Step_5_All_Molli_Frags_ESP_3_01_2023.zip')

# test_col = ml.Collection('test', col.molecules[0:10])

backup_dir = './esp_calcs'

nwchem = ml.NWCHEMDriver(name='bleh', scratch_dir='/scratch/blakeo2/SAD/esp_test',nprocs=16)

concur = ml.Concurrent(col,backup_dir=backup_dir,update=5,timeout=None,concurrent=4)

esp_shit = concur(nwchem.optimize_esp)(functional='b3lyp',basis="6-311G*",maxiter=150,update_geom=False,opt=False,chg=1)

# print(f'This is esp_shit')
# print(esp_shit)
 
esp_dict = dict()

for i in esp_shit:
    # print(i.name)
    fix_name = i.name
    # print(i.espmin)
    # print(i.espmax)
    esp_dict[fix_name] = np.array([i.espmin,i.espmax])
    # print(esp_dict)
    # raise ValueError()
#     print(i.min_a_chg)
#     print(i.max_a_chg)
#     print(i.xyz)


print(esp_dict)
# raise ValueError()
with open(f'Step_6_1000_Entry_ESP_Calc.pkl', 'wb') as f:
    pickle.dump(esp_dict, f)
