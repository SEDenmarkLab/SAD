import molli as ml
from tqdm import tqdm
from pprint import pprint
from molli.pipeline import nwchem

sterimol_mlib = ml.MoleculeLibrary('5_3_DB_OPT_AlignMaxVol.mlib')
esp_mlib = ml.MoleculeLibrary('6_2_2_DB_ESPFrags_Corrected.mlib')
xtb_mlib = ml.MoleculeLibrary('6_4_2_DB_CRESTScreen.mlib')
crest_clib = ml.ConformerLibrary('6_4_3_DB_CRESTScreen_Align.clib')
nbo_mlib = ml.MoleculeLibrary('6_5_2_DB_NBOAdded.mlib')

merged_mlib = ml.MoleculeLibrary('6_6_1_DB_Merged_Desc.mlib', readonly=False, overwrite=True)

def merge_att(old_m:ml.Molecule, lib: ml.MoleculeLibrary | ml.ConformerLibrary):
    '''Modifies an object in place by adding missing attributes from the new library.
    The key should be in both libraries

    Parameters
    ----------
    old_m : ml.Molecule
        Original object that properties are getting added to
    lib : ml.MoleculeLibrary | ml.ConformerLibrary
        Library to pull molecule/conformer ensemble from
    '''
    assert old_m.name in lib, f'Key {old_m.name} not in library!'

    new_obj = lib[old_m.name]

    #Adds new molecule/conformerensemble attributes
    for obj_att in new_obj.attrib:
        if obj_att not in m.attrib:
            m.attrib[obj_att] = new_obj.attrib[obj_att]

        #Adds new atom attributes
        for i,atom in enumerate(new_obj.atoms):
            for atom_att in atom.attrib:
                old_m_atom = old_m.get_atom(i)
                if atom_att not in old_m_atom.attrib:
                    old_m_atom.attrib[atom_att] = atom.attrib[atom_att]

with sterimol_mlib.reading(), esp_mlib.reading(), xtb_mlib.reading(), crest_clib.reading(), nbo_mlib.reading(), merged_mlib.writing():
    for k in tqdm(sterimol_mlib):
        m = sterimol_mlib[k]
        merge_att(m, xtb_mlib)
        merge_att(m, crest_clib)
        merge_att(m, nbo_mlib)

        #For ESPmin/max calcs, I have to remerge the separate fragments
        #This is repetitive and could be wrapped into a function, but I didn't do that.
        for a in m.atoms:
            if 'Q' in a.attrib:
                match a.attrib['Q']:
                    case 1:
                        q1a = a
                        q1_m = esp_mlib[f'{k}_Q1']
                        q1a.attrib['NWESPmin'] = q1_m.attrib['NWESPmin']
                        q1a.attrib['NWESPMax'] = q1_m.attrib['NWESPMax']
                        q1a.attrib['99ESPMax'] = q1_m.attrib['99ESPMax']
                    case 2:
                        q2a = a
                        q2_m = esp_mlib[f'{k}_Q2']
                        q2a.attrib['NWESPmin'] = q2_m.attrib['NWESPmin']
                        q2a.attrib['NWESPMax'] = q2_m.attrib['NWESPMax']
                        q2a.attrib['99ESPMax'] = q2_m.attrib['99ESPMax']
                    case 3:
                        q3a = a
                        q3_m = esp_mlib[f'{k}_Q3']
                        q3a.attrib['NWESPmin'] = q3_m.attrib['NWESPmin']
                        q3a.attrib['NWESPMax'] = q3_m.attrib['NWESPMax']
                        q3a.attrib['99ESPMax'] = q3_m.attrib['99ESPMax']
                    case 4:
                        q4a = a
                        q4_m = esp_mlib[f'{k}_Q4']
                        q4a.attrib['NWESPmin'] = q4_m.attrib['NWESPmin']
                        q4a.attrib['NWESPMax'] = q4_m.attrib['NWESPMax']
                        q4a.attrib['99ESPMax'] = q4_m.attrib['99ESPMax']
        m.attrib['True ESP'] = {
            'Q1':{'NWESPmin':q1a.attrib['NWESPmin'], 'NWESPMax':q1a.attrib['NWESPMax'], '99ESPMax':q1a.attrib['99ESPMax']},
            'Q2':{'NWESPmin':q2a.attrib['NWESPmin'], 'NWESPMax':q2a.attrib['NWESPMax'], '99ESPMax':q2a.attrib['99ESPMax']},
            'Q3':{'NWESPmin':q3a.attrib['NWESPmin'], 'NWESPMax':q3a.attrib['NWESPMax'], '99ESPMax':q3a.attrib['99ESPMax']},
            'Q4':{'NWESPmin':q4a.attrib['NWESPmin'], 'NWESPMax':q4a.attrib['NWESPMax'], '99ESPMax':q4a.attrib['99ESPMax']},
            }

        merged_mlib[k] = m