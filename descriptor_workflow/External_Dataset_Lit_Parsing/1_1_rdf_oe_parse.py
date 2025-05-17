from rdfmodule import rdf_fixer
import pandas as pd
from glob import glob
from openeye import oechem
from openeye import oedepict
from pprint import pprint
import molli as ml
import os
from pprint import pprint
import pickle
from bidict import bidict

'''
This script utilizes RDF files from Scifinder and fixes them with RDF-Fixer

Citation:
https://github.com/DocMinus/chem-rdf-fixer

These are then parsed using Openeye's OEChem TK

Citation:
OEChem TK 2024.2.1 OpenEye, Cadence Molecular Sciences, Santa Fe, NM. http://www.eyesopen.com

'''


def check_reagents(col, df: pd.DataFrame):
    all_smiles = list()

    for res in df[col].values:
        if isinstance(res, str):
            all_smiles.append(res)
    
    return all_smiles

def atom_stereo_present(oemol: oechem.OEMol) -> bool:
    for atom in oemol.GetAtoms():
        cip = oechem.OEPerceiveCIPStereo(oemol, atom)
        if atom.HasStereoSpecified():
            return True
        if cip == oechem.OECIPAtomStereo_UnspecStereo:
            return True
    
    return False  

all_smi_set = set()
list_smi = list()
smi_dict = {
    'Mono': set(),
    'Cis': set(),
    'Gem': set(),
    'Trans': set(),
    'Tri':set(),
    'Tetra': set()
}
#This iterates through all RDF files
for rdf in glob('*.rdf'):
    current_set = set()
    alk_type = rdf.split('_')[0]
    name = rdf.split(".rdf")[0]

    #This fixes the RDF files
    if not os.path.exists(f'./{name}.csv'):
        rdf_fixer.fix(f'{name}.rdf')
        os.remove(f'{name}_fixed.rdf')

    df = pd.read_csv(f'{name}.csv', sep='\t', dtype=str)
    
    all_smiles = list()
    
    all_smiles.extend(check_reagents('Reagent0', df))
    all_smiles.extend(check_reagents('Reagent1', df))

    all_smiles = '\n'.join(all_smiles)

    #This reads in the smiles
    ims = oechem.oemolistream() 
    ims.SetFormat(oechem.OEFormat_SMI)
    ims.openstring(all_smiles)

    mols = list()
    mol = oechem.OEMol()

    for mol in ims.GetOEMols():
        oesmi = oechem.OEMolToSmiles(mol)
        all_smi_set.add(oesmi)
        list_smi.append(oesmi)
        current_set.add(oesmi)
    
    smi_dict_set = smi_dict[alk_type]
    smi_dict[alk_type] = smi_dict_set.union(current_set)

with open('smi_dict_not_filtered.pkl', 'wb') as f:
    pickle.dump(smi_dict, f)

alk_type_dict = dict()

for alk_type,smi_subset in smi_dict.items():
    for smi in smi_subset:
        alk_type_dict[smi] = alk_type

print(f'There are {len(list_smi)} reactants reported in these files!')
print(f'There are {len(all_smi_set)} unique reactants in these files!')

#Searches for diols
diol =oechem.OESubSearch('[OHX2][CX4][CX4][OHX2]')

#Searches for alkenes
alkene = oechem.OESubSearch('C=C')

final_mols = list()
no_match = list()
at_smi = list()
stereo_present = list()
nostereo_present = list()
nostereo = 0
stereo = 0

final_mol_type = dict()

smi_dict['Cis'] = set()
smi_dict['Trans'] = set()


for smi in all_smi_set:
    mol = oechem.OEGraphMol()
    oechem.OESmilesToMol(mol, smi)

    #Looks for a match in the described alkene/diol pair
    if alkene.SingleMatch(mol):
        #Removes if chiral alkene
        if '@' in smi:
            at_smi.append(mol)
            continue
        else:
            for atom in mol.GetAtoms():

                #Finds if there are alkenes with hidden chirality
                cip = oechem.OEPerceiveCIPStereo(mol, atom)
                if atom.HasStereoSpecified():
                    stereo += 1
                    break
                if cip == oechem.OECIPAtomStereo_UnspecStereo:
                    nostereo += 1
                    # print(smi)
                    nostereo_present.append(smi)
                    break
            if atom_stereo_present(mol):
                stereo_present.append(mol)
                continue
            else:
                #Looks for Cis and Trans alkenes to differentiate them
                
                if (alk_type_dict[smi] == 'Cis') or (alk_type_dict[smi] == 'Trans'):
                    i = 0
                    for bond in mol.GetBonds():
                        bond: oechem.OEBondBase
                        oechem.OEBondStereo_Undefined
                        if bond.HasStereoSpecified(oechem.OEBondStereo_CisTrans):
                            for atomB in bond.GetBgn().GetAtoms():
                                if atomB == bond.GetEnd():
                                    continue
                                for atomE in bond.GetEnd().GetAtoms():
                                    if atomE == bond.GetBgn():
                                        continue
                                    v = []
                                    v.append(atomB)
                                    v.append(atomE)
                                    stereo = bond.GetStereo(v, oechem.OEBondStereo_CisTrans)

                                    if stereo == oechem.OEBondStereo_Cis:
                                        if i > 1:
                                            alk_type_dict[smi] = 'Cis_and_Trans'
                                        i += 1
                                        alk_type_dict[smi] = 'Cis'
                                        
                                    elif stereo == oechem.OEBondStereo_Trans:
                                        if i > 1:
                                            alk_type_dict[smi] = 'Cis_and_Trans'
                                        i += 1
                                        alk_type_dict[smi] = 'Trans'


                final_mols.append(mol)
                final_mol_type[smi] = alk_type_dict[smi]
    
    else:
        no_match.append(mol)

with open('problem_mol_type_dict.pkl', 'wb') as f:
    pickle.dump(final_mol_type,f)

print(f'There are {len(no_match)} mols without alkenes!')
print(f'There are {len(at_smi)} mols whose smiles has an @ (i.e. stereocenter)!')
print(f'There are {len(stereo_present)} mols that have atoms with specified stereochemistry ({stereo} mols) and unspecified stereochemistry (i.e. racemic mixtures) ({nostereo} mols)!')
print(f'There are {len(final_mols)} mols that are usable!')

vals = list(final_mol_type.values())

Mono = vals.count('Mono')
Gem = vals.count('Gem')
Cis = vals.count('Cis')
Trans = vals.count('Trans')
Tri = vals.count('Tri')
Tetra = vals.count('Tetra')
Cis_and_Trans = vals.count('Cis_and_Trans')

print(f'There are {Mono} Monosubstituted alkenes!')
print(f'There are {Gem} Gem Disubstituted substituted alkenes!')
print(f'There are {Cis} Cis Disubstituted alkenes!')
print(f'There are {Trans} Trans Disubstituted alkenes!')
print(f'There are {Tri} Trisubstituted alkenes!')
print(f'There are {Tetra} Tetrasubstituted alkenes!')
print(f'There are {Cis_and_Trans} that contain both Cis and Trans Disubstituted alkenes!')