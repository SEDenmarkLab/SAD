import molli as ml
import numpy as np
from tqdm import tqdm

mlib = ml.MoleculeLibrary("6_6_1_DB_Merged_Desc.mlib")
fix_mlib = ml.MoleculeLibrary("6_6_2_DB_Merged_ESPFix.mlib", readonly=False, overwrite=True)

# The normal ESPMin calculation takes place at a Median value of previously calculated Hydrogen ESPMin:
espmin_median = 342.2778625488281

# The 99th ESPMax calculation will utilize the median value of all calculated 99th percentiles for tetramethyl ammonium ions due to its stability:
esp99_median = 537.5414114432903

# The ESPMax calculation will utilize the median value of all calculated Hydrogen ESPMax:
espmax_median = 636.3302001953125

with mlib.reading(), fix_mlib.writing():
    for name in tqdm(mlib):
        m = mlib[name]

        #Find Current Q atoms
        for a in m.atoms:
            if 'Q' in a.attrib:
                qnum = a.attrib['Q']
                match qnum:
                    case 1:
                        q1a = a
                    case 2:
                        q2a = a
                    case 3:
                        q3a = a
                    case 4:
                        q4a = a
        
        q_atoms = np.array([q1a, q2a, q3a, q4a])
        for a in q_atoms:
            assert a in m.atoms, f'{m.get_atom_index(a)}: {a} not in {m}'

        #This standardizes the esp calculations of ESPs as they are necessarily the same
        for a in q_atoms:
            a: ml.Atom
            if a.element == ml.Element.H:
                a.attrib['NWESPmin'] = espmin_median
                a.attrib['NWESPMax'] = espmax_median
                a.attrib['99ESPMax'] = esp99_median

        fix_mlib[name]= m
