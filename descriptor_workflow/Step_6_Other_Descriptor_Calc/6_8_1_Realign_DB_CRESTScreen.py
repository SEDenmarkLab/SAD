import molli as ml
from tqdm import tqdm 
import numpy as np
from molli.math import rotation_matrix_from_vectors, rotation_matrix_from_axis

def set_origin(ml_mol:ml.Molecule,i: int):
    ml_mol.translate(-1*ml_mol.coords[i])

def rot_rz(ml_mol: ml.Molecule, c0:ml.Atom, c1: ml.Atom):
    v1 = ml_mol.vector(c0, c1)
    t_matrix = rotation_matrix_from_vectors(v1, np.array([1,0,0]))
    ml_mol.transform(t_matrix)

def rot_xy(ml_mol: ml.Molecule, q1a:ml.Atom, c0:ml.Atom, c1:ml.Atom):
    '''
    Q1
      \\ 
        Q1C = Q2C
    
    This should be positive with respect to right hand rule
    '''
    # v1 = ml_mol.vector(q1a, c0)
    v1 = ml_mol.vector(c0, q1a)
    v2 = ml_mol.vector(c0, c1)
    # print([ml_mol.get_atom_index(a) for a in [q1a,c0,c1]])
    c = np.cross(v1,v2)
    t_matrix = rotation_matrix_from_vectors(c, np.array([0,0,-1]))
    ml_mol.transform(t_matrix)

def check_q_align(ml_mol: ml.Molecule,q1a:ml.Atom,q2a:ml.Atom,q3a:ml.Atom):
    '''Asserts that the Q1Q2 vector and Q2Q3 vector are negative with respect to the z axis. This confirms that they are ordered correctly.

    Parameters
    ----------
    ml_mol : ml.Molecule
    q1a : ml.Atom
        Q1 Atom
    q2a : ml.Atom
        Q2 Atom
    q3a : ml.Atom
        Q3 Atom
    '''
    q1q2v = ml_mol.vector(q1a, q2a)
    q1q3v = ml_mol.vector(q1a, q3a)

    if np.sign(np.dot(q1q2v, q1q3v)) != 1:
        t_matrix = rotation_matrix_from_axis([1,0,0], np.radians(180))
        ml_mol.transform(t_matrix)
            
        q1q2v = ml_mol.vector(q1a, q2a)
        q1q3v = ml_mol.vector(q2a, q3a)

def realign_clib(ref_mlib: ml.MoleculeLibrary, old_clib: ml.ConformerLibrary, new_clib: ml.ConformerLibrary):
    print(f'''
Reference Library: {ref_mlib}
Old Conformer Library: {old_clib}
New Conformer Library Written To: {new_clib}
''')
    with ref_mlib.reading(), old_clib.reading(), new_clib.writing():
        for name in tqdm(ref_mlib):

            m = ref_mlib[name]

            #Isolates quadrant atoms
            q1a, q2a, q3a, q4a = m.attrib['Q Order']
            q_idx = [q1a, q2a, q3a, q4a]
            
            #Isolates Alkene Carbons
            c0, c1 = m.attrib['C Order']
            c_idx = [c0, c1]

            ens = old_clib[name]

            for conf in ens:
                if conf._conf_id == 0:
                    c_q1a, c_q2a, c_q3a, c_q4a = [m.get_atom(a) for a in q_idx]
                    c_c0, c_c1 = [m.get_atom(a) for a in c_idx]
                else:
                    #Sets Alkene Carbon C0 to be the origin
                    set_origin(m, m.get_atom_index(c_c0))

                    #Rotates molecule such that alkene atoms are along the X-axis (C0 --> C1)
                    rot_rz(m, c_c0, c_c1)

                    #Rotates molecule such that Q1 and alkene atoms are in the XY plane
                    rot_xy(m, c_q1a, c_c0, c_c1)

                    #This asserts that the vectors formed after alignment are correct and rotates 180 degrees if Q1 ends up in below C0.
                    check_q_align(m, c_q1a, c_q2a, c_q3a)
            new_clib[name] = ens




bfs_mlib = ml.MoleculeLibrary("5_6_Realign_BFSVol.mlib")
clib = ml.ConformerLibrary("5_3_2_DB_CRESTScreen_Align.clib")
bfs_clib = ml.ConformerLibrary("5_7_0_CRESTScreen_BFSAlign.clib", readonly=False, overwrite=True)

#Realigns the New Conformer Library based on the Alignment Via 3-BFS Vol Search
realign_clib(ref_mlib=bfs_mlib, old_clib=clib, new_clib=bfs_clib)

max_mlib = ml.MoleculeLibrary("5_6_Realign_MaxVol.mlib")
max_clib = ml.ConformerLibrary("5_7_0_CRESTScreen_MaxAlign.clib", readonly=False, overwrite=True)

#Realigns the New Conformer Library based on the Alignment via Max Vol Search
realign_clib(ref_mlib=max_mlib, old_clib=clib, new_clib=max_clib)
