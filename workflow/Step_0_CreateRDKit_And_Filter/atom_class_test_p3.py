# from rdkit import RDConfig
# from rdkit.Chem import Draw
# from rdkit.Chem.Draw import IPythonConsole
# from rdkit.Chem import Descriptors
# from rdkit import DataStructs
from re import sub
from rdkit import Chem
from rdkit.Chem import rdqueries as chemq
from rdkit.Chem import AllChem as ac
import numpy as np

class atom_class:
    '''
    YOU NEED TO CANONICALIZE THE SMILES BEFORE USING THIS CLASS. IF YOU DO NOT, THIS CLASS WILL FAIL.
    I chose to write these functions as numpy arrays rather than writing individual lists so that I was always working
    with the same size array, and it would allow me to isolate sets of atoms very easily with the goal of the structure being:

    isolated_atoms = (aromatic & sp2 & carbon)

    All the functions of this sheet figure create a boolean array the size of array 1 (in this case "All Atoms"), and then
    they define the intersection of this array with array 2 (the case of the condition), and return array(bool(1 & 2))
    '''

    def __init__(self, mol):
        
        self.mol = mol
        self.atoms = mol.GetAtoms()
        self.atoms_dict = dict(enumerate(self.atoms))
        self.atoms_array = np.array([x.GetIdx() for x in self.atoms])

        if np.all(np.diff(self.atoms_array) >=  0):
            pass
        else:
            raise ValueError(f'The atom IDs are not ordered from least to greatest')

    def sp2_type(self):
        '''
        This takes a numpy array of Atom IDs and returns a boolean for where SP2 atoms exist. 
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        '''
        sp2_atoms = chemq.HybridizationEqualsQueryAtom(Chem.HybridizationType.SP2)
        sp2 = np.array([x.GetIdx() for x in self.mol.GetAtomsMatchingQuery(sp2_atoms)])
        sp2_bool = np.in1d(self.atoms_array, sp2)
        return sp2_bool

    def aromatic_type(self):
        '''
        This takes a numpy array of Atom IDs and returns a boolean for where AROMATIC atoms exist. 
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        '''
        aromatic_atoms = chemq.IsAromaticQueryAtom()
        aromatic = np.array([x.GetIdx() for x in self.mol.GetAtomsMatchingQuery(aromatic_atoms)])
        aromatic_bool = np.in1d(self.atoms_array, aromatic)
        return aromatic_bool

    def ring_type(self):
        '''
        This takes a numpy array of Atom IDs and returns a boolean for where atoms IN A RING exist. 
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        '''
        ring_atoms = chemq.IsInRingQueryAtom()
        ring = np.array([x.GetIdx() for x in self.mol.GetAtomsMatchingQuery(ring_atoms)])
        ring_bool = np.in1d(self.atoms_array, ring)
        return ring_bool

    def carbon_type(self):
        '''
        This takes a numpy array of Atom IDs and returns a boolean for where CARBON atoms exist. 
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        '''
        carbon_atoms = chemq.AtomNumEqualsQueryAtom(6)
        carbon = np.array([x.GetIdx() for x in self.mol.GetAtomsMatchingQuery(carbon_atoms)])
        carbon_bool = np.in1d(self.atoms_array, carbon)
        return carbon_bool

    def nitrogen_type(self):
        '''
        This takes a numpy array of Atom IDs and returns a boolean for where NITROGEN atoms exist. 
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        '''
        nitrogen_atoms = chemq.AtomNumEqualsQueryAtom(7)
        nitrogen = np.array([x.GetIdx() for x in self.mol.GetAtomsMatchingQuery(nitrogen_atoms)])
        nitrogen_bool = np.in1d(self.atoms_array, nitrogen)
        return nitrogen_bool

    def oxygen_type(self):
        '''
        This takes a numpy array of Atom IDs and returns a boolean for where OXYGEN atoms exist. 
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        '''
        oxygen_atoms = chemq.AtomNumEqualsQueryAtom(8)
        oxygen = np.array([x.GetIdx() for x in self.mol.GetAtomsMatchingQuery(oxygen_atoms)])
        oxygen_bool = np.in1d(self.atoms_array, oxygen)
        return oxygen_bool

    def atom_num_less_than(self, number: int):
        '''
        This takes a numpy array of Atom IDs and returns a boolean for where the ATOM NUMBER is LESS than the input.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        '''
        num_light_atoms = chemq.AtomNumLessQueryAtom(number)
        num_light_atom = np.array([x.GetIdx() for x in self.mol.GetAtomsMatchingQuery(num_light_atoms)])
        num_light_atom_bool = np.in1d(self.atoms_array, num_light_atom)
        return num_light_atom_bool

    def atom_num_equals(self, number: int):
        '''
        This takes a numpy array of Atom IDs and returns a boolean for where the ATOM NUMBER is EQUAL to the input.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        '''
        num_equals_atoms = chemq.AtomNumEqualsQueryAtom(number)
        num_equal_atom = np.array([x.GetIdx() for x in self.mol.GetAtomsMatchingQuery(num_equals_atoms)])
        num_equal_atom_bool = np.in1d(self.atoms_array, num_equal_atom)
        return num_equal_atom_bool

    def atom_num_greater_than(self, number: int):
        '''
        This takes a numpy array of Atom IDs and returns a boolean for where the ATOM NUMBER is GREATER than the input.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        '''
        num_heavy_atoms = chemq.AtomNumGreaterQueryAtom(number)
        num_heavy_atom = np.array([x.GetIdx() for x in self.mol.GetAtomsMatchingQuery(num_heavy_atoms)])
        num_heavy_atom_bool = np.in1d(self.atoms_array, num_heavy_atom)
        return num_heavy_atom_bool

    def isotope_type_equals(self, number: int):
        '''
        This takes a numpy array of Atom IDs and returns a boolean for where the ISOTOPE NUMBER is EQUAL to the input.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        '''
        isotope_atoms = chemq.IsotopeEqualsQueryAtom(number)
        isotope_atom = np.array([x.GetIdx() for x in self.mol.GetAtomsMatchingQuery(isotope_atoms)])
        isotope_atom_bool = np.in1d(self.atoms_array, isotope_atom)
        return isotope_atom_bool

    def charge_type_less_than(self, number: int):
        '''
        This takes a numpy array of Atom IDs and returns a boolean for where the FORMAL CHARGE is LESS THAN the input.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        '''
        charge_less_atoms = chemq.FormalChargeLessQueryAtom(number)
        charge_less_atom = np.array([x.GetIdx() for x in self.mol.GetAtomsMatchingQuery(charge_less_atoms)])
        charge_less_atom_bool = np.in1d(self.atoms_array, charge_less_atom)
        return charge_less_atom_bool

    def charge_type_equals(self, number: int):
        '''
        This takes a numpy array of Atom IDs and returns a boolean for where the FORMAL CHARGE is EQUAL to the input.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        '''
        charge_equals_atoms = chemq.FormalChargeEqualsQueryAtom(number)
        charge_equals_atom = np.array([x.GetIdx() for x in self.mol.GetAtomsMatchingQuery(charge_equals_atoms)])
        charge_equals_atom_bool = np.in1d(self.atoms_array, charge_equals_atom)
        return charge_equals_atom_bool

    def charge_type_greater_than(self, number: int):
        '''
        This takes a numpy array of Atom IDs and returns a boolean for where the FORMAL CHARGE is GREATER THAN the input.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        '''
        charge_greater_atoms = chemq.FormalChargeGreaterQueryAtom(number)
        charge_greater_atom = np.array([x.GetIdx() for x in self.mol.GetAtomsMatchingQuery(charge_greater_atoms)])
        charge_greater_atom_bool = np.in1d(self.atoms_array, charge_greater_atom)
        return charge_greater_atom_bool

    def hcount_less_than(self, number: int):
        '''
        This takes a numpy array of Atom IDs and returns a boolean for where the HYDROGEN COUNT is LESS THAN the input.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        '''
        hcount_less_atoms = chemq.HCountLessQueryAtom(number)
        hcount_less_atom = np.array([x.GetIdx() for x in self.mol.GetAtomsMatchingQuery(hcount_less_atoms)])
        hcount_less_atom_bool = np.in1d(self.atoms_array, hcount_less_atom)
        return hcount_less_atom_bool

    def hcount_equals(self, number: int):
        '''
        This takes a numpy array of Atom IDs and returns a boolean for where the HYDROGEN COUNT is EQUAL to the input.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        '''
        hcount_equals_atoms = chemq.HCountEqualsQueryAtom(number)
        hcount_equals_atom = np.array([x.GetIdx() for x in self.mol.GetAtomsMatchingQuery(hcount_equals_atoms)])
        hcount_equals_atom_bool = np.in1d(self.atoms_array, hcount_equals_atom)
        return hcount_equals_atom_bool

    def hcount_greater_than(self, number: int):
        '''
        This takes a numpy array of Atom IDs and returns a boolean for where the HYDROGEN COUNT is GREATER THAN the input.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.       
        '''
        hcount_greater_atoms = chemq.HCountGreaterQueryAtom(number)
        hcount_greater_atom = np.array([x.GetIdx() for x in self.mol.GetAtomsMatchingQuery(hcount_greater_atoms)])
        hcount_greater_atom_bool = np.in1d(self.atoms_array, hcount_greater_atom)
        return hcount_greater_atom_bool


    def in_ring(self):
        '''
        This takes a numpy array of Atom IDs and returns a boolean for where atoms IN A RING exist. 
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        '''
        ring_atoms = chemq.IsInRingQueryAtom()
        ring = np.array([x.GetIdx() for x in self.mol.GetAtomsMatchingQuery(ring_atoms)])
        ring_bool = np.in1d(self.atoms_array, ring)
        return ring_bool

    def ring_size6(self):
        '''
        This takes a numpy array of Atom IDs and returns a boolean for where atoms IN A 6-MEMBERED RING exist. 
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        '''
        ring_6 = chemq.MinRingSizeEqualsQueryAtom(6)
        size6 = np.array([x.GetIdx() for x in self.mol.GetAtomsMatchingQuery(ring_6)])
        size6_bool = np.in1d(self.atoms_array, size6)
        return size6_bool

    def ring_size5(self):
        '''
        This takes a numpy array of Atom IDs and returns a boolean for where atoms IN A 5-MEMBERED RING exist. 
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        '''
        ring_5 = chemq.MinRingSizeEqualsQueryAtom(5)
        size5 = np.array([x.GetIdx() for x in self.mol.GetAtomsMatchingQuery(ring_5)])
        size5_bool = np.in1d(self.atoms_array, size5)
        return size5_bool

    def in_2_rings(self):
        '''
        This takes a numpy array of Atom IDs and returns a boolean for where atoms IN 2 RINGS exist. 
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        '''
        ring_2 = chemq.InNRingsEqualsQueryAtom(2)
        ring2 = np.array([x.GetIdx() for x in self.mol.GetAtomsMatchingQuery(ring_2)])
        ring2_bool = np.in1d(self.atoms_array, ring2)
        return ring2_bool

    def in_1_ring(self):
        '''
        This takes a numpy array of Atom IDs and returns a boolean for where atoms IN 1 RING exist. 
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        '''
        ring_1 = chemq.InNRingsEqualsQueryAtom(1)
        ring1 = np.array([x.GetIdx() for x in self.mol.GetAtomsMatchingQuery(ring_1)])
        ring1_bool = np.in1d(self.atoms_array, ring1)
        return ring1_bool
    
    def het_neighbors_3(self):
        '''
        This takes a numpy array of Atom IDs and returns a boolean for where atoms have HETEROATOM NEIGHBORS = 3. 
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        '''
        het_a_3 = chemq.NumHeteroatomNeighborsEqualsQueryAtom(3)
        heta3 = np.array([x.GetIdx() for x in self.mol.GetAtomsMatchingQuery(het_a_3)])
        heta3_bool = np.in1d(self.atoms_array, heta3)
        return heta3_bool

    def het_neighbors_2(self):
        '''
        This takes a numpy array of Atom IDs and returns a boolean for where atoms have HETEROATOM NEIGHBORS = 2. 
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        '''
        het_a_2 = chemq.NumHeteroatomNeighborsEqualsQueryAtom(2)
        heta2 = np.array([x.GetIdx() for x in self.mol.GetAtomsMatchingQuery(het_a_2)])
        heta2_bool = np.in1d(self.atoms_array, heta2)
        return heta2_bool

    def het_neighbors_1(self):
        '''
        This takes a numpy array of Atom IDs and returns a boolean for where atoms have HETEROATOM NEIGHBORS = 1. 
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        '''
        het_a_1 = chemq.NumHeteroatomNeighborsEqualsQueryAtom(1)
        heta1 = np.array([x.GetIdx() for x in self.mol.GetAtomsMatchingQuery(het_a_1)])
        heta1_bool = np.in1d(self.atoms_array, heta1)
        return heta1_bool

    def het_neighbors_0(self):
        '''
        This takes a numpy array of Atom IDs and returns a boolean for where atoms have HETEROATOM NEIGHBORS = 0. 
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        '''
        het_a_0 = chemq.NumHeteroatomNeighborsEqualsQueryAtom(0)
        heta0 = np.array([x.GetIdx() for x in self.mol.GetAtomsMatchingQuery(het_a_0)])
        heta0_bool = np.in1d(self.atoms_array, heta0)
        return heta0_bool

    def het_neighbors_greater_1(self):
        '''
        This takes a numpy array of Atom IDs and returns a boolean for where atoms have HETEROATOM NEIGHBORS > 1. 
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        '''
        het_a_g1 = chemq.NumHeteroatomNeighborsGreaterQueryAtom(0)
        hetag1 = np.array([x.GetIdx() for x in self.mol.GetAtomsMatchingQuery(het_a_g1)])
        hetag1_bool = np.in1d(self.atoms_array, hetag1)
        return hetag1_bool

    def het_neighbors_greater_0(self):
        '''
        This takes a numpy array of Atom IDs and returns a boolean for where atoms have HETEROATOM NEIGHBORS > 0. 
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        '''
        het_a_g0 = chemq.NumHeteroatomNeighborsGreaterQueryAtom(0)
        hetag0 = np.array([x.GetIdx() for x in self.mol.GetAtomsMatchingQuery(het_a_g0)])
        hetag0_bool = np.in1d(self.atoms_array, hetag0)
        return hetag0_bool

    def aliph_het_neighbors_2(self):
        '''
        This takes a numpy array of Atom IDs and returns a boolean for where atoms are ALIPHATIC AND HAVE 2 HETEROATOM NEIGHBORS. 
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        '''
        a_het_a_2 = chemq.NumAliphaticHeteroatomNeighborsEqualsQueryAtom(2)
        aheta2 = np.array([x.GetIdx() for x in self.mol.GetAtomsMatchingQuery(a_het_a_2)])
        aheta2_bool = np.in1d(self.atoms_array, aheta2)
        return aheta2_bool

    def aliph_het_neighbors_1(self):
        '''
        This takes a numpy array of Atom IDs and returns a boolean for where atoms are ALIPHATIC AND HAS 1 HETEROATOM NEIGHBORS. 
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        '''
        a_het_a_1 = chemq.NumAliphaticHeteroatomNeighborsEqualsQueryAtom(1)
        aheta1 = np.array([x.GetIdx() for x in self.mol.GetAtomsMatchingQuery(a_het_a_1)])
        aheta1_bool = np.in1d(self.atoms_array, aheta1)
        return aheta1_bool

    def smarts_query(self, smarts: str):
        query = Chem.MolFromSmarts(smarts)
        substructs = self.mol.GetSubstructMatches(query)

        idx = np.zeros(len(self.atoms), dtype=bool)
        for s in substructs:
            idx[list(s)] = True

        return idx