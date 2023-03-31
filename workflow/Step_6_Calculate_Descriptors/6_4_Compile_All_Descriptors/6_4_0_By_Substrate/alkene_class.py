from rdkit import Chem
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.PropertyMol import PropertyMol
import pickle
import molli as ml
from pprint import pprint
from atom_class_test_p3 import atom_class

class alkene_recognize:
    def __init__(self, rdkit_alkene_mol):

        self.rdkit_alkene_mol = rdkit_alkene_mol
        self.atoms = rdkit_alkene_mol.GetAtoms()
        self.atoms_dict = dict(enumerate(self.atoms))
        self.atoms_array = np.array([x.GetIdx() for x in self.atoms])
        
        if rdkit_alkene_mol.HasProp("_Name"):
            self.mol_name = rdkit_alkene_mol.GetProp("_Name")
        else:
            self.mol_name = None

        if rdkit_alkene_mol.HasProp("_Alkene_Type"):
            self.alkene_type = rdkit_alkene_mol.GetProp("_Alkene_Type")
        else:
            self.alkene_type = None

        if rdkit_alkene_mol.HasProp("_Alkene_w_H"):
            self.alkene_bool_h = np.array([True if v =='1' else False for v in rdkit_alkene_mol.GetProp("_Alkene_w_H")])
            self.c_idx_np = self.atoms_array[self.alkene_bool_h]
            self.c1_idx = self.c_idx_np[0]
            self.c2_idx = self.c_idx_np[1]
            
        else:
            self.alkene_bool_h = None
            self.c_idx_np = None
            self.c1_idx = None
            self.c2_idx = None

        if np.all(np.diff(self.atoms_array) >=  0):
            pass
        else:
            raise ValueError(f'The atom IDs are not ordered from least to greatest')

    def neighbor_symbol_list(self, atom_idx:int, atom_idx_not_included=9999):
        '''
        Returns neighbor symbol list for atom idx. Can also include if certain atom idx should not be included.
        '''
        neighbor_symbol = [atom.GetSymbol() for atom in self.atoms_dict[atom_idx].GetNeighbors() if atom_idx_not_included != atom.GetIdx()] 
        return neighbor_symbol

    def neighbor_idx_list(self, atom_idx:int, atom_idx_not_included=9999):
        '''
        Returns neighbor idx list for atom idx. Can also include if certain atom idx should not be included
        '''
        neighbor_idx = [atom.GetIdx() for atom in self.atoms_dict[atom_idx].GetNeighbors() if atom_idx_not_included != atom.GetIdx()]
        return neighbor_idx

    def neighbor_atom_list(self, atom_idx: int, atom_idx_not_included=9999):
        '''
        Return neighbor atom list for atom idx. Can also include if a certain atom idx should not be included
        '''
        neighbor_atoms = [atom for atom in self.atoms_dict[atom_idx].GetNeighbors() if atom_idx_not_included != atom.GetIdx()]
        return neighbor_atoms

    def neighbor_symbol(self, symbol:str, atom_idx:int, atom_idx_not_included = 9999):
        '''
        Returns an idx value if there is a neighboring symbol.
        '''
        neighbor_symbol = [atom.GetSymbol() for atom in self.atoms_dict[atom_idx].GetNeighbors() if atom_idx_not_included != atom.GetIdx()]
        
        for atom in neighbor_symbol:
            if symbol == neighbor_symbol:
                return True
            else:
                return False


        

