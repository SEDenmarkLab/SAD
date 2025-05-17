from rdkit import Chem

def find_canonical_smiles(original_smiles: str):
    mol = Chem.MolFromSmiles(original_smiles)
    can_smiles = Chem.MolToSmiles(mol, canonical=True)
    return can_smiles

print(find_canonical_smiles('C#C[C@]([C@@H](O)CCl)(O)C'))