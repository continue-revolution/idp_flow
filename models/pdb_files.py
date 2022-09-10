#!/usr/bin/python

"""
Load molecules from a file.
=================
"""

from rdkit import Chem

def generate_pdb(file_id: int) -> Chem.Mol:
    """Load molecules from a pdb file.

    Parameters
    ----------
    file_id : int
        Name of the file where mol is loaded.
    """
    mol = Chem.rdmolfiles.MolFromPDBFile(f'cleaned_geom_mols/mol_{file_id}.pdb')
    mol = Chem.rdmolops.AddHs(mol)
    return mol

def save_pdb(mol: Chem.Mol, file_name: str):
    """Save molecules to a pdb file.

    Parameters
    ----------
    mol : Chem.Mol
        Molecule to be saved.
    file_name: str
        Name of the file which contains the molecule.
    """
    Chem.rdmolfiles.MolToPDBFile(mol, file_name)
