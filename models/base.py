#!/usr/bin/python

"""Initializer of conformers."""

import torch
from nflows.distributions import StandardNormal
from rdkit import Chem
from rdkit.Chem import AllChem as Chem2
from rdkit.Chem import TorsionFingerprints
import numpy as np
import random


def generate_branched_alkane(num_atoms: int) -> Chem.Mol:
    """Generates a branched alkane.

    Parameters
    ----------
    num_atoms : int
        Number of atoms in molecule to be generated.
    """
    mol = Chem.MolFromSmiles('CCCC')
    edit_mol = Chem.RWMol(mol)
    while edit_mol.GetNumAtoms() < num_atoms:
        x = Chem.rdchem.Atom(6)
        randidx = np.random.randint(len(edit_mol.GetAtoms()))
        atom = edit_mol.GetAtomWithIdx(randidx)
        if atom.GetDegree() > 2:
            continue
        if atom.GetDegree() == 2 and random.random() <= 0.5:
            continue
        idx = edit_mol.AddAtom(x)
        edit_mol.AddBond(idx, randidx, Chem.rdchem.BondType.SINGLE)

    Chem.SanitizeMol(edit_mol)
    mol = Chem.rdmolops.AddHs(edit_mol.GetMol())

    return mol


def get_torsion_tuples(mol):
    """Gets the tuples for the torsion angles of the molecule.

    Parameters
    ----------
    mol : RDKit molecule
        Molecule for which torsion angles are to be extracted

    * tuples_original, tuples_reindexed : list[int]
        Tuples (quadruples) of indices that correspond to torsion angles. The first returns indices
        for the original molecule and the second for a version of the molecule with Hydrogens removed
        (since there are many cases where this stripped molecule is of interest)
    """

    [mol.GetAtomWithIdx(i).SetProp("original_index", str(i))
     for i in range(mol.GetNumAtoms())]
    stripped_mol = Chem2.rdmolops.RemoveHs(mol)

    nonring, _ = TorsionFingerprints.CalculateTorsionLists(mol)
    nonring_original = [list(atoms[0]) for atoms, ang in nonring]

    original_to_stripped = {
        int(stripped_mol.GetAtomWithIdx(reindex).GetProp("original_index")): reindex
        for reindex in range(stripped_mol.GetNumAtoms())
    }
    nonring_reindexed = [
        [original_to_stripped[original] for original in atom_group]
        for atom_group in nonring_original
    ]

    return nonring_original, nonring_reindexed


class Base(StandardNormal):
    """Conformer initialization."""

    def __init__(self, num_atoms: int, batch_size: int):
        """Initialization of base distribution.

        Args:
            batch_size (int): an integer of batch size.
            num_atoms (int): an integer of number of atoms.
        """
        self.mol = generate_branched_alkane(num_atoms)
        Chem.AllChem.EmbedMultipleConfs(self.mol, numConfs=batch_size)
        Chem.rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(
            self.mol, nonBondedThresh=10., )
        self.torsion_angles, _ = get_torsion_tuples(self.mol)
        self.torsion_angles = torch.tensor(self.torsion_angles)
        super().__init__(shape=[self.torsion_angles.shape[0]])
