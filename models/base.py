#!/usr/bin/python

"""Initializer of conformers."""

from typing import Mapping
from logging import Logger
import torch
from nflows.distributions.base import Distribution
from nflows.utils import torchutils
from rdkit import Chem
from rdkit.Chem import AllChem as Chem2
from rdkit.Chem import TorsionFingerprints
import numpy as np


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


def set_angles(mol, angle_values, angle_configs):
    N, K = angle_values.shape
    confs = mol.GetConformers()
    for i in range(N):
        for j in range(K):
            Chem.rdMolTransforms.SetDihedralRad(
                confs[i],
                angle_configs[j, 0].item(),
                angle_configs[j, 1].item(),
                angle_configs[j, 2].item(),
                angle_configs[j, 3].item(),
                angle_values[i, j].item())


def get_energy(mol):
    Chem2.MMFFSanitizeMolecule(mol)
    mmff_props = Chem2.MMFFGetMoleculeProperties(mol)
    energy = []
    for i in range(mol.GetNumConformers()):
        ff = Chem2.MMFFGetMoleculeForceField(
            mol, mmff_props, confId=i)
        energy.append(ff.CalcEnergy())
    return torch.tensor(energy)


class Base(Distribution):
    """Conformer initialization."""

    def __init__(self, 
                 num_atoms: int, 
                 batch_size: int, 
                 mol_generator: Mapping[int, Chem.Mol],
                 logger: Logger, 
                 device='cuda'):
        """Initialization of base distribution.

        Args:
            batch_size (int): an integer of batch size.
            num_atoms (int): an integer of number of atoms.
        """
        super().__init__()
        self.device = device
        self.logger = logger
        self.mol = mol_generator(num_atoms)
        Chem.AllChem.EmbedMultipleConfs(self.mol, numConfs=batch_size)
        Chem.rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(
            self.mol, nonBondedThresh=10., )
        self.torsion_angles, _ = get_torsion_tuples(self.mol)
        self.torsion_angles = torch.tensor(self.torsion_angles, device=self.device)
        shape = [self.torsion_angles.shape[0]]
        self._shape = torch.Size(shape)

        self.register_buffer("_log_z",
                             torch.tensor(0.5 * np.prod(shape) * np.log(2 * np.pi),
                                          dtype=torch.float64,
                                          device=device),
                             persistent=False)
        
    def _log_prob(self, inputs, context):
            # Note: the context is ignored.
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )
        neg_energy = -0.5 * \
            torchutils.sum_except_batch(inputs ** 2, num_batch_dims=1)
        return neg_energy - self._log_z

    def _sample(self, num_samples, context):
        if context is None:
            sample_angles = torch.randn(num_samples, *self._shape, device=self._log_z.device)
            # set_angles(self.mol, sample_angles, self.torsion_angles)
            # self.logger.debug(f'Energy before model: {get_energy(self.mol)}')
            return sample_angles
        else:
            # The value of the context is ignored, only its size and device are taken into account.
            context_size = context.shape[0]
            samples = torch.randn(context_size * num_samples, *self._shape,
                                  device=context.device)
            return torchutils.split_leading_dim(samples, [context_size, num_samples])

    def _mean(self, context):
        if context is None:
            return self._log_z.new_zeros(self._shape)
        else:
            # The value of the context is ignored, only its size is taken into account.
            return context.new_zeros(context.shape[0], *self._shape)
