#!/usr/bin/python

"""
Energy layer to get the loss.
"""

import torch
from torch.nn import Module
from torch import Tensor
from rdkit.Chem.rdchem import Mol
import rdkit.Chem.AllChem as Chem2


class Energy(Module):
    """Energy loss with forward and backward pass."""

    def __init__(self, mol: Mol):
        self.mol = mol
        self.ff_list = []

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass of energy. Only mol matters here."""
        Chem2.MMFFSanitizeMolecule(self.mol)
        mmff_props = Chem2.MMFFGetMoleculeProperties(self.mol)
        energy = []
        for i in range(self.mol.GetNumConformers()):
            ff = Chem2.MMFFGetMoleculeForceField(
                self.mol, mmff_props, confId=i)
            self.ff_list.append(ff)
            # energy += ff.CalcEnergy()
            energy.append(ff.CalcEnergy())
        # energy = energy / self.mol.GetNumConformers()
        energy = torch.tensor(energy)
        return energy

    def backward(self, input: Tensor) -> Tensor:
        """Backward pass of energy. Only ff_list from forward matters here."""
        grad_list = []
        for ff in self.ff_list:
            grad_list.append(torch.tensor(ff.CalcGrad()).reshape(1, -1, 3))
        grad_energy = torch.stack(grad_list)
        return grad_energy
