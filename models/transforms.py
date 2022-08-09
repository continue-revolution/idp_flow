#!/usr/bin/python

"""
Coordinate transformation from dihedral angles to 3D coordinates.
Code adapted from https://github.com/rdkit/rdkit
"""

from rdkit.Chem.rdchem import Mol
import torch
from torch import Tensor
from torch.nn import Module
from rdkit import Chem


class Dihedral2Coord(Module):
    """Transform dihedral angles of a batch of conformers into 3D coordinates."""

    def __init__(self, mol: Mol, angles: Tensor):
        """
        Initialization of D2C layer.

        Args:
            mol (Mol): N molecular conformation with the same backbone and possibly different dihedral angles.
            angles (Tensor): a Tensor of shape (K, 4) where K is the number of dihedral angles for a conformer, 4 is (iAtomId, jAtomId, kAtomId, lAtomId).
        """
        self.mol = mol
        self.angles = angles
        self.alist = {}
        self.toBeMovedIdxList()

    def toBeMovedIdxList(self):
        """
        An implementation of toBeMovedIdxList from rdkit.
        See https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/MolTransforms/MolTransforms.cpp#L426
        """
        nAtoms = self.mol.GetNumAtoms()
        K = self.angles.shape[0]
        for i in K:
            iAtomId = self.angles[i, 1].item()
            jAtomId = self.angles[i, 2].item()
            if (iAtomId, jAtomId) not in self.alist:
                self.alist[(iAtomId, jAtomId)] = []
                visitedIdx = [False for _ in range(nAtoms)]
                stack = []
                stack.append(jAtomId)
                visitedIdx[iAtomId] = 1
                visitedIdx[jAtomId] = 1
                tIdx = 0
                wIdx = 0
                doMainLoop = True
                while len(stack) > 0:
                    doMainLoop = False
                    tIdx = stack[0]
                    tAtom = self.mol.GetAtomWithIdx(tIdx)
                    neighbors = tAtom.GetNeighbors()
                    nbrIdx = 0
                    endNbrs = len(neighbors)
                    while nbrIdx != endNbrs:
                        wIdx = neighbors[nbrIdx].GetIdx()
                        if not visitedIdx[wIdx]:
                            visitedIdx[wIdx] = 1
                            stack.append(wIdx)
                            doMainLoop = True
                            break
                        nbrIdx += 1
                    if doMainLoop:
                        continue
                    visitedIdx[tIdx] = 1
                    stack.pop()
                self.alist[(iAtomId, jAtomId)].clear()
                for j in range(nAtoms):
                    if visitedIdx[j] and j != iAtomId:
                        self.alist[(iAtomId, jAtomId)].append(j)

    def transformPoint(self, pt: Tensor, angle: Tensor, axis: Tensor):
        """
        An implementation of differentiable SetRotation and TransformPoint from rdkit.
        See https://github.com/rdkit/rdkit/blob/master/Code/Geometry/Transform3D.cpp

        Args:
            pt (Tensor): a Tensor of shape (N, 3) where N is the batch size, 3 is 3D coordinates.
            angle (Tensor): a Tensor of shape (N) where N is the batch size, 1 is the rotation angle.
            axis (Tensor): a Tensor of shape (N, 3) where N is the batch size, 3 is 3D coordinates of the axis.
        """
        N = pt.shape[0]
        data = torch.eye(4).reshape(1, 4, 4).repeat(N, 1, 1)
        cosT = angle.cos()
        sinT = angle.sin()
        t = 1 - cosT
        X = axis[:, 0]
        Y = axis[:, 1]
        Z = axis[:, 2]
        data[:, 0, 0] = t * X * X + cosT
        data[:, 0, 1] = t * X * Y - sinT * Z
        data[:, 0, 2] = t * X * Z + sinT * Y
        data[:, 1, 0] = t * X * Y + sinT * Z
        data[:, 1, 1] = t * Y * Y + cosT
        data[:, 1, 2] = t * Y * Z - sinT * X
        data[:, 2, 0] = t * X * Z - sinT * Y
        data[:, 2, 1] = t * Y * Z + sinT * X
        data[:, 2, 2] = t * Z * Z + cosT
        x = data[:, 0, 0] * pt[:, 0] + data[:, 0, 1] * \
            pt[:, 1] + data[:, 0, 2] * pt[:, 2] + data[:, 0, 3]
        y = data[:, 1, 0] * pt[:, 0] + data[:, 1, 1] * \
            pt[:, 1] + data[:, 1, 2] * pt[:, 2] + data[:, 1, 3]
        z = data[:, 2, 0] * pt[:, 0] + data[:, 2, 1] * \
            pt[:, 1] + data[:, 2, 2] * pt[:, 2] + data[:, 2, 3]
        pt[:, 0] = x
        pt[:, 1] = y
        pt[:, 2] = z

    def setDihedralRad(self, input: Tensor, angle: Tensor) -> Tensor:
        """
        An implementation of differentiable setDihedralRad from rdkit.
        Note: This version has eliminated all fault checks temporarily. Add them if needed from the link below.
        See https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/MolTransforms/MolTransforms.cpp#L612

        Args:
            mol (Mol): N molecular conformation with the same backbone and possibly different dihedral angles.
            input (Tensor): a Tensor of shape (N) where N is the batch size, 1 is (dihedral angle value).
            angle (Tensor): a Tensor of shape (4) where 4 is (iAtomId, jAtomId, kAtomId, lAtomId).

        Returns:
            output (Tensor): a Tensor of shape (N, M, 3) where N is the batch size, M is the number of atoms, 3 is the 3D coordinates (x, y, z).
        """
        pos = []
        confs = self.mol.GetConformers()
        for conf in confs:
            pos.append(torch.tensor(conf.GetPositions(),
                                    dtype=torch.float32))
        pos = torch.stack(pos)
        rIJ = pos[:, angle[1], :] - pos[:, angle[0], :]
        rJK = pos[:, angle[2], :] - pos[:, angle[1], :]
        rKL = pos[:, angle[3], :] - pos[:, angle[2], :]
        nIJK = rIJ.cross(rJK, dim=2)
        nJKL = rJK.cross(rKL, dim=2)
        m = nIJK.cross(rJK)
        N, _ = input.shape
        values = input + \
            torch.atan2(m.reshape(N, 1, 3).bmm(
                nJKL.reshape(N, 3, 1)).reshape(N))
        rotAxisBegin = pos[:, angle[1], :]
        rotAxisEnd = pos[:, angle[2], :]
        rotAxis = rotAxisEnd - rotAxisBegin
        rotAxis.norm(dim=1)
        for it in self.alist[(angle[1], angle[2])]:
            pos[:, it, :] -= rotAxisBegin
            self.transformPoint(pos[:, it, :], values, rotAxis)
            pos[:, it, :] += rotAxisBegin
        return pos

    def forward(self, input: Tensor) -> Tensor:
        """
        An implementation of differentiable setDihedralRad from rdkit.
        TODO: This version has eliminated all fault checks temporarily. Add them if needed from the link below.
        See https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/MolTransforms/MolTransforms.cpp#L612

        Args:
            input (Tensor): a Tensor of shape (N, K) where N is the batch size, K is the number of dihedral angles for a conformer, 1 is (dihedral angle value).

        Returns:
            output (Tensor): a Tensor of shape (N, M, 3) where N is the batch size, M is the number of atoms, 3 is the 3D coordinates (x, y, z).
        """
        N, K = input.shape
        for i in range(K):
            output = self.setDihedralRad(input[:, i], self.angles[i, :])
        confs = self.mol.GetConformers()
        for i in range(N):
            for j in range(K):
                Chem.rdMolTransforms.SetDihedralDeg(confs[i],
                                                    self.angles[j, 0].item(),
                                                    self.angles[j, 1].item(),
                                                    self.angles[j, 2].item(),
                                                    self.angles[j, 3].item(),
                                                    input[i, j].item())
        return output
