# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy import typing as npt

from ._matrix_functions import dgesvdq
from .mem.arr import SharedArray
from .tri import Triangular
from .utils import chromosomes_set


@dataclass
class Eigendecomposition(SharedArray):
    chromosome: int | str

    def to_file_name(self) -> str:
        return f"no-chr{self.chromosome}.eig.txt.gz"

    @staticmethod
    def get_prefix(**kwargs) -> str:
        chromosome = kwargs.get("chromosome")
        if chromosome is not None:
            return f"no-chr{chromosome}-eig"
        else:
            return "eig"

    @property
    def eigenvalues(self) -> npt.NDArray:
        a = self.to_numpy()
        return a[:, -1]

    @property
    def eigenvectors(self) -> npt.NDArray:
        a = self.to_numpy()
        return a[:, :-1]

    @classmethod
    def from_tri(
        cls,
        *arrays: Triangular,
        chromosome: int | str | None = None,
    ) -> Eigendecomposition:
        if len(arrays) == 0:
            raise ValueError

        if chromosome is None:
            # determine which chromosomes we are leaving out
            chromosomes = chromosomes_set()
            for tri in arrays:
                chromosomes -= {tri.chromosome}

            if len(chromosomes) > 1:
                if "X" in chromosomes:
                    # when leaving out an autosome, we usually only
                    # consider the other autosomes, so also leaving
                    # out the X chromosome is valid
                    chromosomes -= {"X"}

            if len(chromosomes) != 1:
                raise RuntimeError
            (chromosome,) = chromosomes

        # concatenate triangular matrices
        sw = arrays[0].sw
        tri_array = sw.merge(*(tri.name for tri in arrays))
        tri_array.transpose()

        a = tri_array.to_numpy()

        _, sample_count = a.shape

        # allocate outputs
        name = cls.get_name(sw, chromosome=chromosome)
        sw.alloc(name, sample_count, sample_count + 1)
        eig = cls(
            name=name,
            sw=sw,
            chromosome=chromosome,
        )

        # perform high-precision singular value decomposition
        s = eig.eigenvalues
        v = eig.eigenvectors
        numrank = dgesvdq(a, s, v)

        # ensure that we have full rank
        if numrank < sample_count:
            raise RuntimeError

        # the contents of the input arrays have been destroyed
        # so we remove them from the workspace
        sw.free(tri_array.name)
        sw.squash()
        s = eig.eigenvalues

        # transpose only the singular vectors to get the eigenvectors
        eig.transpose(shape=(sample_count, sample_count))

        # scale the singular values to get eigenvalues
        variant_count = sum(tri.variant_count for tri in arrays)
        s[:] = np.square(s / np.sqrt(variant_count))

        return eig