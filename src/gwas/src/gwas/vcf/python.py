# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import IO

import numpy as np
from numpy import typing as npt

from ..log import logger
from ..utils import chromosome_from_int
from .base import VCFFileReader
from .variant import Variant


class PyVCFFile(VCFFileReader):
    def __init__(self, file_path: Path | str, samples: list[str] | None = None) -> None:
        super().__init__(file_path)

        # Read metadata.
        logger.debug(f'Reading metadata for "{str(file_path)}".')
        n_mandatory_columns = len(self.mandatory_columns)

        vcf_variants: list[Variant] = list()
        with self as file_handle:
            for line in file_handle:
                if line.startswith("#"):
                    continue
                tokens = line.split(maxsplit=n_mandatory_columns)
                chromosome_str = tokens[self.chromosome_column_index]
                position_str = tokens[self.position_column_index]
                reference_allele = tokens[self.reference_allele_column_index]
                alternate_allele = tokens[self.alternate_allele_column_index]

                info_str = tokens[self.info_column_index]
                format_str = tokens[self.format_column_index]

                vcf_variants.append(
                    Variant.from_metadata_columns(
                        chromosome_str,
                        position_str,
                        reference_allele,
                        alternate_allele,
                        info_str,
                        format_str,
                    )
                )
        self.vcf_variants = self.make_data_frame(vcf_variants)
        self.variant_indices = np.arange(self.vcf_variant_count, dtype=np.uint32)

        chromosome_int_set = set(self.vcf_variants["chromosome_int"])
        if len(chromosome_int_set) != 1:
            raise ValueError("Inconsistent chromosomes across variants.")
        self.chromosome = chromosome_from_int(chromosome_int_set.pop())

    def __enter__(self) -> IO[str]:
        self.variant_index = 0
        return super().__enter__()

    def read(
        self,
        dosages: npt.NDArray,
    ) -> None:
        if self.file_handle is None:
            raise ValueError("File handle is not open")

        if dosages.size == 0 or self.variant_count == 0 or self.sample_count == 0:
            return  # Nothing to do.

        if dosages.shape != (self.variant_count, self.sample_count):
            raise ValueError(
                f"Expected shape {(self.variant_count, self.sample_count)} for "
                f"variable `dosages` but received {dosages.shape}"
            )

        n_mandatory_columns = len(self.mandatory_columns)

        variant_indices_index = 0
        for line in self.file_handle:
            if line.startswith("#"):
                continue

            variant_index = self.variant_index
            self.variant_index += 1

            if variant_index == self.variant_indices[variant_indices_index]:
                tokens = line.split(maxsplit=n_mandatory_columns)

                # Parse format.
                genotype_fields = tokens[self.format_column_index].split(":")
                field_count = len(genotype_fields)
                dosage_field_index = genotype_fields.index("DS")

                # Parse dosages.
                fields = tokens[-1].replace(":", "\t").split()
                dosage_fields = fields[dosage_field_index::field_count]
                if self.sample_indices is not None:
                    dosage_fields = [dosage_fields[i] for i in self.sample_indices]
                dosages[variant_indices_index, :] = dosage_fields

                variant_indices_index += 1
                if variant_indices_index == len(self.variant_indices):
                    return
