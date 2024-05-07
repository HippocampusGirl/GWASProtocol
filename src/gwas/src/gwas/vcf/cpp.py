# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy import typing as npt

from ..utils import chromosome_from_int
from .base import VCFFileReader
from .variant import Variant


class CppVCFFile(VCFFileReader):
    def __init__(self, file_path: Path | str) -> None:
        super().__init__(file_path)

        vcf_variants: list[Variant] = list()
        from ._read import read_str

        with self as file_handle:
            read_str(
                vcf_variants,
                Variant.from_metadata_columns,
                file_handle.fileno(),
                self.header_length,
                len(self.columns),
                self.metadata_column_indices,
            )
        self.vcf_variants = self.make_data_frame(vcf_variants)
        self.variant_indices = np.arange(self.vcf_variant_count, dtype=np.uint32)

        chromosome_int_set = set(self.vcf_variants["chromosome_int"])
        if len(chromosome_int_set) != 1:
            raise ValueError("Inconsistent chromosomes across variants.")
        self.chromosome = chromosome_from_int(chromosome_int_set.pop())

        format_str_set = set(self.vcf_variants["format_str"])
        if len(format_str_set) != 1:
            raise ValueError("Inconsistent genotype fields across variants.")
        genotype_fields = format_str_set.pop().split(":")

        self.field_count: int = len(genotype_fields)
        self.dosage_field_index: int = genotype_fields.index("DS")

        example_line = self.example_lines[0]
        tokens = example_line.split(maxsplit=len(self.mandatory_columns))

        token = tokens[-1].replace("\n", "\t")
        self.column_count = self.metadata_column_count + token.count("\t")

        self.float_reader: Any = None

    def set_samples(self, samples: set[str]) -> None:
        super().set_samples(samples)
        self.column_indices = (self.metadata_column_count + self.sample_indices).astype(
            np.uint32
        )

    def __exit__(self, *args: Any) -> None:
        super().__exit__(*args)
        self.float_reader = None

    def read(
        self,
        dosages: npt.NDArray,
    ) -> None:
        if self.file_handle is None:
            raise ValueError("Cannot read from a closed file")

        if dosages.size == 0:
            return  # Nothing to do.

        if dosages.shape[1] != self.sample_count:
            raise ValueError(
                "The output array does not match the number of samples "
                f"({dosages.shape[1]} != {self.sample_count})"
            )

        from ._read import create_float_reader, run_float_reader

        if self.float_reader is None:
            self.float_reader = create_float_reader(
                self.file_handle.fileno(),
                self.header_length,
                self.column_count,
                self.column_indices,
                self.dosage_field_index,
            )

        run_float_reader(
            dosages.transpose(),
            self.float_reader,
            self.variant_indices,
        )
