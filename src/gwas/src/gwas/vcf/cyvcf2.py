# -*- coding: utf-8 -*-
from pathlib import Path
from types import TracebackType

import numpy as np
import pandas as pd
from cyvcf2 import VCF
from numpy import typing as npt

from .base import VCFFile

variant_columns = [
    "chromosome_int",
    "position",
    "reference_allele",
    "alternate_allele",
    "is_imputed",
    "alternate_allele_frequency",
    "minor_allele_frequency",
    "r_squared",
    "format_str",
]


class CyVCF2VCFFile(VCFFile):
    def __init__(self, file_path: str | Path) -> None:
        super().__init__()
        self.vcf = VCF(file_path)
        self.vcf_samples = list(self.vcf.samples)

        self.all_variants = [v for v in self.vcf]
        self.vcf_variants = self.make_data_frame(self.vcf)  # all variants in the file
        self.samples: list[str]  # Samples selected for reading

        self.sample_indices = np.array([], dtype=np.uint32)
        self.variant_indices = np.array([], dtype=np.uint32)

    def set_samples(self, samples: set[str]):
        self.samples = [sample for sample in self.samples if sample in samples]
        self.sample_indices = np.array(
            [self.vcf_samples.index(s) for s in self.samples if s in self.vcf_samples],
            dtype=np.uint32,
        )

    @staticmethod
    def make_data_frame(vcf: VCF) -> pd.DataFrame:
        # Convert VCF data from cyvcf2 to a DataFrame
        variants = []
        for variant in vcf:
            variants.append(
                [
                    variant.CHROM,
                    variant.POS,
                    variant.REF,
                    variant.ALT[0] if variant.ALT else "",
                    variant.INFO.get("IMPUTED", False),
                    variant.INFO.get("AF", float("nan")),
                    variant.INFO.get("MAF", float("nan")),
                    variant.INFO.get("RSQ", float("nan")),
                    variant.FORMAT,
                ]
            )
        return pd.DataFrame(variants, columns=variant_columns)

    def set_variants_from_cutoffs(
        self,
        minor_allele_frequency_cutoff: float = -np.inf,
        r_squared_cutoff: float = -np.inf,
    ):
        cutoff_indices = [
            i
            for i, v in enumerate(self.vcf)
            if float(v.INFO.get("MAF")) > minor_allele_frequency_cutoff
            and float(v.INFO.get("R2")) > r_squared_cutoff
        ]
        self.variant_indices = np.array(cutoff_indices, dtype=np.uint32)

    def read(self, dosages: npt.NDArray) -> None:
        for i, variant_idx in enumerate(self.variant_indices):
            variant = self.all_variants[variant_idx]
            dosages[i, :] = [
                variant.format("DS")[idx][0] for idx in self.sample_indices
            ]  # for loop rausnehmen?

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        return super().__exit__(exc_type, exc_value, traceback)
