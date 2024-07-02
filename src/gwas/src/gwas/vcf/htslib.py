from pathlib import Path

import numpy as np
import pandas as pd
from numpy import typing as npt

# from ._htslib import read_vcf_records, read
from gwas.vcf._htslib import read, read_vcf_records  # from setup.py

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
    # "format_str",
]


class HTSLIBVCFFile(VCFFile):
    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)
        self.vcf_variants: pd.DataFrame
        self.variant_indices: npt.NDArray[np.uint32]
        self.create_dataframe()

    def create_dataframe(self):
        variants = read_vcf_records(str(self.file_path))
        self.vcf_variants = pd.DataFrame(variants, variant_columns)
        self.variant_indices = np.arange(self.vcf_variant_count, dtype=np.uint32)

    def read(self, dosages: npt.NDArray[np.float64]) -> None:
        if dosages.size == 0:
            return
        # if dosages.shape[1] != self.sample_count:
        #     raise ValueError(
        #         "The output array does not match the number of samples "
        #         f"({dosages.shape[1]} != {self.sample_count})"
        #     )
        read(self.file_path, dosages, self.sample_indices)
