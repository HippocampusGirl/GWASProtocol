# -*- coding: utf-8 -*-

from __future__ import annotations

from abc import abstractmethod
from contextlib import AbstractContextManager
from enum import Enum, auto
from functools import partial
from pathlib import Path
from typing import ClassVar, Literal, Self, overload

import numpy as np
import pandas as pd
from numpy import typing as npt
from tqdm.auto import tqdm

from ..compression.pipe import CompressedTextReader, load_from_cache, save_to_cache
from ..log import logger
from ..utils import IterationOrder, make_pool_or_null_context
from .variant import Variant


class Engine(Enum):
    python = auto()
    cpp = auto()
    cyvcf2 = auto()


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


class VCFFile(AbstractContextManager):
    chromosome: int | str

    vcf_samples: list[str]  # All samples in the file
    vcf_variants: pd.DataFrame  # All variants in the file

    samples: list[str]  # Samples selected for reading

    sample_indices: npt.NDArray[np.uint32]  # Indices of the samples selected for reading
    variant_indices: npt.NDArray[np.uint32]  # Variants selected for reading

    minor_allele_frequency_cutoff: float
    r_squared_cutoff: float

    allele_frequency_columns: list[str]

    def __init__(self) -> None:
        self.allele_frequency_columns = [
            "minor_allele_frequency",
            "alternate_allele_frequency",
        ]

        self.minor_allele_frequency_cutoff = -np.inf
        self.r_squared_cutoff = -np.inf

    @abstractmethod
    def read(
        self,
        dosages: npt.NDArray,
    ) -> None:
        """
        Given a target array of the correct shape, read the `DS` field from the file
        for the samples defined by `sample_indices` and the variants defined by
        `variant_indices`
        """
        ...

    @property
    def sample_count(self) -> int:
        """
        How many samples are selected for reading
        """
        return len(self.samples)

    @property
    def vcf_variant_count(self) -> int:
        """
        How many variants are available in the file
        """
        return len(self.vcf_variants.index)

    @property
    def variant_count(self) -> int:
        """
        How many variants are selected for reading
        """
        return self.variant_indices.size

    @property
    def variants(self) -> pd.DataFrame:
        """
        The metadata for the variants that are selected for reading
        """
        return self.vcf_variants.iloc[self.variant_indices, :]

    def reset_variants(self) -> None:
        """
        Reset `variant_indices` to read all variants
        """
        self.set_variants_from_cutoffs()

    def set_variants_from_cutoffs(
        self,
        minor_allele_frequency_cutoff: float = -np.inf,
        r_squared_cutoff: float = -np.inf,
    ):
        """
        Set `variant_indices` based on supplied cutoffs
        """

        def greater_or_close(series: pd.Series, cutoff: float) -> npt.NDArray[np.bool_]:
            value = np.asanyarray(series.values)
            return (value >= cutoff) | np.isclose(value, cutoff) | np.isnan(value)

        allele_frequency_frame = self.vcf_variants[self.allele_frequency_columns].copy()
        allele_frequencies = allele_frequency_frame
        is_major = allele_frequencies > 0.5
        allele_frequencies[is_major] = 1 - allele_frequencies[is_major]
        max_allele_frequency = allele_frequencies.max(axis="columns")
        self.variant_indices = np.flatnonzero(
            greater_or_close(
                max_allele_frequency,
                minor_allele_frequency_cutoff,
            )
            & greater_or_close(self.vcf_variants.r_squared, r_squared_cutoff)
        ).astype(np.uint32)

        logger.debug(
            f"After filtering with "
            f'"minor_allele_frequency >= {minor_allele_frequency_cutoff}" and '
            f'"r_squared >= {r_squared_cutoff}" '
            f"{self.variant_indices.size} of {self.vcf_variant_count} variants remain."
        )
        self.minor_allele_frequency_cutoff = minor_allele_frequency_cutoff
        self.r_squared_cutoff = r_squared_cutoff

    def set_samples(self, samples: set[str]) -> None:
        """
        Set `samples` and `sample_indices` based on list of sample IDs
        """

        self.sample_indices = np.fromiter(
            (i for i, sample in enumerate(self.vcf_samples) if sample in samples),
            dtype=np.uint32,
        )
        self.samples = [sample for sample in self.vcf_samples if sample in samples]

    @overload
    @staticmethod
    def from_path(
        file_path: Path | str,
        samples: set[str] | None = None,
        engine: Literal[Engine.cpp, Engine.python] = Engine.cpp,
    ) -> VCFFileReader: ...

    @overload
    @staticmethod
    def from_path(
        file_path: Path | str,
        samples: set[str] | None = None,
        engine: Engine = Engine.cyvcf2,
    ) -> VCFFile: ...

    @staticmethod
    def from_path(
        file_path: Path | str,
        samples: set[str] | None = None,
        engine: Engine = Engine.cpp,
    ) -> VCFFile:
        if engine == Engine.python:
            from .python import PyVCFFile

            vcf_file: VCFFile = PyVCFFile(file_path)
        elif engine == Engine.cpp:
            from .cpp import CppVCFFile

            vcf_file = CppVCFFile(file_path)
        elif engine == Engine.cyvcf2:
            from .cyvcf2 import CyVCF2VCFFile

            vcf_file = CyVCF2VCFFile(file_path)
        else:
            raise ValueError("Unsupported engine type: {}".format(engine))

        if samples is not None:
            vcf_file.set_samples(samples)

        return vcf_file

    @staticmethod
    def make_data_frame(vcf_variants: list[Variant]) -> pd.DataFrame:
        data_frame = pd.DataFrame(vcf_variants, columns=variant_columns)

        data_frame["position"] = data_frame["position"].astype(np.uint32)

        for column in [
            "chromosome_int",
            "reference_allele",
            "alternate_allele",
            "format_str",
        ]:
            data_frame[column] = data_frame[column].astype("category")

        return data_frame


class VCFFileReader(VCFFile, CompressedTextReader):
    mandatory_columns: ClassVar[tuple[str, ...]] = (
        "CHROM",
        "POS",
        "ID",
        "REF",
        "ALT",
        "QUAL",
        "FILTER",
        "INFO",
        "FORMAT",
    )

    chromosome_column_index: ClassVar[int] = mandatory_columns.index("CHROM")
    position_column_index: ClassVar[int] = mandatory_columns.index("POS")
    reference_allele_column_index: ClassVar[int] = mandatory_columns.index("REF")
    alternate_allele_column_index: ClassVar[int] = mandatory_columns.index("ALT")
    info_column_index: ClassVar[int] = mandatory_columns.index("INFO")
    format_column_index: ClassVar[int] = mandatory_columns.index("FORMAT")

    metadata_column_indices: ClassVar[npt.NDArray[np.uint32]] = np.array(
        [
            chromosome_column_index,
            position_column_index,
            reference_allele_column_index,
            alternate_allele_column_index,
            info_column_index,
            format_column_index,
        ],
        dtype=np.uint32,
    )

    def __init__(self, file_path: Path | str) -> None:
        super(CompressedTextReader, self).__init__(file_path)

        # Read header information and example line.
        self.header_length: int = 0
        column_names_line: str | None = None
        example_lines: list[str] = list()
        with self as file_handle:
            for line in file_handle:
                if line.startswith("#"):
                    self.header_length += len(line)
                    if not line.startswith("##"):
                        column_names_line = line
                    continue
                if len(example_lines) < 2:
                    example_lines.append(line)
                    continue
                break

        if not isinstance(column_names_line, str):
            raise ValueError(f"Could not find column names in {file_path}")

        if not len(example_lines) > 0:
            raise ValueError(f"Could not find example lines in {file_path}")

        self.example_lines = example_lines

        # Extract and check column names.
        self.columns = column_names_line.strip().removeprefix("#").split()
        if tuple(self.columns[: len(self.mandatory_columns)]) != self.mandatory_columns:
            raise ValueError

        self.metadata_column_count = len(self.mandatory_columns)
        self.vcf_samples: list[str] = self.columns[len(self.mandatory_columns) :]
        self.set_samples(set(self.vcf_samples))

    def save_to_cache(self, cache_path: Path) -> None:
        save_to_cache(cache_path, self.cache_key(self.file_path), self)

    @staticmethod
    def cache_key(vcf_path: Path) -> str:
        stem = vcf_path.name.split(".")[0]
        cache_key = f"{stem}.vcf-metadata"
        return cache_key

    @classmethod
    def load_from_cache(cls, cache_path: Path, vcf_path: Path) -> Self:
        return load_from_cache(cache_path, cls.cache_key(vcf_path))


def load_vcf(
    cache_path: Path,
    vcf_path: Path,
    engine: Engine = Engine.cpp,
) -> VCFFile:
    if engine == Engine.cyvcf2:
        return VCFFile.from_path(vcf_path, engine=engine)
    vcf_file: VCFFileReader | None = VCFFileReader.load_from_cache(cache_path, vcf_path)
    if vcf_file is None:
        vcf_file = VCFFileReader.from_path(vcf_path, engine=engine)
        vcf_file.save_to_cache(cache_path)
    else:
        logger.debug(
            f'Cached VCF file metadata for "{VCFFileReader.cache_key(vcf_path)}"'
        )
    return vcf_file


def calc_vcf(
    vcf_paths: list[Path],
    cache_path: Path,
    num_threads: int = 1,
    engine: Engine = Engine.cpp,
) -> list[VCFFile]:
    pool, iterator = make_pool_or_null_context(
        vcf_paths,
        partial(load_vcf, cache_path, engine=engine),
        num_threads=num_threads,
        iteration_order=IterationOrder.UNORDERED,
    )
    with pool:
        vcf_files = list(
            tqdm(
                iterator,
                total=len(vcf_paths),
                unit="files",
                desc="loading vcf metadata",
            )
        )
    return vcf_files
