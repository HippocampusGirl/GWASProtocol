# -*- coding: utf-8 -*-
import pickle
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import zstandard as zstd
from qmplot import manhattanplot
from scipy.stats.distributions import chi2

from gwas.compression.pipe import CompressedBytesReader


def chi2_pvalue(ustat: int | float, vstat: int | float) -> float:
    """Calculates the p-value from U-statistic and V-statistic using the chi-square test."""
    chi2_stat = (ustat**2) / vstat
    p_value = chi2.sf(chi2_stat, df=1)
    return p_value


def find_phenotype_index(phenotype_label: str, phenotypes_list: List) -> Tuple[int, int]:
    """
    Finds the indices of the U-statistic and V-statistic for a given phenotype label.
    Args:
        phenotype_label (str): The label of the phenotype to find indices for.
        phenotypes_list (List[str]): A list of phenotype descriptors.

    Returns:
        Tuple[int, int]: A tuple containing the indices of the U-statistic and V-statistic.

    Raises:
        ValueError: If either the U-statistic or V-statistic index cannot be found for the
        specified phenotype label.
    """
    u_stat_idx = None
    v_stat_idx = None
    # namedTuple für phenotype
    # to use phenotypes_list.index() method we need to build the complete string from the correct label using our included file-name.txt
    for index, phenotype in enumerate(phenotypes_list):
        if phenotype_label in phenotype:
            if "stat-u" in phenotype:
                u_stat_idx = index
            elif "stat-v" in phenotype:
                v_stat_idx = index

    if u_stat_idx is not None and v_stat_idx is not None:
        return u_stat_idx, v_stat_idx
    else:
        raise ValueError(
            f"Matching indices for '{phenotype_label}' not found. u_stat_idx: {u_stat_idx}, v_stat_idx: {v_stat_idx}"
        )


def load_metadata(metadata_path: Path):
    # with open(metadata_path, "rb") as f:
    with CompressedBytesReader(metadata_path) as f:
        decompressor = zstd.ZstdDecompressor()
        compressed_data = f.read()
        # decompressed_data = decompressor.decompress(compressed_data)
        stream_reader = decompressor.stream_reader(compressed_data)
        decompressed_data = stream_reader.read()
        stream_reader.close()
        metadata = pickle.loads(decompressed_data)
    return metadata


def filter_rois(csv_path: str):
    csv_df = pd.read_csv(csv_path)
    filtered_df = csv_df[csv_df["Brainnetome label (hypothesis)"].notna()]
    labels_list = filtered_df["Label"].tolist()
    return labels_list


def save_dataframe_as_pickle(
    dataframe: pd.DataFrame, directory: Path, label: str
) -> None:
    folder_name = "integramoods_gwas_dfs"
    folder_path = directory / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)

    filename = f"integramoods_gwas_{label}_df.pkl"
    file_path = folder_path / filename

    with file_path.open("wb") as file:
        pickle.dump(dataframe, file)


def generate_and_save_manhattan_plot(
    dataframe: pd.DataFrame, directory: Path, label: str
) -> None:
    folder_name = "integramoods_gwas_plots"
    folder_path = directory / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)

    filename = f"manhattanplot_{label}.png"
    file_path = folder_path / filename

    ax = manhattanplot(
        data=dataframe,
        genomewideline=1e-8,
        chrom="CHR",
        pos="BP",
        pv="P",
        snp="SNP",
        xticklabel_kws={"rotation": "vertical"},
    )
    plt.savefig(file_path)


def check_existing_files(directory: Path, label: str) -> bool:
    folder_plots = directory / "integramoods_gwas_plots"
    filename_plot = f"manhattanplot_{label}.png"
    plot_file_path = folder_plots / filename_plot

    folder_dfs = directory / "integramoods_gwas_dfs"
    filename_df = f"integramoods_gwas_{label}_df.pkl"
    df_file_path = folder_dfs / filename_df

    if plot_file_path.is_file() or df_file_path.is_file():
        return True
    return False
