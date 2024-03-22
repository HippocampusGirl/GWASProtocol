# -*- coding: utf-8 -*-
from multiprocessing import Pool
from pathlib import Path
from typing import List

import blosc2
import pandas as pd
from tqdm import tqdm

from gwas.plots.helpers import chi2_pvalue, find_phenotype_index, load_metadata


def create_dataframe_single_chr(args: List[tuple[Path, Path, str]]) -> pd.DataFrame:
    """
    Generates a pandas DataFrame for genetic variants of a single chromosome.

    Args:
        args: A tuple containing three elements:
            - score_path (Path): The path to the chromosome's score file.
            - axis_metadata_path (Path): The path to the chromosome's axis metadata file.
            - phenotype_label (str): The label of the phenotype to calculate p-values for.

    Returns:
        pd.DataFrame: A DataFrame with columns ['SNP', 'CHR', 'BP', 'P'], containing the
        SNP identifier, chromosome number, base pair position, and p-value for each variant.

    Raises:
        ValueError: If the length of the b2 array does not match the length of the metadata.
    """
    # score_path: str | Path, axis_metadata_path: str | Path, phenotype_label: str old args
    score_path, axis_metadata_path, phenotype_label = args

    axis_metadata = load_metadata(axis_metadata_path)
    metadata = pd.DataFrame(axis_metadata[0])
    phenotypes_list = axis_metadata[1]

    b2_array = blosc2.open(urlpath=score_path)

    if b2_array.shape[0] != len(metadata.index):
        raise ValueError(
            "The length of b2_array does not match the length of the metadata."
        )

    u_stat_idx, v_stat_idx = find_phenotype_index(
        phenotype_label=phenotype_label, phenotypes_list=phenotypes_list
    )

    variant_count = len(metadata.index)

    u_stat, v_stat = (
        b2_array[:, u_stat_idx],
        b2_array[:, v_stat_idx],
    )  # possible solution b2_array[:, u_stat_idx:v_stat_idx] to only index array once
    # p = phenotype_indices(u_stat_idx=u_stat, v_stat_idx=v_stat)

    chr_data = {"SNP": [], "CHR": [], "BP": [], "P": []}

    for i in tqdm(range(variant_count), desc="Processing Variant"):
        # indices finden für v_stat u_stat idx
        chrom = metadata.chromosome_int.iloc[i]
        pos = metadata.position.iloc[i]

        p_value = chi2_pvalue(ustat=u_stat[i], vstat=v_stat[i])

        chr_data["SNP"].append("rs0000000")
        chr_data["CHR"].append(chrom)
        chr_data["BP"].append(pos)
        chr_data["P"].append(p_value)

    return pd.DataFrame(chr_data)


def create_dataframe_all_chr(
    score_path: str | Path, metadata_path: str | Path, label: str, cpu_count: int
) -> pd.DataFrame:
    """Generate dataframes with information needed for manhattan plots / qq plots for each chromosome which will be concatenated together.
    Uses multiprocessing to utilize cpu and share the workload.
    """
    # score_files = list(Path(score_path).glob('*.b2array'))

    CHROMOSOMES = list(range(1, 23)) + ["X"]
    tasks = [
        (
            Path(score_path) / f"chr{chromosome}.score.b2array",
            Path(metadata_path) / f"chr{chromosome}.score.axis-metadata.pkl.zst",
            label,
        )
        for chromosome in CHROMOSOMES
    ]
    dfs = []

    with Pool(processes=cpu_count) as pool:
        dfs = pool.map(create_dataframe_single_chr, tasks)

    # for chromosome in tqdm(CHROMOSOMES, desc="Processing Chromosomes"):
    #     df = create_dataframe_single_chr(Path(score_path) / f'chr{chromosome}.score.b2array',
    #                                  Path(metadata_path) / f'chr{chromosome}.score.axis-metadata.pkl.zst',
    #                                  phenotype_label=label)
    #     dfs.append(df)

    final_df = pd.concat(dfs)
    return final_df
